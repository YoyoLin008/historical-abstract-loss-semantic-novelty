"""Revision-stage R3: saved original masks only; no draws or new classifications."""
import argparse, collections, csv, datetime, fcntl, hashlib, json, math, os, pathlib, platform, time
import numpy as np

COLLECTIONS = {
 'ACL': ('confirmation_v1','data/confirmation_v1/records.jsonl','configs/confirmation_seeds_v1.json',3799,1557,1800,1200),
 'Biomedical': ('pubmed_replication_v1','replication/data/records.jsonl','configs/pubmed_replication_seeds_v1.json',3391,1471,800,400),
}
ROLES = ('structured_or_random','matched_random_control')

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def write_json(path,value):
 path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def write_csv(path,rows):
 path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def stats(values):
 a=np.asarray(values,dtype=float)
 assert len(a)>1 and np.isfinite(a).all()
 return dict(n_masks=len(a),mean=float(a.mean()),assignment_sd=float(a.std(ddof=1)),p05=float(np.quantile(a,.05,method='linear')),p95=float(np.quantile(a,.95,method='linear')))

def condition(mechanism,role):
 return mechanism if role==ROLES[0] else mechanism+'_matched_control'

def load_collection(root,study,receipt_inputs=None):
 stage,record_rel,seed_rel,nh,nf,nb,nc=COLLECTIONS[study]
 root=pathlib.Path(root);d=root/'results'/stage
 def read(path,expected=None):
  path=pathlib.Path(path); digest=sha(path)
  if expected is not None: assert digest==expected, ('hash',str(path),digest,expected)
  if receipt_inputs is not None: receipt_inputs[str(path.relative_to(root))]=digest
  return json.loads(path.read_text())
 m=read(d/'run_manifest.json');assert m['status']=='complete'
 records_path=root/record_rel;digest=sha(records_path);assert digest==m['provenance']['inputs']['records']
 if receipt_inputs is not None:receipt_inputs[record_rel]=digest
 records=[json.loads(s) for s in records_path.read_text().splitlines() if s.strip()]
 byid={r['id']:r for r in records};assert len(byid)==len(records)
 histids=m['scope']['history_ids'];focalids=m['scope']['focal_ids']
 assert len(histids)==nh and len(focalids)==nf and len(set(histids+focalids))==nh+nf
 hist=[byid[i] for i in histids];focal=[byid[i] for i in focalids]
 assert histids==sorted(r['id'] for r in records if r['year'] in [2018,2019,2020])
 assert focalids==sorted(r['id'] for r in records if r['year']==2021)
 assert max(r['year'] for r in hist)<min(r['year'] for r in focal)
 b=read(d/'baseline.json',m['output_sha256']['baseline.json']); assert b['scope']==m['scope']
 seeds=read(root/seed_rel,m['provenance']['inputs']['seeds']);assert len(seeds['assignments'])==nb
 assert sum('control_seed' in r for r in seeds['assignments'])==nc
 if study=='ACL':assert all((r['topic']==0)==('translat' in r['title'].lower()) for r in hist)
 return dict(study=study,d=d,manifest=m,history=hist,focal=focal,seeds=seeds,baseline=b,inputs=receipt_inputs,root=root)

def saved_rows(c,source_only=False):
 for seed in c['seeds']['assignments']:
  if source_only and not(seed['mechanism']=='source' and seed['rate'] in [.25,.5]):continue
  path=c['d']/'raw'/(seed['assignment_id']+'.json')
  row=json.loads(path.read_text())
  digest=hashlib.sha256(json.dumps({k:v for k,v in row.items() if k!='content_sha256'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
  assert row['content_sha256']==digest,('content_hash',str(path))
  assert row['run_fingerprint']==c['manifest']['provenance']['run_fingerprint']
  assert all(row[k]==v for k,v in seed.items())
  if c['inputs'] is not None:c['inputs'][str(path.relative_to(c['root']))]=sha(path)
  yield row

def mask_from_arm(arm,history):
 ids=[r['id'] for r in history];missing=arm['missing_history_ids']
 assert len(set(missing))==len(missing) and set(missing)<=set(ids)
 mask=np.isin(ids,missing)
 assert int(mask.sum())==arm['missing_count']==len(missing)
 return mask

def validate_pair(row,history,masks):
 if ROLES[1] not in masks:return 0
 fields=row['control_matching'];assert fields==(['year'] if row['mechanism']=='source' else ['year','source'])
 cells=collections.defaultdict(list)
 for i,r in enumerate(history):cells[tuple(r[k] for k in fields)].append(i)
 assert len(cells)==len(row['control_cells'])
 for meta in row['control_cells']:
  key=tuple(meta['cell'][k] for k in fields);ix=cells[key]
  assert len(ix)==meta['total']
  a=int(masks[ROLES[0]][ix].sum());b=int(masks[ROLES[1]][ix].sum())
  assert a==b==meta['missing'],('matchedcell',row['assignment_id'],key,a,b)
 return len(cells)

def run(root,candidate):
 start=time.monotonic();candidate=pathlib.Path(candidate)
 assert (candidate/'06_private_audits/POST_HOC_ANALYSIS_SPEC.md').is_file()
 out=candidate/'outputs/revision_descriptive/mechanism';audit=candidate/'06_private_audits/components/mechanism';audit.mkdir(parents=True,exist_ok=True)
 inputs={};allrows=[];denoms=[];n_pairs=n_cells=n_arms=0
 with (audit/'intensity.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
  for study in COLLECTIONS:
   c=load_collection(root,study,inputs);hist=c['history'];preferred=c['manifest']['config']['mask_preferred_source']
   strata=[('source',s,np.array([r['source']==s for r in hist])) for s in c['manifest']['config']['sources']]
   if study=='ACL': strata.extend([('lexical_proxy','translat',np.array([r['topic']==0 for r in hist])),('lexical_proxy','nontranslat',np.array([r['topic']!=0 for r in hist]))])
   for kind,s,ix in strata:denoms.append(dict(study=study,stratum_type=kind,stratum=s,N_s=int(ix.sum()),preferred=(s==preferred if kind=='source' else s=='translat')))
   for row in saved_rows(c):
    masks={role:mask_from_arm(row[role],hist) for role in ROLES if role in row}
    nc=validate_pair(row,hist,masks);n_cells+=nc;n_pairs+=bool(nc)
    for role,mask in masks.items():
     n_arms+=1;total=int(mask.sum());assert total==math.floor(row['rate']*len(hist)+.5)
     arm=row[role]
     for key in ['source','year','topic']:
      for value,meta in arm['composition'][key].items():
       ix=np.array([str(r[key])==str(value) for r in hist]);assert int(ix.sum())==meta['total'] and int(mask[ix].sum())==meta['missing']
     for kind,s,ix in strata:
      if kind=='lexical_proxy' and row['mechanism']!='topic':continue
      count=int(mask[ix].sum());n=int(ix.sum())
      allrows.append(dict(study=study,assignment_id=row['assignment_id'],arm_id=arm['assignment_id'],condition=condition(row['mechanism'],role),mechanism=row['mechanism'],rate=row['rate'],arm_role=role,stratum_type=kind,stratum=s,N_s=n,total_masked=total,c_m_s=count,within_stratum_fraction=count/n,masked_set_composition=count/total))
   write_json(audit/'intensity_checkpoint.json',dict(status='running',completed_collection=study,arms=n_arms,pairs=n_pairs))
  groups=collections.defaultdict(list)
  for r in allrows:groups[(r['study'],r['condition'],r['rate'],r['arm_role'],r['stratum_type'],r['stratum'],r['N_s'])].append(r)
  summaries=[]
  for (study,cond,rate,role,kind,s,n),rows in sorted(groups.items()):
   assert len(rows)==200
   item=dict(study=study,condition=cond,rate=rate,arm_role=role,stratum_type=kind,stratum=s,N_s=n,n_masks=len(rows))
   for metric in ['c_m_s','within_stratum_fraction','masked_set_composition']:
    item.update({metric+'_'+k:v for k,v in stats([r[metric] for r in rows]).items() if k!='n_masks'})
   summaries.append(item)
  assert n_arms==4200 and n_pairs==1600
  write_csv(out/'intensity_per_assignment.csv',allrows);write_csv(out/'intensity_summary.csv',summaries);write_csv(out/'history_strata.csv',denoms)
  lines=[r'\begin{longtable}{llrrrrr}',r'\caption{Realized historical substitution within source strata (percent). All entries use the original 200 assignments. Parentheses give assignment SD. Full assignment percentiles and masked-set composition are reported in the numerical resource.}\label{tab:revision_intensity}\\',r'\toprule Collection & Condition & Loss & Preferred & Other & $N_P$ & $N_O$ \\ \midrule',r'\endfirsthead',r'\toprule Collection & Condition & Loss & Preferred & Other & $N_P$ & $N_O$ \\ \midrule',r'\endhead']
  short={'random':'Random','source':'Source','source_matched_control':'Source control','topic':'Lexical','topic_matched_control':'Lexical control'}
  for study in COLLECTIONS:
   preferred='ACL' if study=='ACL' else 'PLOSCB'
   for cond,rate in sorted({(r['condition'],r['rate']) for r in summaries if r['study']==study and r['stratum_type']=='source'}):
    pair=[r for r in summaries if r['study']==study and r['condition']==cond and r['rate']==rate and r['stratum_type']=='source'];pair.sort(key=lambda r:r['stratum']!=preferred)
    def fmt(r):return f"{100*r['within_stratum_fraction_mean']:.1f} ({100*r['within_stratum_fraction_assignment_sd']:.1f})"
    lines.append(f"{study} & {short[cond]} & {100*rate:.0f} & {fmt(pair[0])} & {fmt(pair[1])} & {pair[0]['N_s']} & {pair[1]['N_s']} "+r'\\')
  lines.extend([r'\bottomrule',r'\end{longtable}'])
  (out/'intensity_table.tex').write_text('\n'.join(lines)+'\n')
  lexlines=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{Realized substitution in the ACL lexical-proxy strata (percent). Entries are means (assignment SDs) over 200 masks. The preferred stratum comprises 289 original titles matching the case-insensitive substring \texttt{translat}; the other stratum comprises 3,510 histories.}\label{tab:revision_lexical_intensity}',r'\begin{tabular}{lrrr}',r'\toprule Condition & Loss & Preferred & Other \\ \midrule']
  for cond,rate in sorted({(r['condition'],r['rate']) for r in summaries if r['stratum_type']=='lexical_proxy'}):
   pair=[r for r in summaries if r['condition']==cond and r['rate']==rate and r['stratum_type']=='lexical_proxy'];pair.sort(key=lambda r:r['stratum']!='translat')
   def fmt(r):return f"{100*r['within_stratum_fraction_mean']:.1f} ({100*r['within_stratum_fraction_assignment_sd']:.1f})"
   lexlines.append(f"{short[cond]} & {100*rate:.0f} & {fmt(pair[0])} & {fmt(pair[1])} "+r'\\')
  lexlines.extend([r'\bottomrule\end{tabular}',r'\end{table}']);(out/'intensity_lexical_table.tex').write_text('\n'.join(lexlines)+'\n')
  write_json(audit/'intensity_inputs.json',inputs)
  report=dict(status='VERIFIED',analysis_label='revision-stage descriptive summaries of frozen assignments',method='Authoritative saved missing_history_ids; no masks reconstructed or generated',arms=n_arms,base_assignments=2600,matched_pairs=n_pairs,matched_cells=n_cells,summary_rows=len(summaries),per_assignment_strata_rows=len(allrows),quantiles="numpy.quantile(method='linear')",sd='sample assignment SD, ddof=1',elapsed_seconds=time.monotonic()-start,python=platform.python_version(),numpy=np.__version__,input_manifest_sha256=sha(audit/'intensity_inputs.json'),script_sha256=sha(__file__),output_sha256={p.name:sha(p) for p in out.glob('intensity*')},limitations='Synthetic intervention intensity, not observed historical missingness; rate preference is not marginal odds.')
  write_json(audit/'intensity_verification.json',report);print(json.dumps(report,indent=2));return report

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--candidate',type=pathlib.Path,required=True);a=p.parse_args();run(a.root,a.candidate)
