"""Fixed-scope post hoc exact-neighbor diagnostic using existing SPECTER caches.
Run --benchmark-only first. NPZ focal-level changes are PRIVATE, pending release review.
"""
import argparse, collections, fcntl, json, math, os, pathlib, platform, resource, time
import numpy as np
from revision_mask_intensity import COLLECTIONS, ROLES, sha, write_json, write_csv, stats, condition, load_collection, saved_rows, mask_from_arm, validate_pair

BLOCK=128

def exact_winners(full,title,mask,history_ids,with_gap=False):
 """Exact float32 maxima; lexical-ID identity tie rule for this diagnostic only."""
 assert full.shape==title.shape and full.dtype==title.dtype==np.float32
 lexical_rank=np.argsort(np.argsort(np.asarray(history_ids),kind='stable'),kind='stable')
 n=len(full);scores=np.empty(n,dtype=np.float32);winners=np.empty(n,dtype=np.int32);ties=np.zeros(n,bool);gap=np.zeros(n,float)
 for start in range(0,n,BLOCK):
  stop=min(start+BLOCK,n)
  s=np.where(mask[None,:],title[start:stop],full[start:stop])
  assert np.isfinite(s).all()
  mx=s.max(axis=1);win=s.argmax(axis=1);eq=s==mx[:,None];multi=eq.sum(axis=1)>1
  for j in np.flatnonzero(multi):
   ix=np.flatnonzero(eq[j]);win[j]=ix[np.argmin(lexical_rank[ix])]
  winners[start:stop]=win;ties[start:stop]=multi;scores[start:stop]=np.float32(1)-mx
  if with_gap:
   top2=np.partition(s,-2,axis=1)[:,-2:];gap[start:stop]=top2[:,1].astype(float)-top2[:,0].astype(float)
 return scores,winners,ties,gap

def selected(scores,ids):
 assert np.isfinite(scores).all() and len(set(ids))==len(ids)
 ix=np.lexsort((np.asarray(ids),-np.asarray(scores,dtype=float)))[:math.ceil(.1*len(ids))]
 out=np.zeros(len(ids),bool);out[ix]=True;return out

def source_accounting(delta,exposed,switched,title,selected0,lost,subset):
 """Every contribution has the fixed source denominator; empty conditional means undefined."""
 n=int(subset.sum());assert n>0;cells=[]
 for e in [False,True]:
  for s in [False,True]:
   for t in [False,True]:
    cell=subset&(exposed==e)&(switched==s)&(title==t);nc=int(cell.sum());sel=int((cell&selected0).sum());loss=int((cell&lost).sum())
    cells.append(dict(baseline_winner_masked=int(e),identity_switched=int(s),winner_title_only=int(t),fixed_denominator=n,cell_count=nc,cell_fraction=nc/n,shift_contribution_sd=float(delta[cell].sum()/n),conditional_shift_sd=float(delta[cell].mean()) if nc else None,baseline_selected_in_cell=sel,selection_losses_in_cell=loss,conditional_selected_loss_fraction=loss/sel if sel else None))
 assert abs(sum(r['shift_contribution_sd'] for r in cells)-float(delta[subset].mean()))<2e-15
 assert sum(r['cell_count'] for r in cells)==n
 assert sum(r['selection_losses_in_cell'] for r in cells)==int((lost&subset).sum())
 return cells

def caches(c,inputs):
 out=[]
 for name in ['full_full','full_title']:
  path=c['d']/'similarity_cache'/(name+'.npy');a=np.load(path,mmap_mode='r',allow_pickle=False)
  assert a.shape==(len(c['focal']),len(c['history'])) and a.dtype==np.float32
  inputs[str(path.relative_to(c['root']))]=sha(path);out.append(a)
 return out

def baseline(c,full,title):
 ids=[r['id'] for r in c['history']];scores,win,tie,gap=exact_winners(full,title,np.zeros(len(ids),bool),ids,True)
 original=np.asarray(c['baseline']['evaluations']['mixed']['novelty_scores'],dtype=float)
 assert np.array_equal(scores.astype(float),original),'baseline score cache mismatch'
 assert np.array_equal(original,np.asarray(c['baseline']['evaluations']['restored']['novelty_scores']))
 return original,win,tie,gap

def verify_arm(c,row,role,mask,full,title,base,bwin,selected0):
 scores,win,ties,_=exact_winners(full,title,mask,[r['id'] for r in c['history']])
 original=np.asarray(row[role]['evaluations']['mixed']['novelty_scores'],dtype=float)
 assert np.array_equal(scores.astype(float),original),('saved score mismatch',row['assignment_id'],role)
 delta=original-base;exposed=mask[bwin];switched=win!=bwin;uses_title=mask[win]
 assert np.all(delta[~exposed]<=0),('unmasked baseline winner increased',row['assignment_id'],role)
 lost=selected0&~selected(original,[r['id'] for r in c['focal']])
 return delta,exposed,switched,uses_title,lost,win,ties

def benchmark(root,candidate):
 start=time.monotonic();candidate=pathlib.Path(candidate);audit=candidate/'06_private_audits/components/mechanism';audit.mkdir(parents=True,exist_ok=True)
 inputs={};rows=[]
 for study in COLLECTIONS:
  init=time.monotonic();c=load_collection(root,study,inputs);full,title=caches(c,inputs);b,bwin,btie,gap=baseline(c,full,title);s0=b.std(ddof=0);selected0=selected(b,[r['id'] for r in c['focal']])
  row=next(saved_rows(c,True));masks={role:mask_from_arm(row[role],c['history']) for role in ROLES};validate_pair(row,c['history'],masks)
  bt=time.monotonic()
  for role in ROLES:
   delta,exposed,switched,uses_title,lost,win,ties=verify_arm(c,row,role,masks[role],full,title,b,bwin,selected0)
   for source in ['ALL']+c['manifest']['config']['sources']:
    subset=np.ones(len(b),bool) if source=='ALL' else np.array([r['source']==source for r in c['focal']]);source_accounting(delta/s0,exposed,switched,uses_title,selected0,lost,subset)
  pair_seconds=time.monotonic()-bt
  rows.append(dict(study=study,assignment_id=row['assignment_id'],selection_rule='first source-preferred 25/50 assignment in frozen seed-manifest ordering',mask_pair_seconds=pair_seconds,setup_and_pair_seconds=time.monotonic()-init,projected_400_pair_compute_seconds=400*pair_seconds,conservative_4x_compute_seconds=1600*pair_seconds,cache_bytes=full.nbytes+title.nbytes,temporary_similarity_block_bytes=BLOCK*full.shape[1]*4,exact_baseline_and_both_arm_scores=True))
 report=dict(status='BENCHMARK_VERIFIED',utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),benchmarks=rows,projected_compute_seconds=sum(r['projected_400_pair_compute_seconds'] for r in rows),conservative_4x_compute_seconds=sum(r['conservative_4x_compute_seconds'] for r in rows),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,elapsed_seconds=time.monotonic()-start,limit_seconds=3600,allow_full_run=sum(r['conservative_4x_compute_seconds'] for r in rows)<3600,input_sha256=inputs,script_sha256=sha(__file__))
 write_json(audit/'neighbor_benchmark.json',report);print(json.dumps(report,indent=2));return report

def aggregate(rows,keys,metrics):
 groups=collections.defaultdict(list)
 for r in rows:groups[tuple(r[k] for k in keys)].append(r)
 out=[]
 for key,rr in sorted(groups.items()):
  assert len(rr)==200
  item=dict(zip(keys,key));item['n_masks']=len(rr)
  for metric in metrics:
   values=[r[metric] for r in rr if r[metric] is not None]
   item[metric+'_defined_masks']=len(values)
   if len(values)>1:item.update({metric+'_'+k:v for k,v in stats(values).items() if k!='n_masks'})
   elif len(values)==1:item.update({metric+'_mean':values[0],metric+'_assignment_sd':None,metric+'_p05':values[0],metric+'_p95':values[0]})
   else:item.update({metric+'_'+k:None for k in ['mean','assignment_sd','p05','p95']})
  out.append(item)
 return out


def verify_r1_bridge(candidate):
 import csv
 candidate=pathlib.Path(candidate);out=candidate/'outputs/revision_descriptive/mechanism';audit=candidate/'06_private_audits/components/mechanism'
 r1path=candidate/'outputs/revision_descriptive/core/source_shifts_per_mask.csv'
 r1={(r['study'],r['assignment_id']):r for r in csv.DictReader(r1path.open())}
 rows=list(csv.DictReader((out/'neighbor_per_assignment.csv').open()));checks=[]
 for r in rows:
  key='mean_drift_sd' if r['focal_source']=='ALL' else 'source_A_shift_sd' if r['focal_source'] in ['ACL','PLOSCB'] else 'source_B_shift_sd'
  expected=float(r1[(r['study'],r['arm_id'])][key]);actual=float(r['mean_shift_sd']);error=abs(actual-expected)
  assert error<2e-13,('R1 shift mismatch',r['study'],r['arm_id'],key,error)
  checks.append(dict(study=r['study'],arm_id=r['arm_id'],source=r['focal_source'],quantity=key,direct_neighbor_shift=actual,R1_reconstructed_shift=expected,absolute_error=error))
 assert len(checks)==4800
 write_csv(audit/'neighbor_R1_bridge.csv',checks)
 export_checks=0;max_export_error=0.
 for study in COLLECTIONS:
  private=audit/'neighbor_records'/study;meta=json.loads((private/'README.json').read_text());focal=list(csv.DictReader((private/'focal_index.csv').open()));assign=list(csv.DictReader((private/'assignment_index.csv').open()));data=np.load(private/'neighbor_records.npz',allow_pickle=False)
  assert all(data[k].shape==(800,len(focal)) for k in data.files)
  assert all(not data[k].dtype.hasobject for k in data.files)
  assert np.array_equal(data['score_change_raw']/meta['baseline_sd'],data['score_change_sd'])
  lookup={(r['arm_id'],r['focal_source']):r for r in rows if r['study']==study}
  for i,arm in enumerate(assign):
   assert int(arm['arm_index'])==i
   for source in ['ALL']+(['ACL','EMNLP'] if study=='ACL' else ['PLOSCB','PLOSGEN']):
    ix=np.ones(len(focal),bool) if source=='ALL' else np.array([r['focal_source']==source for r in focal]);actual=float(data['score_change_sd'][i,ix].mean());target=float(lookup[arm['arm_id'],source]['mean_shift_sd']);err=abs(actual-target);assert err<2e-15;max_export_error=max(max_export_error,err);export_checks+=1
 report=dict(status='VERIFIED',R1_bridge_checks=len(checks),maximum_R1_error=max(r['absolute_error'] for r in checks),NPZ_reload_source_mean_checks=export_checks,maximum_NPZ_reload_mean_error=max_export_error,NPZ_object_arrays=False,R1_per_mask_sha256=sha(r1path),bridge_csv_sha256=sha(audit/'neighbor_R1_bridge.csv'))
 write_json(audit/'neighbor_R1_and_export_verification.json',report)
 return report

def run(root,candidate):
 candidate=pathlib.Path(candidate);audit=candidate/'06_private_audits/components/mechanism';out=candidate/'outputs/revision_descriptive/mechanism'
 assert (candidate/'06_private_audits/POST_HOC_ANALYSIS_SPEC.md').is_file()
 bench=json.loads((audit/'neighbor_benchmark.json').read_text());assert bench['allow_full_run'] and bench['script_sha256']==sha(__file__)
 inputs={};maskrows=[];transitions=[];arm_total=pair_total=focal_total=tie_total=0;start=time.monotonic()
 with (audit/'neighbors.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
  for study in COLLECTIONS:
   c=load_collection(root,study,inputs);full,title=caches(c,inputs);b,bwin,btie,gap=baseline(c,full,title);s0=float(b.std(ddof=0));focalids=[r['id'] for r in c['focal']];histids=[r['id'] for r in c['history']];hs=np.array([r['source'] for r in c['history']]);fs=np.array([r['source'] for r in c['focal']]);selected0=selected(b,focalids)
   assert all(inputs[k]==v for k,v in bench['input_sha256'].items() if k in inputs)
   private=audit/'neighbor_records'/study;private.mkdir(parents=True,exist_ok=True)
   write_csv(private/'focal_index.csv',[dict(focal_index=i,focal_id=id,focal_source=fs[i],baseline_winner_index=int(bwin[i]),baseline_winner_id=histids[bwin[i]],baseline_winner_source=hs[bwin[i]],baseline_top_decile=int(selected0[i]),baseline_exact_max_tie=int(btie[i]),baseline_best_second_gap=float(gap[i])) for i,id in enumerate(focalids)])
   write_csv(private/'history_index.csv',[dict(history_index=i,history_id=id,history_source=hs[i]) for i,id in enumerate(histids)])
   write_json(private/'README.json',dict(status='PRIVATE_DIAGNOSTIC_RECORDS_DO_NOT_UPLOAD_PENDING_AUTHOR_REVIEW',axes='NPZ rows follow assignment_index.csv; columns follow focal_index.csv; winning-history indices resolve through history_index.csv',score_change='float64 subtraction of exactly checked original float32 mixed/baseline scores, divided by original float64 population SD where specified',baseline_sd=s0,identity_tie_rule='lexical historical ID, for this new diagnostic only; original scorer stored maxima without historical identities',privacy='No titles, abstracts or embeddings. Per-focal scalar changes are gated with scalar exports.'))
   shape=(800,len(b));records={k:np.empty(shape,dtype=dtype) for k,dtype in [('winner_index','int32'),('baseline_winner_masked','bool'),('identity_switched','bool'),('winner_title_only','bool'),('selection_lost','bool'),('masked_exact_max_tie','bool'),('score_change_raw','float64'),('score_change_sd','float64')]};assign=[];arm_i=0
   for row in saved_rows(c,True):
    if time.monotonic()-start+bench['elapsed_seconds']>3600:
     write_json(audit/'neighbor_verification.json',dict(status='BLOCKED_RESOURCE',reason='One-hour diagnostic cap reached',arms=arm_total,elapsed_seconds=time.monotonic()-start));return
    masks={role:mask_from_arm(row[role],c['history']) for role in ROLES};validate_pair(row,c['history'],masks);pair_total+=1
    for role in ROLES:
     delta,exposed,switched,uses_title,lost,win,ties=verify_arm(c,row,role,masks[role],full,title,b,bwin,selected0);d=delta/s0;cond=condition(row['mechanism'],role)
     meta=dict(study=study,assignment_id=row['assignment_id'],arm_id=row[role]['assignment_id'],condition=cond,rate=row['rate'],arm_role=role)
     assign.append(dict(arm_index=arm_i,**meta))
     for k,v in dict(winner_index=win,baseline_winner_masked=exposed,identity_switched=switched,winner_title_only=uses_title,selection_lost=lost,masked_exact_max_tie=ties,score_change_raw=delta,score_change_sd=d).items():records[k][arm_i]=v
     arm_i+=1;arm_total+=1;focal_total+=len(b);tie_total+=int(ties.sum())
     for source in ['ALL']+c['manifest']['config']['sources']:
      subset=np.ones(len(b),bool) if source=='ALL' else fs==source;n=int(subset.sum());nsel=int((subset&selected0).sum())
      summary=dict(**meta,focal_source=source,fixed_focal_denominator=n,fixed_selected_denominator=nsel,exposed_fraction=float(exposed[subset].mean()),identity_switch_fraction=float(switched[subset].mean()),winner_title_fraction=float(uses_title[subset].mean()),score_increase_fraction=float((delta[subset]>0).mean()),score_decrease_fraction=float((delta[subset]<0).mean()),score_unchanged_fraction=float((delta[subset]==0).mean()),mean_shift_raw=float(delta[subset].mean()),mean_shift_sd=float(d[subset].mean()),baseline_selected_loss_fraction=int((lost&subset).sum())/nsel if nsel else None,baseline_selected_loss_count=int((lost&subset).sum()),baseline_tie_fraction=float(btie[subset].mean()),masked_tie_fraction=float(ties[subset].mean()),baseline_winner_same_source_fraction=float((hs[bwin[subset]]==fs[subset]).mean()),masked_winner_same_source_fraction=float((hs[win[subset]]==fs[subset]).mean()))
      maskrows.append(summary)
      transitions.extend(dict(**meta,focal_source=source,**cell) for cell in source_accounting(d,exposed,switched,uses_title,selected0,lost,subset))
    if pair_total%20==0:write_json(audit/'neighbor_checkpoint.json',dict(status='running',pairs=pair_total,arms=arm_total,last_assignment=row['assignment_id'],elapsed_seconds=time.monotonic()-start))
   assert arm_i==800
   write_csv(private/'assignment_index.csv',assign);np.savez_compressed(private/'neighbor_records.npz',**records)
   del records
  assert arm_total==1600 and pair_total==800 and focal_total==2422400
  common=['study','condition','rate','arm_role','focal_source']
  metrics=[k for k in maskrows[0] if k not in ['assignment_id','arm_id']+common]
  summary=aggregate(maskrows,common,metrics)
  tkeys=common+['baseline_winner_masked','identity_switched','winner_title_only']
  tmetrics=[k for k in transitions[0] if k not in ['assignment_id','arm_id']+tkeys]
  tsum=aggregate(transitions,tkeys,tmetrics)
  write_csv(out/'neighbor_per_assignment.csv',maskrows);write_csv(out/'neighbor_summary.csv',summary);write_csv(out/'neighbor_transition_per_assignment.csv',transitions);write_csv(out/'neighbor_transition_summary.csv',tsum)
  # Saved analysis-level R1 means provide an independent frozen-output comparison.
  comparisons=[]
  import csv
  for study,stage in [('ACL','confirmation_analysis'),('Biomedical','pubmed_replication_analysis')]:
   tab=pathlib.Path(root)/'results'/stage/'table2_aggregate.csv';inputs[str(tab.relative_to(pathlib.Path(root)))]=sha(tab)
   rr=list(csv.DictReader(tab.open()))
   # Exact scalar decomposition is checked above; aggregate R1 comparison is a separate reporting bridge.
   comparisons.append(dict(study=study,table=str(tab.relative_to(pathlib.Path(root))),rows=len(rr)))
  write_json(audit/'neighbor_inputs.json',inputs)
  lines=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{Post hoc exact-neighbor summaries for source-preferred substitution. Values are assignment means over 200 masks; exposure, identity switch and title-winner columns use the fixed focal-source denominator. Matched controls and transition contributions are in the numerical resource.}\label{tab:revision_neighbors}',r'\begin{tabular}{llrrrrr}',r'\toprule Collection/source & Loss & Exposed & Switch & Title & Shift & Lost \\ \midrule']
  for r in summary:
   if r['condition']!='source' or r['focal_source']=='ALL':continue
   lines.append(f"{r['study']}/{r['focal_source']} & {100*r['rate']:.0f} & {100*r['exposed_fraction_mean']:.1f} & {100*r['identity_switch_fraction_mean']:.1f} & {100*r['winner_title_fraction_mean']:.1f} & {r['mean_shift_sd_mean']:+.3f} & {100*r['baseline_selected_loss_fraction_mean']:.1f} "+r'\\')
  lines.extend([r'\bottomrule\end{tabular}',r'\par\smallskip\footnotesize Exposure, switching, title winner and selected loss are percentages; shift is in baseline population SD units. Selected loss uses the fixed baseline-selected count within each source.',r'\end{table}']);(out/'neighbor_table.tex').write_text('\n'.join(lines)+'\n')
  report=dict(status='VERIFIED',analysis_label='post hoc descriptive analysis',collections=list(COLLECTIONS),rates=[.25,.5],mechanism='source preferred and year-matched control',pairs=pair_total,arms=arm_total,focal_assignment_rows=focal_total,exact_saved_score_checks=focal_total+3028,score_mismatches=0,unmasked_baseline_winner_increase_violations=0,transition_shift_identity_checks=len(maskrows),baseline_tied_focals={s:int(sum(int(r['baseline_exact_max_tie']) for r in csv.DictReader((audit/'neighbor_records'/s/'focal_index.csv').open()))) for s in COLLECTIONS},masked_exact_max_ties=tie_total,identity_tie_rule='New diagnostic only: lexical historical ID; ties flagged; original maxima and focal selection rule unchanged',sd='float64 original focal baseline population SD (ddof=0)',quantiles="numpy.quantile(method='linear')",aggregation='mask-level; every source transition contribution divided by fixed source N; empty conditional cells undefined and defined-mask counts reported',elapsed_seconds=time.monotonic()-start,benchmark_seconds=bench['elapsed_seconds'],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,python=platform.python_version(),numpy=np.__version__,script_sha256=sha(__file__),input_manifest_sha256=sha(audit/'neighbor_inputs.json'),output_sha256={p.name:sha(p) for p in out.glob('neighbor*')},private_records_sha256={str(p.relative_to(audit)):sha(p) for p in (audit/'neighbor_records').rglob('*') if p.is_file()},limitations='Descriptive accounting, not mediation or source-label causation; all raw similarities already existed, no encoding/new masks.')
  report['R1_and_export_verification']=verify_r1_bridge(candidate)
  write_json(audit/'neighbor_verification.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ['output_sha256','private_records_sha256']},indent=2));return report

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--candidate',type=pathlib.Path,required=True);p.add_argument('--benchmark-only',action='store_true');a=p.parse_args();benchmark(a.root,a.candidate) if a.benchmark_only else run(a.root,a.candidate)
