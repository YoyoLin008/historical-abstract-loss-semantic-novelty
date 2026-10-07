"""Separately frozen, post hoc studentized whole-mask sensitivity; read originals only."""
from pathlib import Path
import argparse,collections,csv,fcntl,hashlib,json,math,os,platform,resource,socket,time
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
B=200000; BATCH=2000; SEED=20261006; MAX_SECONDS=570
PLAN=OUT/'bootstrap_plan.md'; FREEZE=OUT/'pre_execution_manifest.json'
STUDIES=[('ACL','confirmation_analysis',45),('Biomedical','pubmed_replication_analysis',18)]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def atomic_json(p,v):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(p)
def csv_rows(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def load():
 inputs={}; blocks=[]; originals={}
 for study,folder,J in STUDIES:
  directory=ROOT/'results'/folder;manifest=json.loads((directory/'analysis_manifest.json').read_text())
  inputs[str((directory/'analysis_manifest.json').relative_to(ROOT))]=sha(directory/'analysis_manifest.json')
  for name in ['mask_primary_and_movement.csv','table2_aggregate.csv','table3_rank_selection.csv','table4_paired_primary.csv','joint_classification.csv']:
   p=directory/name;assert sha(p)==manifest['output_sha256'][name],name
   inputs[str(p.relative_to(ROOT))]=sha(p)
  for name in ['table2_aggregate.csv','table3_rank_selection.csv']:
   for row in csv_rows(directory/name):
    if row['policy']=='mixed' and row['condition'] in ['random','source','topic'] and row['metric'] in ['G','T','Q']:
     originals[(study,float(row['rate']),row['condition'],'base',row['metric'])]=row
  for row in csv_rows(directory/'table4_paired_primary.csv'):
   originals[(study,float(row['rate']),row['mechanism'],'paired',row['metric'])]=row
  groups=collections.defaultdict(list)
  for row in csv_rows(directory/'mask_primary_and_movement.csv'):groups[(float(row['rate']),row['condition'])].append(row)
  for (rate,condition),rows in sorted(groups.items()):
   if condition not in ['random','source','topic']:continue
   rows=sorted(rows,key=lambda r:r['assignment_id']);assert len(rows)==200
   values=np.array([[float(r[m]) for m in ['G','T','Q']] for r in rows]); labels=[('base',m) for m in ['G','T','Q']]
   if condition!='random':
    controls={r['assignment_id']:r for r in groups[(rate,condition+'_matched_control')]};assert len(controls)==200 and set(controls)=={r['assignment_id'] for r in rows}
    control=np.array([[float(controls[r['assignment_id']][m]) for m in ['G','T','Q']] for r in rows]);values=np.column_stack([values,values-control]);labels += [('paired',m) for m in ['G','T','Q']]
   assert np.isfinite(values).all()
   mu=values.mean(axis=0); se=values.std(axis=0,ddof=1)/math.sqrt(200);assert (se>0).all()
   for col,(role,metric) in enumerate(labels):
    original=originals[(study,rate,condition,role,metric)]
    assert abs(mu[col]-float(original['mean']))<5e-12
    assert abs(se[col]-float(original['mcse']))<5e-12
   tag=f'{study.lower()}_{condition}_{int(rate*100):02d}'
   payload=json.dumps(['major-revision-studentized-bootstrap-v1',SEED,study,rate,condition],separators=(',',':'))
   seed=int.from_bytes(hashlib.sha256(payload.encode()).digest()[:16],'big')
   blocks.append(dict(study=study,rate=rate,condition=condition,J=J,tag=tag,seed=seed,values=values,labels=labels,mean=mu,se=se))
 assert len(originals)==63 and sum(len(b['labels']) for b in blocks)==63
 return inputs,blocks,originals
def classify(metric,lo,hi):
 threshold={'G':.10,'T':.05,'Q':.01}[metric]
 return 'below_tolerance' if hi<threshold else ('above_tolerance' if lo>threshold else 'unresolved')
def direction(lo,hi):return 'positive' if lo>0 else ('negative' if hi<0 else 'unresolved')
def mc_quantile_band(t,p):
 # Approximate binomial-order-statistic uncertainty in a resampling quantile.
 ordered=np.sort(t); center=B*p; radius=1.959963984540054*math.sqrt(B*p*(1-p))
 low=max(0,int(math.floor(center-radius))-1);high=min(B-1,int(math.ceil(center+radius))-1)
 return float(ordered[low]),float(ordered[high]),low+1,high+1
def write_csv(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
 inputs,blocks,originals=load()
 fingerprint={'code_sha256':sha(Path(__file__)),'plan_sha256':sha(PLAN),'inputs_sha256':inputs,'python':platform.python_version(),'numpy':np.__version__,'B':B,'batch':BATCH,'resampling_root_seed':SEED,'max_seconds':MAX_SECONDS,'block_seeds':{b['tag']:str(b['seed']) for b in blocks}}
 if args.freeze:
  assert not FREEZE.exists(),'Existing freeze must not be replaced'
  atomic_json(FREEZE,dict(status='posthoc_plan_frozen_before_resampling',created_utc=now(),scope='after primary outcomes; not preregistration or original confirmatory inference',**fingerprint));print(json.dumps({'status':'frozen','manifest_sha256':sha(FREEZE)}));return
 frozen=json.loads(FREEZE.read_text());assert all(frozen[k]==v for k,v in fingerprint.items()),'Pre-execution fingerprint changed'
 lock=(OUT/'bootstrap.lock').open('a+')
 try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise RuntimeError('Bootstrap lock is held; no duplicate job permitted')
 job=dict(status='running',pid=os.getpid(),host=socket.gethostname(),started_utc=now(),freeze_sha256=sha(FREEZE),completed_blocks=[],checkpoints={})
 oldpath=OUT/'job_manifest.json'
 if oldpath.exists():
  old=json.loads(oldpath.read_text());assert old['freeze_sha256']==job['freeze_sha256']
  if old['status']=='complete':print('Already complete; no rerun');return
  job['checkpoints']=old.get('checkpoints',{});job['completed_blocks']=old.get('completed_blocks',[])
 atomic_json(oldpath,job);start=time.monotonic();results=[]
 try:
  for block in blocks:
   tag=block['tag'];path=OUT/(tag+'_studentized.npy');progress=job['checkpoints'].get(tag,{})
   done=progress.get('completed',0);rng=np.random.Generator(np.random.PCG64(block['seed']))
   if progress:rng.bit_generator.state=progress['rng_state']
   if path.exists():t=np.lib.format.open_memmap(path,mode='r+');assert t.shape==(B,len(block['labels']))
   else:assert done==0;t=np.lib.format.open_memmap(path,mode='w+',dtype='float64',shape=(B,len(block['labels'])))
   while done<B:
    if time.monotonic()-start>MAX_SECONDS:
     job.update(status='partial',finished_utc=now(),elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);atomic_json(oldpath,job);print(json.dumps({'status':'partial','block':tag,'completed':done}));return
    stop=min(done+BATCH,B);idx=rng.integers(0,200,size=(stop-done,200));sample=block['values'][idx]
    mu=sample.mean(axis=1);se=sample.std(axis=1,ddof=1)/math.sqrt(200)
    if not np.isfinite(se).all() or (se==0).any():raise ValueError('Undefined or zero bootstrap SE; fixed plan stops without substitution')
    t[done:stop]=(mu-block['mean'])/se;t.flush();done=stop
    job['checkpoints'][tag]={'completed':done,'rng_state':rng.bit_generator.state};job['elapsed_seconds']=time.monotonic()-start;atomic_json(oldpath,job)
   if tag not in job['completed_blocks']:job['completed_blocks'].append(tag)
   a=.05/(2*block['J'])
   for col,(role,metric) in enumerate(block['labels']):
    x=np.asarray(t[:,col]);ql,qh=np.quantile(x,[a,1-a],method='linear');lo=float(block['mean'][col]-qh*block['se'][col]);hi=float(block['mean'][col]-ql*block['se'][col]);original=originals[(block['study'],block['rate'],block['condition'],role,metric)]
    ol=float(original['ci_low_raw']);oh=float(original['ci_high_raw']);orig_label=classify(metric,ol,oh) if role=='base' else direction(ol,oh);new_label=classify(metric,lo,hi) if role=='base' else direction(lo,hi)
    ql_lo,ql_hi,ql_lrank,ql_hrank=mc_quantile_band(x,a);qh_lo,qh_hi,qh_lrank,qh_hrank=mc_quantile_band(x,1-a)
    low_mc_min=float(block['mean'][col]-qh_hi*block['se'][col]);low_mc_max=float(block['mean'][col]-qh_lo*block['se'][col]);high_mc_min=float(block['mean'][col]-ql_hi*block['se'][col]);high_mc_max=float(block['mean'][col]-ql_lo*block['se'][col]);half=np.quantile(x[:B//2],[a,1-a],method='linear');half2=np.quantile(x[B//2:],[a,1-a],method='linear')
    label_wide=classify(metric,low_mc_min,high_mc_max) if role=='base' else direction(low_mc_min,high_mc_max)
    label_narrow=classify(metric,low_mc_max,high_mc_min) if role=='base' else direction(low_mc_max,high_mc_min)
    results.append(dict(study=block['study'],rate=block['rate'],mechanism=block['condition'],role=role,metric=metric,family_size=block['J'],n_masks=200,B=B,mean=float(block['mean'][col]),mcse=float(block['se'][col]),student_low=ol,student_high=oh,bootstrap_low=lo,bootstrap_high=hi,student_classification=orig_label,bootstrap_classification=new_label,classification_concordant=orig_label==new_label,tail_probability=a,expected_resamples_per_tail=B*a,t_quantile_low=float(ql),t_quantile_high=float(qh),t_quantile_low_first_half=float(half[0]),t_quantile_low_second_half=float(half2[0]),t_quantile_high_first_half=float(half[1]),t_quantile_high_second_half=float(half2[1]),lower_endpoint_mc_min=low_mc_min,lower_endpoint_mc_max=low_mc_max,upper_endpoint_mc_min=high_mc_min,upper_endpoint_mc_max=high_mc_max,classification_stable_in_approx_mc_band=label_wide==label_narrow,lower_quantile_mc_rank_low=ql_lrank,lower_quantile_mc_rank_high=ql_hrank,upper_quantile_mc_rank_low=qh_lrank,upper_quantile_mc_rank_high=qh_hrank))
   print(json.dumps({'block':tag,'status':'complete','elapsed_seconds':time.monotonic()-start}),flush=True);atomic_json(oldpath,job)
  assert len(results)==63;write_csv(OUT/'interval_sensitivity.csv',results)
  joints=[]
  for block in blocks:
   rr={r['metric']:r for r in results if r['study']==block['study'] and r['rate']==block['rate'] and r['mechanism']==block['condition'] and r['role']=='base'}
   original=rr['G']['student_high']<.1 and rr['T']['student_low']>.05;boot=rr['G']['bootstrap_high']<.1 and rr['T']['bootstrap_low']>.05
   joints.append(dict(study=block['study'],rate=block['rate'],mechanism=block['condition'],student_joint=original,bootstrap_joint=boot,concordant=original==boot,mc_classification_stable=rr['G']['classification_stable_in_approx_mc_band'] and rr['T']['classification_stable_in_approx_mc_band']))
  write_csv(OUT/'joint_sensitivity.csv',joints)
  report=dict(status='complete',scope='post hoc studentized whole-mask bootstrap; original frozen inference unchanged',n_means=63,n_joint_conditions=13,endpoint_discordances=[r for r in results if not r['classification_concordant']],joint_discordances=[r for r in joints if not r['concordant']],mc_sensitive_endpoints=[{k:r[k] for k in ['study','rate','mechanism','role','metric','bootstrap_classification','lower_endpoint_mc_min','lower_endpoint_mc_max','upper_endpoint_mc_min','upper_endpoint_mc_max']} for r in results if not r['classification_stable_in_approx_mc_band']],B=B,expected_tail_counts={'ACL':B*.05/90,'Biomedical':B*.05/36},monte_carlo_scope='Finite resampling uncertainty; approximate95%binomial order-statistic bands, not scientific intervals, not familywise MC guarantees',inference_scope='approximate bootstrap-Bonferroni sensitivity with separate45/18families; no exact finite-sample coverage or cross-study95%guarantee',elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  atomic_json(OUT/'sensitivity_summary.json',report)
  job.update(status='complete',finished_utc=now(),elapsed_seconds=report['elapsed_seconds'],peak_rss_bytes=report['peak_rss_bytes'],output_sha256={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and (p.suffix in ['.csv','.npy'] or p.name=='sensitivity_summary.json')});atomic_json(oldpath,job)
  print(json.dumps({k:v for k,v in report.items() if k not in ['endpoint_discordances']},indent=2),flush=True)
 except BaseException as exc:
  job.update(status='failed',finished_utc=now(),error=repr(exc),elapsed_seconds=time.monotonic()-start);atomic_json(oldpath,job);raise
 finally:fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()
if __name__=='__main__':main()
