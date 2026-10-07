"""Readout-only Student solver repair; never re-encodes or recomputes novelty vectors."""
from pathlib import Path
import csv,fcntl,hashlib,importlib.util,json,math,os,resource,socket,time
from datetime import datetime,timezone
import mpmath as mp
import numpy as np
BASE=Path(__file__).resolve().parents[3]/'external_work';OUT=BASE/'encoder_sensitivity';FREEZE=OUT/'readout_solver_amendment.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def atomic(p,v):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(p)
def critical_value():
 with mp.workdps(50):
  df=mp.mpf(199);target=mp.mpf('.05')/24;lo=mp.mpf(2);hi=mp.mpf(4)
  def tail(t):return mp.betainc(df/2,mp.mpf('.5'),0,df/(df+t*t),regularized=True)/2
  assert tail(lo)>target>tail(hi)
  for _ in range(80):
   mid=(lo+hi)/2
   if tail(mid)>target:lo=mid
   else:hi=mid
  critical=float((lo+hi)/2)
  assert abs(critical-2.8987694542968003)<1e-12
  return critical
def main():
 freeze=json.loads(FREEZE.read_text());assert sha(Path(__file__))==freeze['code_sha256']
 for rel,digest in freeze['bound_files_sha256'].items():assert sha(BASE/rel)==digest,rel
 spec=importlib.util.spec_from_file_location('frozen_vector_code',BASE/'methods/encoder_check_v1_1.py');original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original);binding=original.verify()
 lock=(OUT/'scoring.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 prior=json.loads((OUT/'scoring_manifest.json').read_text());assert prior['status']=='failed' and sha(OUT/'scoring_manifest.json')==freeze['failed_scoring_manifest_sha256']
 start=time.monotonic();job={'status':'running','stage':'readout_only_solver_repair','pid':os.getpid(),'host':socket.gethostname(),'started_utc':now(),'freeze_sha256':binding,'readout_solver_amendment_sha256':sha(FREEZE),'code_sha256':sha(Path(__file__)),'vector_code_sha256':sha(BASE/'methods/encoder_check_v1_1.py'),'preserved_failure_manifest_sha256':freeze['failed_scoring_manifest_sha256']};atomic(OUT/'scoring_manifest.json',job)
 try:
  with (OUT/'mask_results.jsonl').open() as f:results=[json.loads(line) for line in f]
  assert len(results)==1000
  critical=critical_value();rows=[]
  for study,mechanism,role in [('acl','random','base'),('biomedical','random','base'),('biomedical','source','base'),('biomedical','source','paired')]:
   subset=[r for r in results if (r['study'],r['mechanism'],r['role'])==(study,mechanism,role)];assert len(subset)==200 and len({r['assignment_id'] for r in subset})==200
   for metric in ['G','T','Q']:
    x=np.array([r[metric] for r in subset]);assert np.isfinite(x).all();mean=float(x.mean());sdx=float(x.std(ddof=1));se=sdx/math.sqrt(200);lo=mean-critical*se;hi=mean+critical*se;threshold={'G':.1,'T':.05,'Q':.01}[metric];label=('below_tolerance' if hi<threshold else 'above_tolerance' if lo>threshold else 'unresolved') if role=='base' else ('positive' if lo>0 else 'negative' if hi<0 else 'unresolved')
    rows.append(dict(study=study,mechanism=mechanism,role=role,rate=.25,metric=metric,n=200,mean=mean,sd=sdx,mcse=se,ci_low=lo,ci_high=hi,p05=float(np.quantile(x,.05)),median=float(np.median(x)),p95=float(np.quantile(x,.95)),classification=label,expected_lost_papers=mean*(156 if study=='acl' else 148) if metric=='T' and role=='base' else None))
  with (OUT/'primary_and_paired.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  atomic(OUT/'readout.json',dict(status='complete',family_size=12,critical=critical,df=199,scope='post hoc single related encoder sensitivity; separate approximate family; original inference unchanged',means=rows))
  # Verify the point vectors were not changed by this interval-only repair.
  for name,digest in freeze['saved_vectors_sha256'].items():assert sha(OUT/name)==digest
  job.update(status='complete',finished_utc=now(),elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Novelty vectors produced once by v1.1; only the incomplete Student-quantile/readout stage completed by v1.2; original primary study unchanged',output_sha256={n:sha(OUT/n) for n in ['baselines.json','numerical_verification.json','mask_results.jsonl','primary_and_paired.csv','readout.json']});atomic(OUT/'scoring_manifest.json',job);print(json.dumps({'status':'complete','critical':critical,'elapsed_seconds':job['elapsed_seconds'],'rows':len(rows)}))
 except BaseException as exc:job.update(status='failed',finished_utc=now(),error=repr(exc));atomic(OUT/'scoring_manifest.json',job);raise
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':main()
