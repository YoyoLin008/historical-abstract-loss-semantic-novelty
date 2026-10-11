"""One post hoc policy comparison, fixed in POLICY_SPEC.md before execution."""
import argparse,collections,csv,datetime,fcntl,hashlib,json,math,os,pathlib,sys,time
import numpy as np

def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def js(p,d):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)
def csvout(p,rr):
 with pathlib.Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def ranks(x):
 order=np.argsort(x,kind='stable');r=np.empty(len(x),float);i=0
 while i<len(x):
  j=i+1
  while j<len(x) and x[order[j]]==x[order[i]]:j+=1
  r[order[i:j]]=(i+j-1)/2+1;i=j
 return r
def selected(x,ids):return np.lexsort((np.array(ids),-x))[:math.ceil(.1*len(x))]
def endpoints(x,b,ids,a):
 sd=float(b.std(ddof=0));s0=selected(b,ids);sm=selected(x,ids);delta=x-b
 c=float(x[a].mean()-x[~a].mean());c0=float(b[a].mean()-b[~a].mean())
 R=float((b[s0].mean()-b[sm].mean())/sd);assert R>=-1e-14
 return dict(G=abs(c-c0)/sd,T=1-len(set(s0)&set(sm))/len(s0),Q=float(np.abs(ranks(x)-ranks(b)).mean()/(len(x)-1)),signed_contrast_sd=c/sd,drift_sd=float(delta.mean()/sd),source_A_shift_sd=float(delta[a].mean()/sd),source_B_shift_sd=float(delta[~a].mean()/sd),R=max(0.,R),increase_fraction=float((delta>0).mean()),decrease_fraction=float((delta<0).mean()))
def scores(full,title,mask,policy):
 assert (~mask).any();out=[]
 for i in range(0,len(full),128):
  f=full[i:i+128]
  matrix=np.where(mask[None,:],title[i:i+128],f) if policy=='title' else f[:,~mask]
  out.extend((np.float32(1)-matrix.max(axis=1)).astype(float))
 return np.array(out)
def tests():
 f=np.array([[.9,.5,.4],[.2,.8,.4],[.3,.4,.7]],np.float32);t=np.array([[.1,.5,.4],[.95,.2,.4],[.9,.4,.1]],np.float32);mask=np.array([1,0,0],bool);b=(np.float32(1)-f.max(axis=1)).astype(float)
 assert np.all(scores(f,t,mask,'removal')>=b)
 d=scores(f,t,mask,'title')-b;assert d[0]>0 and d[1]<0
 assert np.array_equal(ranks(np.array([1,1,3.])),[1.5,1.5,3])
 assert list(selected(np.ones(10),list('jihgfedcba')))==[9]
 b=np.arange(20,dtype=float);a=np.arange(20)%2==0;e=endpoints(b,b,list(map(str,range(20))),a);assert all(e[k]==0 for k in ['G','T','Q','R','drift_sd'])
 x=b[::-1];assert endpoints(x,b,list(map(str,range(20))),a)['R']>0
 return 6
def run(root,audit):
 start=time.monotonic();audit=pathlib.Path(audit);out=audit/'outputs/policy';out.mkdir(parents=True,exist_ok=True)
 assert (audit/'POLICY_SPEC.md').is_file();assert not (out/'verification.json').exists(),'completed result exists; do not duplicate'
 sys.path.insert(0,str(root/'scientometrics_submission_v1.1.0-rc.2/analysis'))
 from revision_mask_intensity import load_collection,saved_rows,mask_from_arm,validate_pair,ROLES
 frozen=json.loads((root/'scientometrics_submission_v1.1.0-rc.2/06_private_audits/components/mechanism/neighbor_inputs.json').read_text())
 inputs={};rows=[];checks=0;bench=[];bases=[]
 with (out/'run.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
  nt=tests()
  for study in ['ACL','Biomedical']:
   c=load_collection(root,study,inputs);full=np.load(c['d']/'similarity_cache/full_full.npy',mmap_mode='r');title=np.load(c['d']/'similarity_cache/full_title.npy',mmap_mode='r')
   for name in ['full_full','full_title']:
    p=c['d']/f'similarity_cache/{name}.npy';key=str(p.relative_to(root));inputs[key]=sha(p);assert inputs[key]==frozen[key]
   assert full.shape==title.shape==(len(c['focal']),len(c['history'])) and full.dtype==title.dtype==np.float32
   b=(np.float32(1)-full.max(axis=1)).astype(float);assert np.array_equal(b,c['baseline']['evaluations']['mixed']['novelty_scores'])
   ids=[r['id'] for r in c['focal']];a=np.array([r['source']==c['manifest']['config']['sources'][0] for r in c['focal']])
   bases.append(dict(study=study,N=len(b),H=full.shape[1],k=len(selected(b,ids)),s0=float(b.std()),baseline_contrast_sd=float((b[a].mean()-b[~a].mean())/b.std())))
   for row in saved_rows(c):
    if row['rate']!=.25 or row['mechanism'] not in ['random','source']:continue
    masks={role:mask_from_arm(row[role],c['history']) for role in ROLES if role in row};validate_pair(row,c['history'],masks)
    for role,mask in masks.items():
     assert mask.sum()==math.floor(.25*len(mask)+.5)
     init=time.monotonic()
     for policy in ['title','removal']:
      x=scores(full,title,mask,policy)
      if policy=='title':assert np.array_equal(x,np.array(row[role]['evaluations']['mixed']['novelty_scores'])),('title mismatch',row['assignment_id']);checks+=len(x)
      else:assert np.all(x>=b);checks+=len(x)
      e=endpoints(x,b,ids,a);assert abs(e['drift_sd']-(a.mean()*e['source_A_shift_sd']+(1-a.mean())*e['source_B_shift_sd']))<1e-13
      rows.append(dict(study=study,assignment_id=row['assignment_id'],arm_id=row[role]['assignment_id'],condition=row['mechanism'] if role==ROLES[0] else 'source_matched_control',policy=policy,**e))
     if not any(v['study']==study for v in bench):
      elapsed=time.monotonic()-init;bench.append(dict(study=study,first_arm_both_policies_seconds=elapsed,conservative_4x_600_arms_seconds=elapsed*2400));assert elapsed*2400<600
     if len(rows)%100==0:js(out/'checkpoint.json',dict(status='running',pid=os.getpid(),rows=len(rows),elapsed_seconds=time.monotonic()-start))
     assert time.monotonic()-start<600,'ten-minute hard cap'
  assert len(rows)==2400
  metrics=list(e);summary=[];groups=collections.defaultdict(list)
  for r in rows:groups[(r['study'],r['condition'],r['policy'])].append(r)
  def summarize(meta,rr):
   assert len(rr)==200;ans=[]
   for k in metrics:
    v=np.array([r[k] for r in rr]);ans.append(dict(**meta,endpoint=k,n=200,mean=float(v.mean()),assignment_sd=float(v.std(ddof=1)),mcse=float(v.std(ddof=1)/math.sqrt(200)),p05=float(np.quantile(v,.05)),p95=float(np.quantile(v,.95))))
   return ans
  for (s,c,p),rr in sorted(groups.items()):summary.extend(summarize(dict(study=s,condition=c,policy=p),rr))
  diffs=[]
  for s in ['ACL','Biomedical']:
   for c in ['random','source','source_matched_control']:
    aa={r['arm_id']:r for r in groups[(s,c,'title')]};bb={r['arm_id']:r for r in groups[(s,c,'removal')]};assert aa.keys()==bb.keys()
    diffs.extend(summarize(dict(study=s,condition=c,comparison='removal_minus_title'),[{k:bb[i][k]-aa[i][k] for k in metrics} for i in aa]))
   for p in ['title','removal']:
    aa={r['assignment_id']:r for r in groups[(s,'source',p)]};bb={r['assignment_id']:r for r in groups[(s,'source_matched_control',p)]};assert aa.keys()==bb.keys()
    diffs.extend(summarize(dict(study=s,condition='source_minus_matched_control',comparison=p),[{k:aa[i][k]-bb[i][k] for k in metrics} for i in aa]))
  csvout(out/'per_assignment.csv',rows);csvout(out/'summary.csv',summary);csvout(out/'paired_differences.csv',diffs);js(out/'input_hashes.json',inputs)
  js(out/'verification.json',dict(status='PASS',label='post hoc policy comparison specified after original results',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),arms=1200,policy_evaluations=2400,focal_score_checks=checks,exact_title_mismatches=0,removal_monotonicity_violations=0,synthetic_tests=nt,baselines=bases,benchmark=bench,elapsed_seconds=time.monotonic()-start,numpy=np.__version__,python=sys.version,script_sha256=sha(__file__),spec_sha256=sha(audit/'POLICY_SPEC.md'),inputs_manifest_sha256=sha(out/'input_hashes.json'),outputs_sha256={p.name:sha(p) for p in out.glob('*.csv')},uncertainty='Assignment SD, p05/p95 variability and MCSE only; no confidence intervals, testing, tolerance classes, population generalization or extension of frozen confirmatory families.'))
  js(out/'checkpoint.json',dict(status='complete',pid=os.getpid(),rows=len(rows),elapsed_seconds=time.monotonic()-start));print((out/'verification.json').read_text())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--audit',type=pathlib.Path,required=True);args=p.parse_args();run(args.root,args.audit)
