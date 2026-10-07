"""Limited post hoc SPECTER2 robustness: immutable inputs, CPU checkpoints, separate outputs."""
from pathlib import Path
import argparse,collections,csv,fcntl,hashlib,json,math,os,platform,resource,socket,sys,time
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
BASE=Path(__file__).resolve().parents[3]/'external_work'
OUT=BASE/'encoder_sensitivity';MODELS=BASE/'encoder_models';OUT.mkdir(exist_ok=True)
PLAN=Path(__file__).with_name('encoder_protocol.md');FREEZE=OUT/'pre_execution_serialization_amendment.json'
STUDIES={'acl':('ACL','data/confirmation_v1/records.jsonl','results/confirmation_v1',('ACL','EMNLP'),5356,1557,3799,156),'biomedical':('Biomedical','replication/data/records.jsonl','results/pubmed_replication_v1',('PLOSCB','PLOSGEN'),4862,1471,3391,148)}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def atomic(p,v):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(p)
def records(study):
 with (ROOT/STUDIES[study][1]).open() as f:return [json.loads(x) for x in f]
def dependencies():
 import torch,transformers,adapters
 return dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,transformers=transformers.__version__,adapters=adapters.__version__)
def fingerprints():
 deps=dependencies();assert deps==dict(python='3.12.14',numpy='2.3.5',torch='2.8.0',transformers='4.57.6',adapters='1.3.0'),deps
 inventory=json.loads((MODELS/'download_manifest.json').read_text());assert inventory['status']=='complete' and inventory['actual_total_bytes']<=1000000000
 for r in inventory['files']:
  sub='base' if r['model']=='allenai/specter2_base' else 'proximity';p=MODELS/sub/r['filename'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
 inputs={}
 for _,spec in STUDIES.items():
  manifest=json.loads((ROOT/spec[2]/'run_manifest.json').read_text());assert manifest['status']=='complete'
  for p in [ROOT/spec[1],ROOT/spec[2]/'baseline.json',ROOT/spec[2]/'mask_results.jsonl']:
   inputs[str(p.relative_to(ROOT))]=sha(p)
  assert inputs[spec[2]+'/baseline.json']==manifest['output_sha256']['baseline.json']
  assert inputs[spec[2]+'/mask_results.jsonl']==manifest['output_sha256']['mask_results.jsonl']
 for rel in ['src/measure.py','src/masks.py','requirements.lock']:
  inputs[rel]=sha(ROOT/rel)
 return dict(code_sha256=sha(Path(__file__)),protocol_sha256=sha(PLAN),dependencies=deps,model_manifest_sha256=sha(MODELS/'download_manifest.json'),inputs_sha256=inputs,conditions=[['acl','random',.25],['biomedical','random',.25],['biomedical','source',.25]],batch=8,chunk=64,threads=4,max_invocation_seconds=1650,representation='authentic title+separator+abstract/title+separator;CLS;L2;float32;512tokens;eager',family_size=12)
def verify():
 frozen=json.loads(FREEZE.read_text());fp=fingerprints();assert all(frozen[k]==v for k,v in fp.items()),'Post hoc fingerprint changed'
 original=OUT/'pre_execution_manifest.json';assert sha(original)==frozen['supersedes_freeze_sha256']
 return frozen['supersedes_freeze_sha256']
def load_model():
 import torch
 from adapters import AutoAdapterModel
 from transformers import AutoTokenizer
 torch.set_num_threads(4);torch.manual_seed(20261006)
 tok=AutoTokenizer.from_pretrained(MODELS/'base',local_files_only=True,trust_remote_code=False)
 model=AutoAdapterModel.from_pretrained(MODELS/'base',local_files_only=True,attn_implementation='eager').eval()
 model.load_adapter(str(MODELS/'proximity'),load_as='proximity',set_active=True)
 assert str(model.active_adapters)=='Stack[proximity]',str(model.active_adapters)
 def embed(texts):
  tokens=tok(texts,padding=True,truncation=True,max_length=512,return_tensors='pt',return_token_type_ids=False)
  with torch.inference_mode():v=torch.nn.functional.normalize(model(**tokens).last_hidden_state[:,0,:],p=2,dim=1)
  arr=v.cpu().numpy().astype('float32');assert arr.shape==(len(texts),768) and np.isfinite(arr).all() and np.allclose(np.linalg.norm(arr,axis=1),1,atol=1e-5)
  return arr
 return tok,embed
def input_text(row,mode,tok):return row['title']+tok.sep_token+(row['abstract'] if mode=='full' else '')
def probe():
 freeze=verify();tok,embed=load_model();texts=[];labels=[]
 for study in STUDIES:
  rr=records(study);lengths=[len(tok(input_text(r,'full',tok),truncation=False)['input_ids']) for r in rr];order=np.argsort(lengths,kind='stable');ix=order[np.linspace(0,len(rr)-1,16,dtype=int)]
  for mode in ['full','title']:
   texts.extend(input_text(rr[i],mode,tok) for i in ix);labels.extend({'study':study,'id':rr[i]['id'],'mode':mode,'raw_tokens':len(tok(input_text(rr[i],mode,tok),truncation=False)['input_ids'])} for i in ix)
 embed(texts[:8]);start=time.monotonic();allv=np.concatenate([embed(texts[i:i+8]) for i in range(0,64,8)]);elapsed=time.monotonic()-start;repeat=embed(texts[:8]);difference=float(np.max(abs(repeat-allv[:8])));assert difference<2e-6
 peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;projected=elapsed/64*20436;passed=projected<=5400 and peak<=4000000000
 atomic(OUT/'probe.json',dict(status='pass' if passed else 'gate_failed',freeze_sha256=freeze,n_inputs=64,seconds=elapsed,inputs_per_second=64/elapsed,projected_full_encoding_seconds=projected,projection_scope='linear extrapolation from length-stratified64input probe; actual full execution may differ',peak_rss_bytes=peak,repeated_forward_max_abs_difference=difference,records=labels))
 print(json.dumps({'status':'pass' if passed else 'gate_failed','seconds':elapsed,'projected_seconds':projected,'peak_rss_bytes':peak}));assert passed,'Measured probe gate failed'
def encode(study,max_seconds):
 freeze=verify();probe_info=json.loads((OUT/'probe.json').read_text());assert probe_info['status']=='pass' and probe_info['freeze_sha256']==freeze
 directory=OUT/study;directory.mkdir(exist_ok=True);chunks=directory/'chunks';chunks.mkdir(exist_ok=True);lock=(directory/'encoding.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 rr=records(study);assert len(rr)==STUDIES[study][4];start=time.monotonic();job=dict(status='running',pid=os.getpid(),host=socket.gethostname(),started_utc=now(),freeze_sha256=freeze,completed_chunks={})
 jp=directory/'encoding_manifest.json'
 if jp.exists():
  old=json.loads(jp.read_text());assert old['freeze_sha256']==freeze
  if old['status']=='complete':
   for name,digest in old['output_sha256'].items():assert sha(directory/name)==digest
   print('Already encoded');fcntl.flock(lock.fileno(),fcntl.LOCK_UN);return
  job['completed_chunks']=old['completed_chunks']
 atomic(jp,job);tok,embed=load_model();truncation=collections.Counter()
 try:
  for begin in range(0,len(rr),64):
   end=min(begin+64,len(rr));name=f'chunk_{begin:05d}_{end:05d}.npz';p=chunks/name
   if name in job['completed_chunks']:
    assert sha(p)==job['completed_chunks'][name]
    continue
   if time.monotonic()-start>=max_seconds:
    job.update(status='partial',finished_utc=now(),elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);atomic(jp,job);print(json.dumps({'status':'partial','study':study,'completed_chunks':len(job['completed_chunks'])}));return
   vv={};lens={}
   for mode in ['full','title']:
    texts=[input_text(row,mode,tok) for row in rr[begin:end]];vv[mode]=np.concatenate([embed(texts[j:j+8]) for j in range(0,len(texts),8)]);lens[mode]=np.array([len(tok(t,truncation=False)['input_ids']) for t in texts],dtype='int32')
   tmp=p.with_suffix('.partial.npz');np.savez(tmp,**vv,full_lengths=lens['full'],title_lengths=lens['title']);tmp.replace(p);job['completed_chunks'][name]=sha(p);job['elapsed_seconds']=time.monotonic()-start;atomic(jp,job)
   if len(job['completed_chunks'])%10==0:print(study,len(job['completed_chunks']),'chunks',job['elapsed_seconds'],'seconds',flush=True)
  arrays={mode:[] for mode in ['full','title']};lengths={mode:[] for mode in ['full','title']}
  for begin in range(0,len(rr),64):
   end=min(begin+64,len(rr));name=f'chunk_{begin:05d}_{end:05d}.npz';p=chunks/name;assert sha(p)==job['completed_chunks'][name]
   with np.load(p,allow_pickle=False) as data:
    for mode in arrays:arrays[mode].append(data[mode]);lengths[mode].append(data[mode+'_lengths'])
  audit=[]
  for mode in arrays:
   values=np.concatenate(arrays[mode]);ll=np.concatenate(lengths[mode]);assert values.shape==(len(rr),768) and np.isfinite(values).all() and np.allclose(np.linalg.norm(values,axis=1),1,atol=1e-5);np.save(directory/(mode+'.npy'),values,allow_pickle=False)
   for i,row in enumerate(rr):
    if ll[i]>512:truncation[(mode,row['year'],row['source'])]+=1
   audit.append(dict(mode=mode,n_records=len(rr),truncated=int(np.sum(ll>512)),max_raw_tokens=int(ll.max())))
  atomic(directory/'token_audit.json',dict(totals=audit,by_source_year=[dict(mode=m,year=y,source=s,n=v) for (m,y,s),v in sorted(truncation.items())]))
  job.update(status='complete',finished_utc=now(),elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,output_sha256={n:sha(directory/n) for n in ['full.npy','title.npy','token_audit.json']});atomic(jp,job);print(json.dumps({'status':'complete','study':study,'elapsed_seconds':job['elapsed_seconds'],'peak_rss_bytes':job['peak_rss_bytes']}))
 except BaseException as exc:job.update(status='failed',error=repr(exc),finished_utc=now());atomic(jp,job);raise
 finally:fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()
def ranks(x):
 order=np.argsort(x,kind='stable');r=np.empty(len(x));start=0
 while start<len(x):
  stop=start+1
  while stop<len(x) and x[order[stop]]==x[order[start]]:stop+=1
  r[order[start:stop]]=(start+stop-1)/2;start=stop
 return r
def selected(x,ids,k):return set(np.lexsort((np.array(ids),-np.array(x)))[:k].tolist())
def score_all():
 freeze=verify();lock=(OUT/'scoring.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.monotonic();job=dict(status='running',pid=os.getpid(),host=socket.gethostname(),freeze_sha256=freeze,started_utc=now(),scoring_amendment_sha256=sha(FREEZE),code_sha256=sha(Path(__file__)));atomic(OUT/'scoring_manifest.json',job)
 try:
  results=[];baselines={};verifications={}
  for study,spec in STUDIES.items():
   em=json.loads((OUT/study/'encoding_manifest.json').read_text());assert em['status']=='complete' and em['freeze_sha256']==freeze
   for n,d in em['output_sha256'].items():assert sha(OUT/study/n)==d
   rr=records(study);lookup={r['id']:i for i,r in enumerate(rr)};scope=json.loads((ROOT/spec[2]/'baseline.json').read_text())['scope'];fi=np.array([lookup[i] for i in scope['focal_ids']]);hi=np.array([lookup[i] for i in scope['history_ids']]);assert(len(fi),len(hi))==(spec[5],spec[6]);assert all(rr[i]['year']==2021 for i in fi) and all(rr[i]['year']<2021 for i in hi)
   full=np.load(OUT/study/'full.npy',mmap_mode='r');title=np.load(OUT/study/'title.npy',mmap_mode='r');sims=[]
   for right in [full,title]:
    matrix=np.empty((len(fi),len(hi)),dtype='float32')
    for st in range(0,len(fi),128):matrix[st:st+128]=np.clip(full[fi[st:st+128]]@right[hi].T,-1,1)
    sims.append(matrix)
   baseline=(1-sims[0].max(axis=1)).astype('float64');sd=float(baseline.std(ddof=0));assert sd>0;ids=scope['focal_ids'];src=np.array([rr[i]['source'] for i in fi]);A=src==spec[3][0];Bmask=src==spec[3][1];contrast=float(baseline[A].mean()-baseline[Bmask].mean());k=spec[7];old=selected(baseline,ids,k);br=ranks(baseline);order=np.lexsort((np.array(ids),-baseline));cut=float((baseline[order[k-1]]+baseline[order[k]])/2);margin=(baseline-cut)/sd;percentile=(np.arange(1,len(fi)+1)-.5)/len(fi);pos=np.flatnonzero((percentile>=.09)&(percentile<=.11));band=(baseline<=baseline[order[pos[0]]])&(baseline>=baseline[order[pos[-1]]]);hlookup={ident:i for i,ident in enumerate(scope['history_ids'])}
   def outcomes(z):
    sm=selected(z,ids,k);lost=old-sm
    return dict(G=abs(float(z[A].mean()-z[Bmask].mean())-contrast)/sd,T=len(lost)/k,Q=float(np.mean(abs(ranks(z)-br))/(len(fi)-1)),selected_lost_count=len(lost),outside_band_lost_count=sum(not band[i] for i in lost),far_0_1_sd_lost_count=int(sum(margin[i]>.1 for i in lost)))
   base=dict(study=study,n_focal=len(fi),n_history=len(hi),k=k,sd=sd,source_contrast_raw=contrast,source_contrast_sd=contrast/sd,focal_ids=ids,baseline_scores=baseline.tolist(),selected_ids=[ids[i] for i in sorted(old)],cutoff_gap_sd=float((baseline[order[k-1]]-baseline[order[k]])/sd),band_n=int(band.sum()),baseline_selected_in_band=int(sum(band[i] for i in old)));baselines[study]=base
   direct_errors=[];count=collections.Counter();first_mask=None
   with (ROOT/spec[2]/'mask_results.jsonl').open() as f:
    for line in f:
     record=json.loads(line)
     if record['rate']!=.25 or record['mechanism'] not in (['random'] if study=='acl' else ['random','source']):continue
     evaluated={}
     for role in ['structured_or_random','matched_random_control']:
      if role not in record:continue
      mids=record[role]['missing_history_ids'];mask=np.zeros(len(hi),dtype=bool);mask[[hlookup[x] for x in mids]]=True;assert mask.sum()==math.floor(len(hi)*.25+.5)
      z=(1-np.where(mask[None,:],sims[1],sims[0]).max(axis=1)).astype('float64');o=outcomes(z);evaluated[role]=o
      results.append(dict(study=study,mechanism=record['mechanism'],role='base' if role=='structured_or_random' else 'control',assignment_id=record['assignment_id'],missing_history_ids=mids,novelty_scores=z.tolist(),**o));count[(record['mechanism'],role)]+=1
      if first_mask is None:first_mask=mask
     if record['mechanism']=='source':results.append(dict(study=study,mechanism='source',role='paired',assignment_id=record['assignment_id'],**{m:evaluated['structured_or_random'][m]-evaluated['matched_random_control'][m] for m in ['G','T','Q']}))
   assert count[('random','structured_or_random')]==200
   if study=='biomedical':assert count[('source','structured_or_random')]==count[('source','matched_random_control')]==200
   for ii in np.linspace(0,len(fi)-1,16,dtype=int):
    fv=np.array(full[fi[ii]],dtype='float64');historical=np.array(full[hi],dtype='float64');direct=1-np.max(np.clip(historical@fv,-1,1));direct_errors.append(abs(direct-baseline[ii]))
    historical[first_mask]=np.array(title[hi[first_mask]],dtype='float64');z=(1-np.where(first_mask,sims[1][ii],sims[0][ii]).max()).item();direct_errors.append(abs(1-np.max(np.clip(historical@fv,-1,1))-z))
   assert max(direct_errors)<2e-6;verifications[study]=dict(n_scalar_checks=32,max_scalar_error=float(max(direct_errors)),condition_role_counts={str(key):v for key,v in count.items()})
  atomic(OUT/'baselines.json',baselines);atomic(OUT/'numerical_verification.json',verifications)
  p=OUT/'mask_results.jsonl';tmp=p.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in results));tmp.replace(p)
  import mpmath as mp
  with mp.workdps(50):
   df=mp.mpf(199);tail=mp.mpf('.05')/24;critical=float(mp.findroot(lambda t:mp.betainc(df/2,mp.mpf('.5'),0,df/(df+t*t),regularized=True)/2-tail,(2,4)))
  rows=[]
  for study,mechanism,role in [('acl','random','base'),('biomedical','random','base'),('biomedical','source','base'),('biomedical','source','paired')]:
   subset=[r for r in results if (r['study'],r['mechanism'],r['role'])==(study,mechanism,role)];assert len(subset)==200
   for metric in ['G','T','Q']:
    x=np.array([r[metric] for r in subset]);mean=float(x.mean());sdx=float(x.std(ddof=1));se=sdx/math.sqrt(200);lo=mean-critical*se;hi=mean+critical*se;threshold={'G':.1,'T':.05,'Q':.01}[metric];label=('below_tolerance' if hi<threshold else 'above_tolerance' if lo>threshold else 'unresolved') if role=='base' else ('positive' if lo>0 else 'negative' if hi<0 else 'unresolved')
    rows.append(dict(study=study,mechanism=mechanism,role=role,rate=.25,metric=metric,n=200,mean=mean,sd=sdx,mcse=se,ci_low=lo,ci_high=hi,p05=float(np.quantile(x,.05)),median=float(np.median(x)),p95=float(np.quantile(x,.95)),classification=label,expected_lost_papers=mean*STUDIES[study][7] if metric=='T' and role=='base' else None))
  with (OUT/'primary_and_paired.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  atomic(OUT/'readout.json',dict(status='complete',family_size=12,critical=critical,df=199,scope='post hoc single related encoder sensitivity; separate approximate family; original inference unchanged',means=rows))
  job.update(status='complete',finished_utc=now(),elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,output_sha256={n:sha(OUT/n) for n in ['baselines.json','numerical_verification.json','mask_results.jsonl','primary_and_paired.csv','readout.json']});atomic(OUT/'scoring_manifest.json',job);print(json.dumps({'status':'complete','elapsed_seconds':job['elapsed_seconds'],'critical':critical,'means':rows},indent=2))
 except BaseException as exc:job.update(status='failed',finished_utc=now(),error=repr(exc));atomic(OUT/'scoring_manifest.json',job);raise
 finally:fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','probe','encode','score']);p.add_argument('--study',choices=list(STUDIES));p.add_argument('--max-seconds',type=int,default=1650);a=p.parse_args()
 if a.stage=='freeze':assert not FREEZE.exists();atomic(FREEZE,dict(status='posthoc_plan_frozen_before_encoder_outcomes',created_utc=now(),**fingerprints()));print('Frozen',sha(FREEZE))
 elif a.stage=='probe':probe()
 elif a.stage=='encode':assert a.study and 1<=a.max_seconds<=1650;encode(a.study,a.max_seconds)
 else:score_all()
if __name__=='__main__':main()
