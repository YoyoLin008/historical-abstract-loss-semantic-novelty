"""Pinned SPECTER encodings; local inference; immutable resumable chunks."""
import argparse,datetime,hashlib,json,os,time,fcntl
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
MODEL='allenai/specter';REV='81cbfb43d4fc2e728d5b4201ce14987db8d0854c'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def atomic(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');p.add_argument('--benchmark',action='store_true');p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=8);p.add_argument('--chunk-size',type=int,default=64);p.add_argument('--dry-run',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
 records_path=ROOT/'data/processed/records.jsonl';rows=[json.loads(x) for x in records_path.read_text().splitlines()]
 if a.dry_run:print(json.dumps({'records':len(rows),'representations':2,'model':MODEL,'revision':REV,'batch':a.batch_size,'chunk':a.chunk_size}));return
 if a.download:
  from huggingface_hub import snapshot_download
  folder=snapshot_download(MODEL,revision=REV,local_dir=ROOT/'data/model',allow_patterns=['*.json','vocab.txt','pytorch_model.bin','README.md'],max_workers=2)
  atomic(ROOT/'data/manifests/model_files.json',{'model':MODEL,'revision':REV,'files':{f.name:sha(f) for f in Path(folder).iterdir() if f.is_file()}});print(folder,flush=True);return
 import torch
 from transformers import AutoTokenizer,AutoModel
 torch.set_num_threads(4);torch.manual_seed(20261004)
 tok=AutoTokenizer.from_pretrained(ROOT/'data/model',local_files_only=True)
 model=AutoModel.from_pretrained(ROOT/'data/model',local_files_only=True,attn_implementation='eager').eval()
 def texts(mode,part):return [r['title']+tok.sep_token+(r['abstract'] if mode=='full' else '') for r in part]
 def embed(part,device):
  inputs=tok(part,padding=True,truncation=True,max_length=512,return_tensors='pt').to(device)
  with torch.inference_mode():v=model(**inputs).last_hidden_state[:,0,:]
  return torch.nn.functional.normalize(v,p=2,dim=1).cpu().numpy().astype('float32')
 if a.benchmark:
  out={'model':MODEL,'revision':REV,'torch':torch.__version__,'mps_available':torch.backends.mps.is_available(),'batch':a.batch_size,'threads':4,'sample_count':min(32,len(rows)),'timings':{}}
  sample=texts('full',rows[:32]);emb={}
  for device in ['cpu']+(['mps'] if torch.backends.mps.is_available() else []):
   model.to(device);embed(sample[:a.batch_size],device)
   t=time.perf_counter();emb[device]=np.concatenate([embed(sample[i:i+a.batch_size],device) for i in range(0,len(sample),a.batch_size)]);out['timings'][device]={'seconds':time.perf_counter()-t}
  if 'mps'in emb:out['cpu_mps_max_abs_component_difference']=float(np.max(np.abs(emb['cpu']-emb['mps'])))
  atomic(ROOT/'data/manifests/encoder_benchmark.json',out);print(json.dumps(out,indent=2));return
 out=ROOT/'data/embeddings';out.mkdir(parents=True,exist_ok=True)
 lock=(out/'encode.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 model.to(a.device);fingerprint={'data_sha256':sha(records_path),'model':MODEL,'revision':REV,'pooling':'last_hidden_state_CLS','normalize':'L2','truncation':512,'input':'title + [SEP] + abstract; title-only title + [SEP]','dtype':'float32','device':a.device,'batch':a.batch_size,'chunk':a.chunk_size,'code_sha256':sha(__file__)}
 state_path=out/'encoding_manifest.json'
 if state_path.exists():
  prev=json.loads(state_path.read_text());assert prev['fingerprint']==fingerprint,'Changed fingerprint; use separate output directory'
  if not a.resume:raise SystemExit('Use --resume for existing encoding run')
 else:prev={'fingerprint':fingerprint,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'chunks':{},'status':'running'}
 start=time.perf_counter()
 for mode in ['full','title']:
  chunks=[]
  for offset in range(0,len(rows),a.chunk_size):
   key=f'{mode}_{offset:06d}';path=out/(key+'.npy')
   if key in prev['chunks']:
    assert path.exists() and sha(path)==prev['chunks'][key]['sha256'];chunks.append(np.load(path));continue
   part=texts(mode,rows[offset:offset+a.chunk_size]);lengths=[len(tok(x,truncation=False)['input_ids']) for x in part];t=time.perf_counter()
   v=np.concatenate([embed(part[i:i+a.batch_size],a.device) for i in range(0,len(part),a.batch_size)])
   assert np.isfinite(v).all() and np.allclose(np.linalg.norm(v,axis=1),1,atol=1e-5)
   tmp=path.with_suffix('.tmp')
   with tmp.open('wb') as f:np.save(f,v)
   tmp.replace(path);chunks.append(v)
   prev['chunks'][key]={'sha256':sha(path),'count':len(v),'truncated':sum(x>512 for x in lengths),'seconds':time.perf_counter()-t};atomic(state_path,prev)
   print(json.dumps({'mode':mode,'completed':offset+len(v),'total':len(rows),'elapsed_seconds':round(time.perf_counter()-start,2)}),flush=True)
  dest=out/(mode+'.npy');tmp=dest.with_suffix('.tmp')
  with tmp.open('wb') as f:np.save(f,np.concatenate(chunks))
  tmp.replace(dest)
 prev['status']='complete';prev['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();prev['output_sha256']={m:sha(out/(m+'.npy')) for m in ['full','title']};atomic(state_path,prev)
if __name__=='__main__':main()
