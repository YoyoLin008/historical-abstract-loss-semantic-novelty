"""Versioned SPECTER encoder with verified weights, environment, and safe reuse."""
import argparse,datetime,hashlib,json,os,time,fcntl,resource,importlib.metadata,sys
from pathlib import Path
import numpy as np
from src.encode import MODEL, REV, sha, atomic

def verify_model(root):
 m=json.loads((root/'data/manifests/model_files.json').read_text())
 if m['model']!=MODEL or m['revision']!=REV:raise ValueError('Model identity mismatch')
 actual={f.name for f in (root/'data/model').iterdir() if f.is_file()}
 if actual!=set(m['files']):raise ValueError('Unmanifested or missing model files')
 for name,expected in m['files'].items():
  if sha(root/'data/model'/name)!=expected:raise ValueError('Model file checksum mismatch: '+name)
 for line in (root/'requirements.lock').read_text().splitlines():
  if line.strip() and not line.startswith('#'):
   name,version=line.split('==');assert importlib.metadata.version(name)==version,'Dependency drift: '+name
 return m

def reuse_arrays(root,rows,enabled=True):
 if any(not r.get('id') for r in rows) or len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate or empty destination IDs')
 for r in rows:
  for field in ['title','abstract']:
   if not isinstance(r.get(field),str) or not r[field].strip() or hashlib.sha256(r[field].encode()).hexdigest()!=r.get(field+'_sha256'):raise ValueError('Destination text or hash invalid: '+r['id'])
 if not enabled:return {},None
 orig=root/'data/processed/records.jsonl';folder=root/'data/embeddings';m=json.loads((folder/'encoding_manifest.json').read_text());assert m['status']=='complete';assert m['fingerprint']['data_sha256']==sha(orig);assert m['fingerprint']['model']==MODEL and m['fingerprint']['revision']==REV
 for mode in ['full','title']:assert m['output_sha256'][mode]==sha(folder/(mode+'.npy'))
 expected={'pooling':'last_hidden_state_CLS','normalize':'L2','truncation':512,'input':'title + [SEP] + abstract; title-only title + [SEP]','dtype':'float32','device':'cpu','batch':8,'chunk':64}
 if any(m['fingerprint'].get(k)!=v for k,v in expected.items()):raise ValueError('Legacy encoding settings mismatch')
 source=[json.loads(x) for x in orig.read_text().splitlines()]
 if len({r['id'] for r in source})!=len(source):raise ValueError('Duplicate legacy IDs')
 for r in source+rows:
  for field in ['title','abstract']:
   if hashlib.sha256(r[field].encode()).hexdigest()!=r[field+'_sha256']:raise ValueError('Text hash mismatch: '+r['id'])
 arrays={mode:np.load(folder/(mode+'.npy'),mmap_mode='r') for mode in ['full','title']}
 for arr in arrays.values():
  if arr.shape!=(len(source),768) or arr.dtype!=np.float32 or not np.isfinite(arr).all() or not np.allclose(np.linalg.norm(arr,axis=1),1,atol=1e-5):raise ValueError('Invalid legacy embedding array')
 index={r['id']:i for i,r in enumerate(source)};mapping={}
 for j,r in enumerate(rows):
  if r['id'] in index:
   old=source[index[r['id']]]
   if (old['title_sha256'],old['abstract_sha256'])==(r['title_sha256'],r['abstract_sha256']):mapping[j]=index[r['id']]
 return mapping,{'manifest_sha256':sha(folder/'encoding_manifest.json'),'records_sha256':sha(orig),'outputs':m['output_sha256'],'arrays':arrays}

def main():
 p=argparse.ArgumentParser();p.add_argument('--records',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--benchmark',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--no-reuse',action='store_true');p.add_argument('--max-chunks',type=int);a=p.parse_args();root=Path.cwd();rows=[json.loads(x) for x in a.records.read_text().splitlines()];a.output.mkdir(parents=True,exist_ok=True)
 started=time.perf_counter();model_files=verify_model(root);reuse,legacy=reuse_arrays(root,rows,not a.no_reuse)
 import torch,transformers
 from transformers import AutoTokenizer,AutoModel
 torch.set_num_threads(4);torch.manual_seed(2026100401);tok=AutoTokenizer.from_pretrained(root/'data/model',local_files_only=True,trust_remote_code=False);model=AutoModel.from_pretrained(root/'data/model',local_files_only=True,trust_remote_code=False,use_safetensors=False,attn_implementation='eager').eval()
 def inputs(mode,ix):return [rows[i]['title']+tok.sep_token+(rows[i]['abstract'] if mode=='full' else '') for i in ix]
 def embed(texts):
  tokens=tok(texts,padding=True,truncation=True,max_length=512,return_tensors='pt')
  with torch.inference_mode():values=torch.nn.functional.normalize(model(**tokens).last_hidden_state[:,0,:],p=2,dim=1)
  return values.numpy().astype('float32')
 check=list(reuse)[:8]
 agreement={mode:float(np.max(abs(embed(inputs(mode,check))-legacy['arrays'][mode][[reuse[i] for i in check]]))) for mode in ['full','title']} if check else {}
 if any(v>=2e-6 for v in agreement.values()):raise ValueError('Legacy reuse numerical mismatch')
 fingerprint={'schema':2,'data_sha256':sha(a.records),'model':MODEL,'revision':REV,'model_files':model_files['files'],'requirements_lock_sha256':sha(root/'requirements.lock'),'code_sha256':sha(__file__),'python':sys.version.split()[0],'numpy':np.__version__,'torch':torch.__version__,'transformers':transformers.__version__,'pooling':'last_hidden_state_CLS','normalize':'L2','truncation':512,'input':'title + [SEP] + abstract; title-only title + [SEP]','dtype':'float32','device':'cpu','threads':4,'batch':8,'chunk':64,'legacy_sample_max_abs_difference':agreement,'reuse':None if not legacy else {k:v for k,v in legacy.items() if k!='arrays'}}
 if a.benchmark:
  todo=[i for i in range(len(rows)) if i not in reuse][:32] or list(range(min(32,len(rows))));embed(inputs('full',todo[:8]));timing={}
  for mode in ['full','title']:
   t=time.perf_counter()
   for start in range(0,len(todo),8):embed(inputs(mode,todo[start:start+8]))
   timing[mode]=time.perf_counter()-t
  check=list(reuse)[:8];agreement={}
  for mode in ['full','title']:
   if check:agreement[mode]=float(np.max(abs(embed(inputs(mode,check))-legacy['arrays'][mode][[reuse[i] for i in check]])))
  assert all(v<2e-6 for v in agreement.values()),'Legacy reuse numerical settings mismatch'
  result={'fingerprint':fingerprint,'sample_new_records':len(todo),'total_records':len(rows),'reuse_records':len(reuse),'new_records':len(rows)-len(reuse),'measured_seconds':timing,'estimated_new_encoding_seconds':sum(timing.values())/len(todo)*(len(rows)-len(reuse)),'old_vector_reencoding_max_abs_error':agreement,'peak_rss_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'elapsed_seconds_including_setup':time.perf_counter()-started}
  atomic(a.output/'benchmark.json',result);print(json.dumps(result,indent=2));return
 lock=(a.output/'encode.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);statepath=a.output/'encoding_manifest.json'
 if statepath.exists():
  state=json.loads(statepath.read_text());assert state['fingerprint']==fingerprint,'Changed encoding fingerprint'
  if not a.resume:raise ValueError('Existing output requires --resume')
 else:state={'status':'running','fingerprint':fingerprint,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'chunks':{},'reused_record_count':len(reuse)}
 atomic(statepath,state);computed=0
 for mode in ['full','title']:
  arrays=[]
  for offset in range(0,len(rows),64):
   key=f'{mode}_{offset:06d}';path=a.output/(key+'.npy')
   if key in state['chunks']:
    assert sha(path)==state['chunks'][key]['sha256'];arrays.append(np.load(path));continue
   if a.max_chunks is not None and computed>=a.max_chunks:
    state['status']='partial';atomic(statepath,state);print(json.dumps({'status':'partial','new_chunks':computed}));return
   t=time.perf_counter();ix=list(range(offset,min(offset+64,len(rows))));v=np.empty((len(ix),768),dtype=np.float32);new=[i for i in ix if i not in reuse]
   for pos,i in enumerate(ix):
    if i in reuse:v[pos]=legacy['arrays'][mode][reuse[i]]
   for st in range(0,len(new),8):
    batch=new[st:st+8];res=embed(inputs(mode,batch))
    for i,z in zip(batch,res):v[i-offset]=z
   lens=[len(tok(x,truncation=False)['input_ids']) for x in inputs(mode,ix)];assert np.isfinite(v).all() and np.allclose(np.linalg.norm(v,axis=1),1,atol=1e-5)
   temp=path.with_suffix('.tmp')
   with temp.open('wb') as f:np.save(f,v)
   temp.replace(path);arrays.append(v);computed+=1;state['chunks'][key]={'sha256':sha(path),'count':len(v),'reused':len(ix)-len(new),'truncated':sum(n>512 for n in lens),'seconds':time.perf_counter()-t};state['status']='running';atomic(statepath,state);print(json.dumps({'mode':mode,'completed':offset+len(ix),'total':len(rows),'elapsed_seconds':time.perf_counter()-started}),flush=True)
  path=a.output/(mode+'.npy');temp=path.with_suffix('.tmp')
  with temp.open('wb') as f:np.save(f,np.concatenate(arrays))
  temp.replace(path)
 state.update(status='complete',output_sha256={m:sha(a.output/(m+'.npy')) for m in ['full','title']},completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started,peak_rss_bytes_macos=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);atomic(statepath,state)
if __name__=='__main__':main()
