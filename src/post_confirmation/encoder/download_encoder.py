from pathlib import Path
import json,hashlib,requests,time
ROOT=Path(__file__).resolve().parents[3]/'external_work';models=ROOT/'encoder_models';models.mkdir(exist_ok=True)
receipt=[];total=0
for meta_name,sub in [('specter2_base_metadata.json','base'),('specter2_adapter_metadata.json','proximity')]:
 meta=json.loads((ROOT/meta_name).read_text());target=models/sub;target.mkdir(exist_ok=True)
 for row in meta['siblings']:
  name=row['rfilename']
  if name=='.gitattributes':continue
  assert '/' not in name and name not in ['.','..'];size=row['size'];assert total+size<=1000000000
  p=target/name;expected=row.get('lfs',{}).get('sha256');url='https://huggingface.co/'+meta['id']+'/resolve/'+meta['sha']+'/'+name
  if not p.exists() or p.stat().st_size!=size:
   response=requests.get(url,stream=True,timeout=(20,120));response.raise_for_status();tmp=p.with_suffix(p.suffix+'.partial')
   with tmp.open('wb') as f:
    for chunk in response.iter_content(1048576):
     if chunk:f.write(chunk)
   assert tmp.stat().st_size==size;tmp.replace(p)
  digest=hashlib.sha256(p.read_bytes()).hexdigest()
  if expected:assert digest==expected,(name,digest,expected)
  total+=size;receipt.append({'model':meta['id'],'revision':meta['sha'],'filename':name,'url':url,'bytes':size,'sha256':digest,'registry_lfs_sha256':expected})
  print(name,size,'verified',flush=True)
(models/'download_manifest.json').write_text(json.dumps({'status':'complete','actual_total_bytes':total,'files':receipt},indent=2)+'\n')
