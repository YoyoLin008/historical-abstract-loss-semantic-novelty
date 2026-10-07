"""Acquire immutable official ACL XML and audit a declared venue/year frame."""
import argparse, collections, datetime, hashlib, json, os, re, time, urllib.request
from pathlib import Path
from xml.etree import ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
COMMIT='0e1bf6e9d40de745a39a59d437ee4f12ffe6118f'
FILES=['P18.xml','D18.xml','P19.xml','D19.xml','2020.acl.xml','2020.emnlp.xml']

def sha(b): return hashlib.sha256(b).hexdigest()
def atomic(p,data):
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+'.tmp');t.write_text(data);t.replace(p)
def fetch(url,path):
 if path.exists():return path.read_bytes()
 for attempt in range(3):
  try:
   with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'HistoricalAbstractMissingnessResearch/0.1'}),timeout=60) as r: b=r.read()
   t=path.with_suffix(path.suffix+'.tmp');t.parent.mkdir(parents=True,exist_ok=True);t.write_bytes(b);t.replace(path);return b
  except Exception:
   if attempt==2:raise
   time.sleep(2**attempt)
def text(e):return '' if e is None else ' '.join(''.join(e.itertext()).split())
def topic(title):
 patterns=[r'translat',r'pars|syntax|syntactic',r'question|answer|comprehension',r'sentiment|emotion|opinion',r'extract|entit|relation',r'generat|summari',r'semantic|represent|embedding']
 return next((i for i,p in enumerate(patterns) if re.search(p,title,re.I)),7)
def main():
 a=argparse.ArgumentParser();a.add_argument('--dry-run',action='store_true');args=a.parse_args()
 if args.dry_run:print(json.dumps({'commit':COMMIT,'files':FILES,'frame':'ACL/EMNLP main long+short 2018-2020; no Findings/workshops/demos'}));return
 raw=ROOT/'data/raw';raw.mkdir(exist_ok=True,parents=True);existing_path=ROOT/'data/manifests/acquisition.json';expected={x['file']:x for x in json.loads(existing_path.read_text())} if existing_path.exists() else {};manifest=[];records=[];ledger=[];volumes=[]
 for name in FILES+['schema.rnc']:
  url=f'https://raw.githubusercontent.com/acl-org/acl-anthology/{COMMIT}/data/xml/{name}'
  b=fetch(url,raw/name)
  if name in expected:assert sha(b)==expected[name]['sha256'] and expected[name]['commit']==COMMIT,'Immutable raw input mismatch'
  manifest.append({'file':name,'url':url,'commit':COMMIT,'sha256':sha(b),'retrieved_utc':datetime.datetime.fromtimestamp((raw/name).stat().st_mtime,datetime.timezone.utc).isoformat(),'bytes':len(b)})
  if name.endswith('.rnc'):continue
  root=ET.fromstring(b)
  for v in root.findall('volume'):
   vid=v.attrib['id'];meta=v.find('meta');vt=text(meta.find('booktitle'));year=int(text(meta.find('year')))
   selected=(name=='P18.xml' and vid in ['1','2']) or (name=='P19.xml' and vid=='1') or (name in ['D18.xml','D19.xml'] and vid=='1') or (name.startswith('2020.') and vid=='main')
   volumes.append({'file':name,'volume':vid,'title':vt,'year':year,'selected':selected,'count':len(v.findall('paper')),'frontmatter_elements_excluded':len(v.findall('frontmatter'))})
   for p in v.findall('paper'):
    pid=p.attrib['id'];old=bool(re.fullmatch('[A-Z][0-9]{2}',root.attrib['id']));identifier=root.attrib['id']+'-'+vid+pid.zfill(4-len(vid)) if old else root.attrib['id']+'-'+vid+'.'+pid
    title=text(p.find('title'));abstract=text(p.find('abstract'));doi=text(p.find('doi')).lower().removeprefix('https://doi.org/').removeprefix('doi:').strip();reasons=[]
    if not selected:reasons.append('outside_declared_main_venue_frame')
    if pid=='0' or not p.findall('author'):reasons.append('front_matter_or_no_authors')
    if not title:reasons.append('missing_title')
    if not abstract:reasons.append('absent_or_blank_abstract')
    elif abstract.strip().lower() in ['n/a','none','null','abstract','no abstract available']:reasons.append('placeholder_abstract')
    r={'id':identifier,'year':year,'source':'ACL' if name.startswith('P') or '.acl.' in name else 'EMNLP','volume':vid,'title':title,'abstract':abstract,'doi':doi,'topic':topic(title),'url':f'https://aclanthology.org/{identifier}/','title_sha256':sha(title.encode()),'abstract_sha256':sha(abstract.encode()),'source_file':name,'source_commit':COMMIT,'language':'not_explicitly_encoded_in_source','has_abstract_element':p.find('abstract') is not None}
    ledger.append({'id':identifier,'year':year,'source':r['source'],'in_declared_frame':selected,'abstract_present':bool(abstract),'abstract_element_present':p.find('abstract') is not None,'exclusion_reasons':reasons})
    if not reasons:records.append(r)
 seenid=set();seendoi=set();seentext=set();dedup=[];duplicate=[]
 for r in sorted(records,key=lambda r:(r['year'],r['id'])):
  reason='duplicate_id' if r['id'] in seenid else 'duplicate_doi' if r['doi'] and r['doi'] in seendoi else 'duplicate_exact_title_abstract' if (r['title_sha256'],r['abstract_sha256']) in seentext else None
  if reason:
   duplicate.append({'id':r['id'],'reason':reason});next(z for z in ledger if z['id']==r['id'])['exclusion_reasons'].append(reason)
  else:dedup.append(r);seenid.add(r['id']);seendoi.add(r['doi']) if r['doi'] else None;seentext.add((r['title_sha256'],r['abstract_sha256']))
 records=dedup
 profile=[]
 for year,source in sorted({(x['year'],x['source']) for x in ledger if x['in_declared_frame']}):
  subset=[x for x in ledger if x['in_declared_frame'] and x['year']==year and x['source']==source]
  profile.append({'year':year,'source':source,'records_in_frame':len(subset),'missing_abstract':sum(not x['abstract_present'] for x in subset),'eligible_complete':sum(not x['exclusion_reasons'] for x in subset)})
 repeated_titles=[{'title_hash':h,'ids':[r['id'] for r in records if r['title_sha256']==h]} for h,c in collections.Counter(r['title_sha256'] for r in records).items() if c>1]
 audit={'commit':COMMIT,'all_records':len(ledger),'eligible_records':len(records),'profile':profile,'duplicate_records':duplicate,'repeated_exact_title_hashes':repeated_titles,'topics':dict(collections.Counter(r['topic'] for r in records)),'language_limitation':'XML does not explicitly identify language; English venue context is not per-record language validation.','version_limitation':'ID/DOI and exact title+abstract dedup; lowest chronological/id record retained, exclusions logged; no fuzzy version merging.'}
 for name,rows in [('records',records),('exclusions',ledger)]:atomic(ROOT/f'data/processed/{name}.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 for name,data in [('acquisition',manifest),('volumes',volumes),('data_audit',audit)]:atomic(ROOT/f'data/manifests/{name}.json',json.dumps(data,indent=2))
 print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
