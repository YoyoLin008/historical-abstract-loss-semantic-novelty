"""Metadata-only construction of the first untouched qualifying ACL cohort."""
import argparse,datetime,json,hashlib,collections
from pathlib import Path
from xml.etree import ElementTree as ET
from src.acquire import COMMIT, fetch, text, topic, sha, atomic

def deduplicate(rows):
 seen_id={};seen_doi={};seen_text={};out=[];excluded=[]
 for r in sorted(rows,key=lambda r:(r['year'],r['id'])):
  pair=(r['title_sha256'],r['abstract_sha256'])
  reason='duplicate_id' if r['id'] in seen_id else 'duplicate_doi' if r['doi'] and r['doi'] in seen_doi else 'duplicate_exact_title_abstract' if pair in seen_text else None
  if reason:
   canonical=seen_id.get(r['id']) or (seen_doi.get(r['doi']) if r['doi'] else None) or seen_text.get(pair)
   excluded.append({'id':r['id'],'year':r['year'],'source':r['source'],'reason':reason,'canonical_id':canonical['id'],'doi':r['doi'],'title_sha256':r['title_sha256'],'abstract_sha256':r['abstract_sha256']});continue
  out.append(r);seen_id[r['id']]=r;seen_text[pair]=r
  if r['doi']:seen_doi[r['doi']]=r
 return out,excluded

def main():
 a=argparse.ArgumentParser();a.add_argument('--root',type=Path,default=Path.cwd());a.add_argument('--year',type=int,default=2021);arg=a.parse_args();p=arg.root;y=arg.year
 if y!=2021:raise SystemExit('2021 is predetermined first candidate; amend explicitly only after recorded failure.')
 original=p/'data/processed/records.jsonl';prior_manifest=json.loads((p/'data/embeddings/encoding_manifest.json').read_text());assert sha(original.read_bytes())==prior_manifest['fingerprint']['data_sha256'],'Original record file changed';tree=json.loads((p/'data/manifests/source_tree.json').read_text());blob_hashes={x['path']:x['sha'] for x in tree['tree']};rows=[json.loads(s) for s in original.read_text().splitlines()];assert set(r['year'] for r in rows)=={2018,2019,2020}
 files=[];ledger=[];volumes=[]
 for venue in ['acl','emnlp']:
  name=f'{y}.{venue}.xml';url=f'https://raw.githubusercontent.com/acl-org/acl-anthology/{COMMIT}/data/xml/{name}';path=p/'data/raw'/name;data=fetch(url,path);expected_blob=blob_hashes['data/xml/'+name];actual_blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest();assert actual_blob==expected_blob,'Raw XML does not match pinned source tree';files.append({'path':str(path.relative_to(p)),'sha256':sha(data),'url':url,'source_commit':COMMIT,'git_blob_sha1':actual_blob,'retrieved_utc':datetime.datetime.fromtimestamp(path.stat().st_mtime,datetime.timezone.utc).isoformat()});root=ET.fromstring(data)
  assert root.attrib['id']==f'{y}.{venue}'
  for v in root.findall('volume'):
   vid=v.attrib['id'];meta=v.find('meta');year=int(text(meta.find('year')));assert year==y;selected=vid in ('main','long','short');volumes.append({'source':venue.upper(),'year':year,'volume':vid,'title':text(meta.find('booktitle')),'selected':selected,'paper_count':len(v.findall('paper')),'frontmatter_count':len(v.findall('frontmatter'))})
   for paper in v.findall('paper'):
    pid=f'{root.attrib["id"]}-{vid}.{paper.attrib["id"]}';t=text(paper.find('title'));ab=text(paper.find('abstract'));doi=text(paper.find('doi')).lower().removeprefix('https://doi.org/').removeprefix('doi:').strip();reasons=[]
    if not selected:reasons.append('outside_main_venue_frame')
    if not paper.findall('author') or paper.attrib['id']=='0':reasons.append('frontmatter_or_no_authors')
    if not t:reasons.append('missing_title')
    if not ab:reasons.append('missing_abstract')
    elif ab.lower() in ['n/a','none','null','abstract','no abstract available']:reasons.append('placeholder_abstract')
    ledger.append({'id':pid,'year':year,'source':venue.upper(),'selected_main':selected,'title_present':bool(t),'abstract_present':bool(ab),'reasons':reasons})
    if not reasons:rows.append({'id':pid,'year':year,'source':venue.upper(),'volume':vid,'title':t,'abstract':ab,'doi':doi,'topic':topic(t),'url':f'https://aclanthology.org/{pid}/','title_sha256':sha(t.encode()),'abstract_sha256':sha(ab.encode()),'source_file':name,'source_commit':COMMIT,'language':'not_explicitly_encoded_in_source','has_abstract_element':paper.find('abstract') is not None})
 rows,duplicates=deduplicate(rows)
 for r in duplicates:
  for x in ledger:
   if x['id']==r['id']:x['reasons'].append(r['reason'])
 profile=[]
 for venue in ['ACL','EMNLP']:
  sub=[r for r in ledger if r['selected_main'] and r['source']==venue];eligible=[r for r in rows if r['year']==y and r['source']==venue]
  profile.append({'source':venue,'year':y,'main_papers':len(sub),'complete_before_dedup':sum(not any(z in r['reasons'] for z in ['missing_title','missing_abstract','placeholder_abstract','frontmatter_or_no_authors']) for r in sub),'missing_title':sum(not r['title_present'] for r in sub),'missing_abstract':sum(not r['abstract_present'] for r in sub),'eligible_focal':len(eligible)})
 history=[r for r in rows if r['year']<y];focals=[r for r in rows if r['year']==y];passed=len(history)>=2000 and all(r['eligible_focal']>=400 and r['complete_before_dedup']/r['main_papers']>=.95 for r in profile)
 out=p/'data/confirmation_v1';out.mkdir(exist_ok=True,parents=True)
 content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
 if (out/'records.jsonl').exists():assert sha((out/'records.jsonl').read_bytes())==sha(content.encode()),'Refuse changed cohort'
 atomic(out/'records.jsonl',content);atomic(out/'exclusions.jsonl',''.join(json.dumps(r)+'\n' for r in ledger))
 manifest={'stage':'confirmatory_metadata_audit','cohort_year':y,'source_commit':COMMIT,'prior_records_sha256':sha(original.read_bytes()),'records_sha256':sha(content.encode()),'metadata_files':files,'quality_rule':'docs/cohort_selection_v1.md','quality_pass':passed,'profile':profile,'history_count':len(history),'focal_count':len(focals),'new_duplicate_exclusions':duplicates,'title_only_repeated_groups':[[r['id'] for r in rows if r['title_sha256']==h] for h,n in collections.Counter(r['title_sha256'] for r in rows).items() if n>1],'volumes':volumes,'history':[ {k:r[k] for k in ['id','year','source','topic','doi','title_sha256','abstract_sha256']} for r in history],'focal':[{k:r[k] for k in ['id','year','source','topic','doi','title_sha256','abstract_sha256']} for r in focals],'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 atomic(out/'cohort_manifest.json',json.dumps(manifest,indent=2));print(json.dumps({k:v for k,v in manifest.items() if k not in ['history','focal','volumes']},indent=2))
 if not passed:raise SystemExit('First cohort fails declared quality criteria; document before considering2022')
if __name__=='__main__':main()
