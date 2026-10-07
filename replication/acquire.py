"""Public PLOS research DOI frame joined exactly to dated PubMed metadata.
No outcome scores; current snapshot, bounded cached retrieval with honest losses.
"""
from pathlib import Path
from datetime import datetime,timezone,date
from collections import Counter,defaultdict
import argparse,hashlib,json,html,re,time,unicodedata,os,tempfile
from urllib.parse import urlencode,urlsplit
from urllib.request import Request,urlopen
import xml.etree.ElementTree as ET

SOURCES={'PLOSCB':('PLOS Computational Biology','PLoS Comput Biol'),'PLOSGEN':('PLOS Genetics','PLoS Genet')}
YEARS=[2018,2019,2020,2021]
EXCLUDED_TYPES={'Review','Systematic Review','Meta-Analysis','Editorial','Letter','Comment','News','Published Erratum','Retraction of Publication','Retracted Publication','Expression of Concern','Corrected and Republished Article'}
PLACEHOLDERS={'n/a','na','none','not available','no abstract available','abstract not available'}
def now():return datetime.now(timezone.utc).isoformat()
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha(p):return sha_bytes(Path(p).read_bytes())
def clean(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFC',html.unescape(s or ''))).strip()
def text(e):return clean(''.join(e.itertext())) if e is not None else ''
def doi(s):
 s=clean(s).lower();return re.sub(r'^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)','',s).strip()
def substantive(s):return bool(s) and s.casefold() not in PLACEHOLDERS
def describe(values):
 values=sorted(values);n=len(values)
 return {'n':n,'min':values[0] if n else None,'median':(values[(n-1)//2]+values[n//2])/2 if n else None,'max':values[-1] if n else None}
def atomic(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile('w',dir=p.parent,delete=False,encoding='utf-8') as f:
  json.dump(value,f,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno());tmp=f.name
 os.replace(tmp,p)
class Client:
 def __init__(self,folder):self.folder=folder;self.last={};self.calls=[];folder.mkdir(parents=True,exist_ok=True)
 def get(self,label,url):
  raw=self.folder/(label+'.response');meta=self.folder/(label+'.json')
  if raw.exists() and meta.exists():
   m=json.loads(meta.read_text());assert m['url']==url and m['sha256']==sha(raw),'Immutable cache mismatch';self.calls.append(m);return raw.read_bytes()
  host=urlsplit(url).hostname;interval=6.2 if host=='api.plos.org' else .6
  time.sleep(max(0,self.last.get(host,0)+interval-time.monotonic()));self.last[host]=time.monotonic()
  with urlopen(Request(url,headers={'User-Agent':'historical-abstract-missingness-research/1.0 (public metadata study)'}),timeout=60) as response:
   body=response.read(50_000_001);status=response.status;headers=dict(response.headers)
  if len(body)>50_000_000:raise RuntimeError('Oversized response')
  if status!=200:raise RuntimeError('HTTP failure, not missingness: '+str(status))
  m={'label':label,'url':url,'retrieved_utc':now(),'sha256':sha_bytes(body),'bytes':len(body),'status':status,'headers':headers}
  raw.write_bytes(body);atomic(meta,m);self.calls.append(m);return body
 def plos(self,source,year,start):
  query=f'journal:"{SOURCES[source][0]}" AND article_type:"Research Article" AND doc_type:full AND publication_date:[{year}-01-01T00:00:00Z TO {year}-12-31T23:59:59Z]'
  params={'q':query,'wt':'json','rows':100,'start':start,'sort':'id asc','fl':'id,journal,article_type,publication_date,doc_type,title_display,author_display,copyright'}
  return json.loads(self.get(f'plos_{source}_{year}_{start:05d}','https://api.plos.org/search?'+urlencode(params)))['response']
 def ids(self,source,year):
  params={'db':'pubmed','term':f'"{SOURCES[source][1]}"[ta] AND {year}[epdat]','retmode':'json','retmax':10000,'tool':'historical_abstract_missingness_research'}
  result=json.loads(self.get(f'pubmed_ids_{source}_{year}','https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?'+urlencode(params)))['esearchresult']
  assert int(result['count'])==len(result['idlist'])<10000 and len(set(result['idlist']))==len(result['idlist']),'Incomplete or duplicate search IDs';return result['idlist']
 def doi_ids(self,dois,start):
  assert 0<len(dois)<=20 and len(set(dois))==len(dois),'Invalid rescue DOI batch'
  assert all('"' not in d for d in dois),'Unexpected quote in publisher DOI'
  params={'db':'pubmed','term':' OR '.join('"'+d+'"[aid]' for d in dois),'retmode':'json','retmax':10000,'tool':'historical_abstract_missingness_research'}
  result=json.loads(self.get(f'pubmed_rescue_ids_{start:05d}','https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?'+urlencode(params)))['esearchresult']
  assert int(result['count'])==len(result['idlist'])<10000 and len(set(result['idlist']))==len(result['idlist']),'Incomplete or duplicate rescue IDs'
  return result['idlist']
 def articles(self,source,year,ids,start):
  params={'db':'pubmed','id':','.join(ids),'retmode':'xml','tool':'historical_abstract_missingness_research'}
  root=ET.fromstring(self.get(f'pubmed_xml_{source}_{year}_{start:05d}','https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?'+urlencode(params)))
  elements=root.findall('PubmedArticle');actual=[text(e.find('./MedlineCitation/PMID')) for e in elements]
  assert set(actual)==set(ids) and len(actual)==len(ids)==len(set(actual)),'Missing or duplicate PubMed batch IDs';return elements

def parse_pubmed(e):
 article=e.find('./MedlineCitation/Article');pmid=text(e.find('./MedlineCitation/PMID'))
 dates=[];date_error=None
 for d in article.findall('./ArticleDate'):
  if d.get('DateType')=='Electronic':
   try:dates.append(date(int(text(d.find('Year'))),int(text(d.find('Month'))),int(text(d.find('Day')))).isoformat())
   except (ValueError,TypeError):date_error='malformed_electronic_date'
 if not dates:date_error=date_error or 'missing_electronic_date'
 if len(set(dates))>1:date_error='ambiguous_electronic_date'
 parts=[];section_contents=[]
 for node in article.findall('./Abstract/AbstractText'):
  content=text(node);label=clean(node.get('Label',''))
  section_contents.append(content)
  if content:parts.append((label+': ' if label else '')+content)
 raw_dois=[text(d) for d in e.findall('./PubmedData/ArticleIdList/ArticleId') if d.get('IdType')=='doi'];ids=[doi(d) for d in raw_dois]
 authors=[];author_details=[]
 for a in article.findall('./AuthorList/Author'):
  item={key:text(a.find(key)) for key in ['LastName','ForeName','Initials','Suffix','CollectiveName']}
  name=item['CollectiveName'] or ' '.join(x for x in [item['ForeName'] or item['Initials'],item['LastName'],item['Suffix']] if x)
  authors.append(name);author_details.append(item)
 return {'pmid':pmid,'dois':sorted(set(ids)),'raw_dois':raw_dois,'journal':text(article.find('./Journal/Title')),'date':dates[0] if dates and not date_error else None,'date_error':date_error,'title':text(article.find('ArticleTitle')),'abstract':clean(' '.join(parts)),'has_substantive_abstract':any(substantive(s) for s in section_contents),'publication_types':[text(d) for d in article.findall('./PublicationTypeList/PublicationType')],'languages':[text(d) for d in article.findall('./Language')],'correction_relationships':[dict(ref_type=d.get('RefType'),pmid=text(d.find('PMID')),ref_source=text(d.find('RefSource'))) for d in e.findall('./MedlineCitation/CommentsCorrectionsList/CommentsCorrections')],'authors':authors,'author_details':author_details,'copyright':text(article.find('./Abstract/CopyrightInformation'))}

def run(root):
 data=root/'replication/data';data.mkdir(parents=True,exist_ok=True);client=Client(data/'raw');candidates=[];publisher_counts={};pubmed={}
 for year in YEARS:
  for source in SOURCES:
   start=0;expected=None;seen=set()
   while expected is None or start<expected:
    page=client.plos(source,year,start)
    if expected is None:expected=page['numFound']
    assert page['numFound']==expected and page['start']==start,'Moving publisher count/page'
    assert len(page['docs'])==min(100,expected-start),'Incomplete publisher page'
    for r in page['docs']:
     rid=doi(r['id']);d=date.fromisoformat(r['publication_date'][:10])
     assert rid not in seen and r['doc_type']=='full' and r['article_type']=='Research Article' and r['journal']==SOURCES[source][0] and d.year==year,'Publisher frame mismatch'
     seen.add(rid);candidates.append({'doi':rid,'source':source,'year':year,'publisher_date':d.isoformat(),'publisher':r})
    start+=len(page['docs']);atomic(data/'progress.json',{'stage':'publisher_frame','source':source,'year':year,'retrieved':start,'expected':expected,'updated_utc':now()});print(json.dumps({'publisher':source,'year':year,'retrieved':start,'expected':expected}),flush=True)
   publisher_counts[f'{source}_{year}']=expected
 assert len({c['doi'] for c in candidates})==len(candidates),'Publisher DOI collision across years/journals'
 atomic(data/'publisher_frame.json',candidates)
 for year in YEARS:
  for source in SOURCES:
   ids=client.ids(source,year)
   for start in range(0,len(ids),200):
    for e in client.articles(source,year,ids[start:start+200],start):
     r=parse_pubmed(e)
     if r['pmid'] in pubmed:assert pubmed[r['pmid']]==r,'Conflicting repeated PubMed record'
     pubmed[r['pmid']]=r
    atomic(data/'progress.json',{'stage':'pubmed','source':source,'year':year,'retrieved':min(start+200,len(ids)),'expected':len(ids),'updated_utc':now()})
    print(json.dumps({'pubmed':source,'year':year,'retrieved':min(start+200,len(ids)),'expected':len(ids)}),flush=True)
 bydoi=defaultdict(list)
 for r in pubmed.values():
  for d in r['dois']:bydoi[d].append(r)
 # EPDAT is discovery only: an exact DOI rescue without a date filter identifies
 # undated or cross-year records before they are classified as unmatched.
 initial_unmatched=sorted(c['doi'] for c in candidates if not bydoi[c['doi']]);rescue_ids=set();rescue_lookups=[]
 for start in range(0,len(initial_unmatched),20):
  batch=initial_unmatched[start:start+20];ids=client.doi_ids(batch,start);rescue_ids.update(ids)
  rescue_lookups.append({'dois':batch,'returned_pmids':ids})
 new_pmids=sorted(rescue_ids-set(pubmed))
 for start in range(0,len(new_pmids),200):
  for e in client.articles('doi_rescue','all',new_pmids[start:start+200],start):
   r=parse_pubmed(e)
   assert r['pmid'] not in pubmed,'Rescue unexpectedly returned an existing PMID'
   pubmed[r['pmid']]=r
 bydoi=defaultdict(list)
 for r in pubmed.values():
  for d in r['dois']:bydoi[d].append(r)
 rescue={'initial_unmatched_dois':initial_unmatched,'lookups':rescue_lookups,'new_pmids_fetched':new_pmids,'unmatched_after_rescue':[d for d in initial_unmatched if not bydoi[d]]}
 atomic(data/'doi_rescue.json',rescue)
 ledger=[];eligible=[];date_disagreements=[]
 for c in candidates:
  reasons=[];hits=bydoi[c['doi']];r=hits[0] if len(hits)==1 else None
  if not hits:reasons.append('unmatched_pubmed_doi')
  elif len(hits)>1:reasons.append('multiple_pubmed_records_for_doi')
  if r:
   if len(r['dois'])!=1:reasons.append('conflicting_pubmed_dois')
   if r['journal'].casefold()!=SOURCES[c['source']][0].casefold():reasons.append('journal_mismatch')
   if r['date_error']:reasons.append(r['date_error'])
   elif int(r['date'][:4])!=c['year']:reasons.append('publisher_electronic_year_mismatch')
   elif r['date']!=c['publisher_date']:date_disagreements.append({'doi':c['doi'],'publisher':c['publisher_date'],'electronic':r['date']})
   if 'Journal Article' not in r['publication_types']:reasons.append('not_pubmed_journal_article')
   if set(r['publication_types'])&EXCLUDED_TYPES:reasons.append('excluded_publication_type')
   if any(d['ref_type'] in ['RetractionIn','ExpressionOfConcernIn'] for d in r['correction_relationships']):reasons.append('retraction_or_concern')
   if r['languages']!=['eng']:reasons.append('nonunique_english_language')
  valid=not reasons;has_title=substantive(r['title']) if r else False;has_abstract=r['has_substantive_abstract'] if r else False
  if r and not has_title:reasons.append('missing_title')
  if r and not has_abstract:reasons.append('missing_abstract')
  row={'doi':c['doi'],'source':c['source'],'publisher_year':c['year'],'publisher_date':c['publisher_date'],'matched_pmids':[h['pmid'] for h in hits],'date_type_id_valid':valid,'has_title':has_title,'has_abstract':has_abstract,'exclusion_reasons':reasons}
  if r:row.update(electronic_date=r['date'],publication_types=r['publication_types'],languages=r['languages'],correction_relationships=r['correction_relationships'],raw_pubmed_dois=r['raw_dois'],title_chars=len(r['title']),abstract_chars=len(r['abstract']),title_tokens=len(r['title'].split()),abstract_tokens=len(r['abstract'].split()))
  ledger.append(row)
  if not reasons:
   eligible.append({'id':c['doi'],'doi':c['doi'],'pmid':r['pmid'],'year':c['year'],'source':c['source'],'topic':0,'topic_note':'Unused placeholder; no topic intervention in replication','title':r['title'],'abstract':r['abstract'],'title_sha256':sha_bytes(r['title'].encode()),'abstract_sha256':sha_bytes(r['abstract'].encode()),'electronic_date':r['date'],'publisher_date':c['publisher_date'],'authors':r['authors'],'author_details':r['author_details'],'publisher_authors':c['publisher'].get('author_display',[]),'copyright':r['copyright'],'publisher_copyright':c['publisher'].get('copyright'),'source_url':'https://pubmed.ncbi.nlm.nih.gov/'+r['pmid']+'/','publisher_url':'https://doi.org/'+c['doi'],'publication_types':r['publication_types'],'languages':r['languages']})
 eligible.sort(key=lambda r:(r['electronic_date'],r['doi']));seen={};records=[];duplicates=[]
 for r in eligible:
  key=(r['title_sha256'],r['abstract_sha256'])
  if key in seen:duplicates.append({'id':r['id'],'canonical_id':seen[key],'reason':'exact_title_abstract_duplicate'})
  else:seen[key]=r['id'];records.append(r)
 records.sort(key=lambda r:(r['year'],r['source'],r['id']))
 (data/'records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in records))
 atomic(data/'join_ledger.json',ledger);atomic(data/'duplicate_exclusions.json',duplicates)
 profile=[]
 for year in YEARS:
  for source in SOURCES:
   frame=[r for r in ledger if r['source']==source and r['publisher_year']==year];valid=[r for r in frame if r['date_type_id_valid']];both=sum(r['has_title'] and r['has_abstract'] for r in valid)
   profile.append({'year':year,'source':source,'publisher_candidates':len(frame),'doi_matched_unique':sum(len(r['matched_pmids'])==1 for r in frame),'date_type_id_valid':len(valid),'valid_with_title':sum(r['has_title'] for r in valid),'valid_with_abstract':sum(r['has_abstract'] for r in valid),'valid_with_both':both,'joint_availability_valid_frame':both/len(valid) if valid else 0,'complete_coverage_publisher_frame':both/len(frame) if frame else 0,'eligible_final':sum(r['year']==year and r['source']==source for r in records),'text_distribution_valid_frame':{field:describe([r[field] for r in valid]) for field in ['title_chars','abstract_chars','title_tokens','abstract_tokens']}})
 focal=[r for r in records if r['year']==2021];history=[r for r in records if r['year']<2021]
 quality=all(sum(r['source']==s for r in focal)>=200 for s in SOURCES) and len(history)>=2000 and all(r['joint_availability_valid_frame']>=.95 and r['complete_coverage_publisher_frame']>=.95 for r in profile)
 view=lambda r:{k:r[k] for k in ['id','doi','pmid','year','source','topic','electronic_date','publisher_date','title_sha256','abstract_sha256']}
 manifest={'stage':'replication_metadata_audit','created_utc':now(),'quality_pass':quality,'records_sha256':sha(data/'records.jsonl'),'date_definition':'PubMed ArticleDate DateType=Electronic, unique valid Gregorian full date; year must agree with PLOS publication_date; no fallback','rules_sha256':sha(root/'docs/replication_cohort_rules_v1.md'),'profile':profile,'publisher_frame_count':len(candidates),'history_count':len(history),'focal_count':len(focal),'history':[view(r) for r in history],'focal':[view(r) for r in focal],'new_duplicate_exclusions':duplicates,'same_year_date_disagreements':date_disagreements,'exclusions':dict(Counter(reason for r in ledger for reason in r['exclusion_reasons'])),'retrieved_pubmed_records':len(pubmed),'text_short_records':[{'id':r['id'],'title_chars':len(r['title']),'abstract_chars':len(r['abstract'])} for r in records if len(r['abstract'])<100],'current_snapshot_not_historical_visibility':True}
 atomic(data/'cohort_manifest.json',manifest)
 atomic(data/'acquisition_manifest.json',{'stage':'replication_acquisition','created_utc':now(),'code_sha256':sha(__file__),'rules_sha256':manifest['rules_sha256'],'calls':client.calls,'data_sha256':{name:sha(data/name) for name in ['records.jsonl','publisher_frame.json','join_ledger.json','duplicate_exclusions.json','cohort_manifest.json','doi_rescue.json']},'policies':'NCBI ≤2req/sec; PLOS ≤10req/min,100rows/page,300/hour,7200/day; no credentials or email invented'})
 atomic(data/'progress.json',{'stage':'complete','quality_pass':quality,'updated_utc':now(),'focal':len(focal),'history':len(history)})
 print(json.dumps({'quality_pass':quality,'focal':len(focal),'history':len(history),'profile':profile,'exclusions':manifest['exclusions']},indent=2))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();run(args.root.resolve())
