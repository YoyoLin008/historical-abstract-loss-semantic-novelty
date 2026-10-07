"""Dated exact-DOI audit; protocol is separate and predates bulk acquisition."""
import argparse, csv, datetime, fcntl, hashlib, json, os, pathlib, time
import urllib.request, urllib.parse, urllib.error

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def stamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(p,obj):
    q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(obj,indent=2));os.replace(q,p)
def norm(v): return (v or '').lower().removeprefix('https://doi.org/').removeprefix('http://doi.org/').strip()
def state(x):
    if 'abstract_inverted_index' not in x: return 'field_omitted_unknown'
    a=x['abstract_inverted_index']
    if a is None:return 'explicit_null'
    if not isinstance(a,dict) or not a:return 'empty_or_malformed'
    positions=[]
    for token, ps in a.items():
        if not isinstance(token,str) or not token or not isinstance(ps,list) or not ps:return 'empty_or_malformed'
        if any(type(z) is not int or z<0 for z in ps):return 'empty_or_malformed'
        positions.extend(ps)
    if len(positions)!=len(set(positions)) or sorted(positions)!=list(range(max(positions)+1)):return 'empty_or_malformed'
    return 'valid_present'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',required=True);ap.add_argument('--output',required=True);v=ap.parse_args()
    root=pathlib.Path(v.project);out=pathlib.Path(v.output);out.mkdir(parents=True,exist_ok=True)
    lock=open(out/'job.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    frame=[];inputs={}
    for collection,p in [('ACL',root/'data/confirmation_v1/records.jsonl'),('Biomedical',root/'replication/data/records.jsonl')]:
        inputs[str(p.relative_to(root))]=digest(p)
        rr=[json.loads(l) for l in p.read_text().splitlines()]
        for r in rr:
            if 2018<=r['year']<=2020:frame.append({'collection':collection,'record_id':r['id'],'doi':norm(r['doi']),'source':r['source'],'year':r['year']})
    assert len(frame)==7190 and len({r['doi'] for r in frame})==7190 and all(r['doi'] for r in frame)
    frame.sort(key=lambda r:r['doi'])
    fingerprint={'inputs':inputs,'protocol_sha256':digest(out/'protocol.md'),'code_sha256':digest(pathlib.Path(__file__)),'dois_sha256':hashlib.sha256(json.dumps([r['doi'] for r in frame]).encode()).hexdigest()}
    job={'status':'running','pid':os.getpid(),'started_utc':stamp(),'fingerprint':fingerprint,'completed_batches':0}
    if (out/'manifest.json').exists():
        old=json.loads((out/'manifest.json').read_text());assert old['fingerprint']==fingerprint,'resume mismatch'
    atomic(out/'manifest.json',job);responses={};raw=out/'raw';raw.mkdir(exist_ok=True)
    for i in range(0,len(frame),100):
        batch=frame[i:i+100];idx=i//100;manifest=raw/f'{idx:03d}.manifest.json';body=raw/f'{idx:03d}.json'
        url='https://api.openalex.org/works?'+urllib.parse.urlencode({'filter':'doi:'+'|'.join('https://doi.org/'+r['doi'] for r in batch),'per_page':200,'select':'id,doi,publication_year,abstract_inverted_index'})
        if manifest.exists():
            m=json.loads(manifest.read_text());assert m['url']==url and m['sha256']==digest(body) and m['status']==200
            b=body.read_bytes()
        else:
            time.sleep(1)
            for attempt in range(3):
                try:
                    req=urllib.request.Request(url,headers={'User-Agent':'HistoricalAbstractCoverageResearch/major-revision-2026-10-06'})
                    with urllib.request.urlopen(req,timeout=45) as h:b=h.read();headers=dict(h.headers);status=h.status
                    break
                except urllib.error.HTTPError as e:
                    if e.code not in (429,500,502,503,504) or attempt==2:raise
                    time.sleep(min(30,5*(attempt+1)))
                except urllib.error.URLError:
                    if attempt==2:raise
                    time.sleep(5*(attempt+1))
            tmp=body.with_name(body.name+'.tmp');tmp.write_bytes(b);os.replace(tmp,body)
            m={'utc':stamp(),'url':url,'status':status,'headers':headers,'sha256':digest(body),'requested_dois':[r['doi'] for r in batch]};atomic(manifest,m)
        data=json.loads(b);assert isinstance(data.get('results'),list)
        assert data['meta']['count']==len(data['results']), 'response pagination or truncation'
        requested={r['doi'] for r in batch}
        for x in data['results']:
            d=norm(x.get('doi'))
            if d not in requested:continue
            responses.setdefault(d,[]).append((x,m['utc']))
        job['completed_batches']=idx+1;job['updated_utc']=stamp();atomic(out/'manifest.json',job)
        print(json.dumps({'batch':idx+1,'total_batches':72,'returned':len(data['results'])}),flush=True)
    rows=[]
    for r in frame:
        matches=responses.get(r['doi'],[]);s='unmatched' if not matches else 'ambiguous_duplicate' if len(matches)!=1 else state(matches[0][0])
        rows.append(dict(r,state=s,openalex_id=matches[0][0]['id'] if len(matches)==1 else '',provider_year=matches[0][0].get('publication_year') if len(matches)==1 else '',retrieved_utc=matches[0][1] if len(matches)==1 else ''))
    with (out/'field_states.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    coverage=[]
    for collection in ('ACL','Biomedical'):
        rr=[r for r in rows if r['collection']==collection];cells=[('ALL','ALL')]+[(s,y) for s in sorted({r['source'] for r in rr}) for y in (2018,2019,2020)]
        for source,year in cells:
            cc=[r for r in rr if (source=='ALL' or r['source']==source) and (year=='ALL' or r['year']==year)]
            states={s:sum(r['state']==s for r in cc) for s in ('valid_present','explicit_null','unmatched','empty_or_malformed','field_omitted_unknown','ambiguous_duplicate')}
            matched=len(cc)-states['unmatched']-states['ambiguous_duplicate'];known=states['valid_present']+states['explicit_null']
            coverage.append(dict(collection=collection,source=source,year=year,total=len(cc),**states,uniquely_matched=matched,known_field_states=known,null_percent_known=100*states['explicit_null']/known if known else None,full_frame_null_mask_gate=known==len(cc)))
    with (out/'coverage.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(coverage[0]));w.writeheader();w.writerows(coverage)
    job.update(status='complete',completed_utc=stamp(),outputs={p.name:digest(p) for p in (out/'field_states.csv',out/'coverage.csv')},collection_summaries=[r for r in coverage if r['source']=='ALL']);atomic(out/'manifest.json',job)
    print(json.dumps(job['collection_summaries']),flush=True)
if __name__=='__main__':main()
