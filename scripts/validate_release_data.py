"""Validate every released CSV against declared counts/types/missingness. No network."""
from pathlib import Path
import argparse,csv,hashlib,json,math,re

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--output',type=Path);a=ap.parse_args();root=a.root.resolve()
 schema=json.loads((root/'manifests/csv_schemas.json').read_text());checks=0;records=0;receipts=[]
 for rel,s in schema.items():
  p=root/rel
  with p.open(encoding='utf-8',newline='') as f:
   reader=csv.DictReader(f);assert reader.fieldnames==s['field_order'],rel+': column order';rr=list(reader)
  assert len(rr)==s['record_count'],rel+': record count';checks+=2;records+=len(rr)
  for col in s['columns']:
   k=col['name'];typ=col['type'];assert sum(x[k]=='' for x in rr)==col['missing_count'],rel+': missing count '+k;checks+=1
   for i,row in enumerate(rr):
    v=row[k]
    if v=='':assert col['nullable'],(rel,i,k);continue
    if typ=='integer':assert re.fullmatch(r'-?\d+',v),(rel,i,k,'integer expected')
    elif typ=='integer-or-ALL':assert v=='ALL' or re.fullmatch(r'\d+',v),(rel,i,k)
    elif typ=='boolean':assert v in ['True','False','true','false'],(rel,i,k)
    elif typ.startswith('float'):
     z=float(v);assert not math.isnan(z) and (math.isfinite(z) or col['allow_infinity']),(rel,i,k)
    elif typ=='json-string-array':
     identifier_list=json.loads(v);assert isinstance(identifier_list,list) and all(isinstance(t,str) and t for t in identifier_list) and len(set(identifier_list))==len(identifier_list),(rel,i,k)
    elif typ=='string':assert isinstance(v,str) and '\x00' not in v,(rel,i,k)
    else:raise ValueError('Unknown schema type '+typ)
    checks+=1
  if 'assignment_id' in s['field_order']:assert all(isinstance(x['assignment_id'],str) for x in rr);checks+=1
  if 'record_id' in s['field_order']:assert all(x['record_id'] and isinstance(x['record_id'],str) for x in rr);checks+=1
  receipts.append(dict(file=rel,rows=len(rr),columns=len(s['columns']),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 jsonl_schema=json.loads((root/'manifests/jsonl_schemas.json').read_text());jsonl_records=0
 for rel,decl in jsonl_schema.items():
  with (root/rel).open(encoding='utf-8') as handle:entries=[json.loads(line) for line in handle]
  assert len(entries)==decl['record_count'];checks+=1;jsonl_records+=len(entries)
  for row in entries:
   assert list(row)==decl['field_order'];checks+=1
   for key,typ in decl['fields'].items():
    value=row[key];assert value is not None,(rel,key)
    if typ=='string':assert isinstance(value,str) and value,(rel,key)
    elif typ=='integer':assert type(value)==int and value>=0,(rel,key)
    elif typ=='finite-number':assert type(value) in (int,float) and math.isfinite(value),(rel,key)
    elif typ=='unique-string-array':assert isinstance(value,list) and all(isinstance(z,str) and z for z in value) and len(set(value))==len(value),(rel,key)
    elif typ=='four-threshold-unique-string-arrays':
     assert set(value)=={'0.01','0.025','0.05','0.1'},(rel,key)
     for ids in value.values():assert isinstance(ids,list) and all(isinstance(z,str) and z for z in ids) and len(set(ids))==len(ids),(rel,key)
    else:raise ValueError('Unknown JSONL schema type'+typ)
    checks+=1
  receipts.append(dict(file=rel,rows=len(entries),columns=len(decl['fields']),sha256=hashlib.sha256((root/rel).read_bytes()).hexdigest()))
 actual={str(p.relative_to(root)) for p in root.rglob('*.csv') if 'outputs' not in p.relative_to(root).parts}
 assert actual==set(schema),(sorted(actual-set(schema)),sorted(set(schema)-actual))
 result=dict(status='PASS',csv_files=len(schema),csv_records=records,jsonl_files=len(jsonl_schema),jsonl_records=jsonl_records,assertions=checks,scope='UTF-8 CSV/JSONL schemas, observed missing values, exact row counts, finite numeric/native integer/boolean types and string/unique-list identifiers; not empirical rerun',files=receipts)
 if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='files'}))
if __name__=='__main__':main()
