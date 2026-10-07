"""Descriptive reference table; no pooled or cross-encoder inferential test."""
import csv,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
PROJECT=Path(__file__).resolve().parents[3]
BASE=Path(__file__).resolve().parents[3]/'external_work'
OUT=BASE/'encoder_sensitivity'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def label(metric,lo,hi,role):
 if role=='paired':return 'positive' if lo>0 else 'negative' if hi<0 else 'unresolved'
 t={'G':.1,'T':.05,'Q':.01}[metric]
 return 'below_tolerance' if hi<t else 'above_tolerance' if lo>t else 'unresolved'
new=rows(OUT/'primary_and_paired.csv');assert len(new)==12
original={};inputs={}
for study,folder,J in [('acl','confirmation_analysis',45),('biomedical','pubmed_replication_analysis',18)]:
 directory=PROJECT/'results'/folder;manifest=json.loads((directory/'analysis_manifest.json').read_text())
 for name in ['table2_aggregate.csv','table3_rank_selection.csv','table4_paired_primary.csv']:
  p=directory/name;assert sha(p)==manifest['output_sha256'][name];inputs[str(p.relative_to(PROJECT))]=sha(p)
  for r in rows(p):
   if float(r['rate'])!=.25 or r['metric'] not in ['G','T','Q']:continue
   role='paired' if name=='table4_paired_primary.csv' else 'base'
   mechanism=r['mechanism'] if role=='paired' else r['condition']
   if role=='base' and r['policy']!='mixed':continue
   if (study,mechanism,role) not in [('acl','random','base'),('biomedical','random','base'),('biomedical','source','base'),('biomedical','source','paired')]:continue
   original[(study,mechanism,role,r['metric'])]=(r,J)
assert len(original)==12
comparison=[]
for nr in new:
 old,J=original[(nr['study'],nr['mechanism'],nr['role'],nr['metric'])]
 for encoder,r,lo,hi,family in [('SPECTER1',old,'ci_low_raw','ci_high_raw',J),('SPECTER2',nr,'ci_low','ci_high',12)]:
  mean=float(r['mean']);low=float(r[lo]);high=float(r[hi]);k=156 if nr['study']=='acl' else 148
  comparison.append(dict(encoder=encoder,study=nr['study'],mechanism=nr['mechanism'],role=nr['role'],rate=.25,metric=nr['metric'],n=200,mean=mean,ci_low=low,ci_high=high,family_size=family,classification=label(nr['metric'],low,high,nr['role']),expected_lost_papers=mean*k if nr['metric']=='T' and nr['role']=='base' else ''))
with (OUT/'encoder_reference_comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(comparison[0]));w.writeheader();w.writerows(comparison)
receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'complete','scope':'Descriptive side-by-side original and new means; interval families differ; no cross-encoder or pooled inferential test.','code_sha256':sha(Path(__file__)),'original_inputs_sha256':inputs,'specter2_input_sha256':sha(OUT/'primary_and_paired.csv'),'output_sha256':sha(OUT/'encoder_reference_comparison.csv')}
(OUT/'encoder_reference_comparison_manifest.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps({'rows':len(comparison),'output':str(OUT/'encoder_reference_comparison.csv')}))
