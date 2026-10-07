"""Reporting-level cutoff proof using released numeric/identifier evidence.

This verifier reconstructs baseline selected sets, rank bands and score margins
from released baseline scores, then verifies the reported intersection counts
against released saved lost-paper IDs. It does NOT regenerate masked scores,
encode abstracts, reacquire text, draw masks or reproduce a raw-to-model study.

Usage: python verify_public_cutoff_membership.py --data DIRECTORY
DIRECTORY contains baseline_membership.csv, cutoff_set_check.csv and the three
*_lost_ids.jsonl files. NumPy is the only non-standard dependency.
"""
from pathlib import Path
from collections import defaultdict
import argparse, csv, hashlib, json, math
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);a=p.parse_args();d=a.data
groups=defaultdict(list);checks=0;maskcount=0
def verify(ok,msg):
 global checks
 checks+=1
 if not ok:raise AssertionError(msg)
for r in csv.DictReader((d/'baseline_membership.csv').open()):groups[(r['study'],r['encoder'])].append(r)
contexts={}
for key,rows in groups.items():
 ids=np.array([r['paper_id'] for r in rows]);score=np.array([float(r['score']) for r in rows]);n=len(score);k=math.ceil(.1*n)
 order=np.lexsort((ids,-score));selected=set(ids[order[:k]]);cut=(score[order[k-1]]+score[order[k]])/2;sd=score.std(ddof=0)
 percentiles=(np.arange(1,n+1)-.5)/n;ix=np.flatnonzero((percentiles>=.09)&(percentiles<=.11))
 band=set(ids[(score<=score[order[ix[0]]])&(score>=score[order[ix[-1]]])]);outside=selected-band
 margins=(score-cut)/sd;far={v:{ident for ident,m in zip(ids,margins) if ident in selected and m>v} for v in [.01,.025,.05,.1]}
 verify(outside==set(ids[order[:len(outside)]]),'outside-band selected set is an upper prefix')
 for v,c in far.items():
  verify(c==set(ids[order[:len(c)]]),'positive-margin selected set is an upper prefix')
  verify(not(outside-c and c-outside),'baseline nested sets')
 for i,r in enumerate(rows):
  verify((r['selected']=='True')==(ids[i] in selected),'selected membership')
  verify((r['in_band']=='True')==(ids[i] in band),'rank-band membership')
  verify((r['outside_selected']=='True')==(ids[i] in outside),'outside-band membership')
  verify(abs(float(r['margin_sd'])-margins[i])<2e-12,'baseline standardized margin')
  for v in far:verify((r['C_'+str(v)]=='True')==(ids[i] in far[v]),'score-margin membership')
 contexts[key]=(selected,outside,far)
expected={}
for r in csv.DictReader((d/'cutoff_set_check.csv').open()):
 key=(r['study'],r['encoder'],r['assignment_id'],r['role'],r['delta'])
 verify(key not in expected,'unique composite cutoff row identity');expected[key]=r
observed=set()
for name in ['acl_lost_ids.jsonl','biomedical_lost_ids.jsonl','specter2_lost_ids.jsonl']:
 for r in map(json.loads,(d/name).open()):
  maskcount+=1;selected,outside,far=contexts[(r['study'],r['encoder'])];lost=set(r['lost_ids']);olost=lost&outside
  verify(lost<=selected and len(lost)==len(r['lost_ids']),'unique original-selected lost identities')
  verify(olost==set(r['outside_ids']),'outside lost identities')
  verify(len(lost)==r['selected_lost_count'],'total lost count')
  k=len(selected);verify(abs(r['T']-len(lost)/k)<2e-12,'T denominator')
  verify(len(olost)==r['outside_count'] and len(lost)-len(olost)==r['inside_count'],'inside/outside partition')
  verify(abs(r['loss_outside_cutoff_band_per_k']-len(olost)/k)<2e-12 and abs(r['loss_in_cutoff_band_per_k']-(len(lost)-len(olost))/k)<2e-12,'rank-band fractions')
  verify(abs(r['loss_conditional_selected_outside_band']-len(olost)/len(outside))<2e-12,'conditional outside denominator')
  for v,c in far.items():
   clost=lost&c;verify(clost==set(r['far_ids'][str(v)]),'margin lost identities');verify(len(clost)==r['far_loss_count_'+str(v)],'saved far count')
   key=(r['study'],r['encoder'],r['assignment_id'],r['role'],str(v));z=expected[key];observed.add(key)
   left=olost-clost;right=clost-olost
   verify(not(left and right),'lost intersections nested')
   for col,val in [('O_count',len(olost)),('C_count',len(clost)),('O_minus_C',len(left)),('C_minus_O',len(right))]:verify(int(z[col])==val,'saved cutoff set difference count')
   verify(set(json.loads(z['O_minus_C_ids']))==left and set(json.loads(z['C_minus_O_ids']))==right,'saved cutoff difference identities')
for (study,encoder),(selected,outside,far) in contexts.items():
 for v,c in far.items():
  z=expected[(study,encoder,'','',str(v))];left=outside-c;right=c-outside
  for col,val in [('O_count',len(outside)),('C_count',len(c)),('O_minus_C',len(left)),('C_minus_O',len(right))]:verify(int(z[col])==val,'baseline actual set difference count')
  verify(set(json.loads(z['O_minus_C_ids']))==left and set(json.loads(z['C_minus_O_ids']))==right,'baseline actual difference identities')
verify(len(observed)==20000 and maskcount==5000,'all original/limited-SPECTER2 saved masks checked')
verify({k for k,r in expected.items() if r['level']=='mask'}==observed,'complete cutoff row coverage')
print(json.dumps({'status':'pass','level':'reporting-level numeric/ID reconstruction only','checks':checks,'saved_lost_id_sets':maskcount,'mask_threshold_comparisons':len(observed),'baseline_contexts':len(contexts)}))
