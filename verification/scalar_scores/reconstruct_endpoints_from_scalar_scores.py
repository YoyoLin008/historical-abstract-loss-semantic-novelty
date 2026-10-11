"""Independent scalar-only reconstruction. No original endpoint implementation imported.

Run: python reconstruct_endpoints_from_scalar_scores.py --package . --output verification
Dependencies: Python 3.12, NumPy, mpmath. No text, embedding, model or network input.
"""
import argparse, collections, csv, hashlib, json, math, pathlib, platform, resource, sys, time
import numpy as np

EDGES=np.array([-np.inf,-.1,-.05,-.025,-.01,0,.01,.025,.05,.1,np.inf])

def require(ok,message):
 if not ok:raise ValueError(message)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def csv_read(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
def csv_write(p,rs):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,list(rs[0]));w.writeheader();w.writerows(rs)
def json_write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def ranks(values):
 """Independent tie grouping: each distinct scalar gets its midrank."""
 distinct,inverse,counts=np.unique(values,return_inverse=True,return_counts=True)
 stop=np.cumsum(counts);mid=stop-(counts+1)/2
 return mid[inverse]
def ordering(values,ids):return np.array(sorted(range(len(ids)),key=lambda i:(-float(values[i]),ids[i])),dtype=int)
def context(b,ids,sources,source_order):
 b=np.asarray(b,dtype=np.float64);ids=list(ids);src=np.asarray(sources)
 require(b.ndim==1 and len(b)>1 and len(ids)==len(b)==len(src),'Baseline dimensions')
 require(len(ids)==len(set(ids)) and all(ids),'Duplicate/empty IDs');require(np.isfinite(b).all(),'Nonfinite baseline')
 a=src==source_order[0];bb=src==source_order[1];require(a.any() and bb.any() and np.all(a|bb),'Unexpected source labels');sd=float(np.std(b,ddof=0));require(sd>0,'Constant baseline')
 order=ordering(b,ids);n=len(b);k=math.ceil(n/10);selected=np.zeros(n,bool);selected[order[:k]]=True
 positions=[j for j in range(n) if .09<=(j+.5)/n<=.11]
 band=(b<=b[order[positions[0]]])&(b>=b[order[positions[-1]]]) if positions else np.zeros(n,bool)
 cutoff=float((b[order[k-1]]+b[order[k]])/2);margin=(b-cutoff)/sd
 return dict(b=b,ids=ids,sources=src,A=a,B=bb,sd=sd,contrast=float(b[a].mean()-b[bb].mean()),k=k,old=selected,rank=ranks(b),band=band,margin=margin,bins=np.digitize(margin,EDGES,right=False)-1,cutoff=cutoff,kth=float(b[order[k-1]]),next=float(b[order[k]]),baseline_rank_tied_papers=int(sum(c for c in np.unique(b,return_counts=True)[1] if c>1)))
def endpoint(z,c):
 z=np.asarray(z,dtype=np.float64);require(z.shape==c['b'].shape and np.isfinite(z).all(),'Incomplete/nonfinite scores')
 selection=np.zeros(len(z),bool);selection[ordering(z,c['ids'])[:c['k']]]=True;lost=c['old']&~selection;gained=selection&~c['old'];dz=z-c['b'];A=float(dz[c['A']].mean()/c['sd']);B=float(dz[c['B']].mean()/c['sd']);D=float((z[c['A']].mean()-z[c['B']].mean()-c['contrast'])/c['sd']);M=float(dz.mean()/c['sd']);w=float(c['A'].mean())
 require(abs(D-(A-B))<1e-12 and abs(M-(w*A+(1-w)*B))<1e-12,'Source shift identity')
 rank=ranks(z);q=float(np.mean(abs(rank-c['rank']))/(len(z)-1));L=int(lost.sum());I=int((lost&c['band']).sum());O=L-I;capacity=int((c['old']&c['band']).sum());require(L==int(gained.sum()) and I<=capacity and O>=max(0,L-capacity),'Selection/partition/capacity invariant')
 v=dict(G=abs(D),T=L/c['k'],Q=q,signed_change_sd=D,masked_contrast_sd=D+c['contrast']/c['sd'],baseline_contrast_sd=c['contrast']/c['sd'],mean_drift_sd=M,source_A_shift_sd=A,source_B_shift_sd=B,selected_lost_count=L,inside_count=I,outside_count=O,rank_tied_papers=int(sum(count for count in np.unique(z,return_counts=True)[1] if count>1)))
 return v,lost,selection

def summarize(xs,critical):
 x=np.asarray(xs,float);require(len(x)==200 and np.isfinite(x).all(),'Expected 200 inferential units');sd=float(np.std(x,ddof=1));se=sd/math.sqrt(len(x));m=float(x.mean())
 return dict(n=len(x),mean=m,sd=sd,mcse=se,ci_low=m-critical*se,ci_high=m+critical*se,p05=float(np.quantile(x,.05,method='linear')),p95=float(np.quantile(x,.95,method='linear')),median=float(np.median(x)))
def verify_critical(family,critical):
 import mpmath as mp
 with mp.workdps(45):
  df=mp.mpf(199);target=mp.mpf('.05')/(2*family);lo=mp.mpf(0);hi=mp.mpf(10)
  for _ in range(100):
   x=(lo+hi)/2;tail=mp.betainc(df/2,mp.mpf('.5'),0,df/(df+x*x),regularized=True)/2
   if tail>target:lo=x
   else:hi=x
  expected=float((lo+hi)/2);require(abs(expected-critical)<1e-10,'Recorded family critical value differs')
 return expected

def run(package,output):
 started=time.monotonic();output.mkdir(parents=True,exist_ok=True);manifest=json.loads((package/'manifest.json').read_text());receipt=json.loads((package/'export_receipt.json').read_text());checks=collections.Counter();maxerr=collections.defaultdict(float);comparisons=[];source_means=[];all_arms=[];summaries=[];baseline_outputs=[];failures=[];critical_checks={}
 def check(ok,label):checks[label]+=1;require(ok,label)
 def close(actual,expected,tolerance,label,key):
  error=abs(float(actual)-float(expected));maxerr[label]=max(maxerr[label],error);check(error<=tolerance,label);comparisons.append(dict(key=key,quantity=label,actual=float(actual),expected=float(expected),absolute_error=error,tolerance=tolerance))
 for rel,digest in receipt['output_checksums'].items():check(sha(package/rel)==digest,'export_payload_hash')
 refs=csv_read(package/'reference_summaries.csv');check(len(refs)==75,'family_reference_rows')
 for ds in manifest['datasets']:
  name=ds['key'];dp=package/name;f=csv_read(dp/'focal.csv');h=csv_read(dp/'history.csv');arms=csv_read(dp/'arms.csv');expected=[json.loads(l) for l in (dp/'reference_arms.jsonl').read_text().splitlines()];bm=csv_read(dp/'reference_membership.csv');refbase=json.loads((dp/'reference_baseline.json').read_text());ids=[r['focal_id'] for r in f];src=[r['source'] for r in f]
  with np.load(dp/'scores.npz',allow_pickle=False) as data:
   check(set(data.files)=={'baseline','scores','missing_history'},'npz_schema');check(all(not data[k].dtype.hasobject for k in data.files),'non_object_arrays');b=data['baseline'];zs=data['scores'];masks=data['missing_history'];check(b.dtype==np.float64 and zs.dtype==np.float64,'no_score_downcast')
  check(zs.shape==(ds['n_arms'],ds['n_focal']) and b.shape==(ds['n_focal'],),'expected_scalar_shape');check(masks.shape==(ds['n_arms'],ds['n_history']) and masks.dtype==np.bool_,'mask_shape');check(len(h)==ds['n_history'] and len({r['history_id'] for r in h})==len(h),'history_identity');check(all(int(r['year'])<2021 for r in h),'strict_prior_history');check(all(int(r['year'])==2021 for r in f),'fixed_focal_year');check(not set(ids)&{r['history_id'] for r in h},'no_focal_history_overlap');check([int(r['position']) for r in f]==list(range(len(f))),'ordered_focal_positions');check([int(r['position']) for r in h]==list(range(len(h))),'ordered_history_positions');check([src.count(s) for s in ds['sources']]==ds['source_counts'],'source_counts')
  check(len(arms)==len(expected)==ds['n_arms'],'expected_arm_count');check([int(r['row']) for r in arms]==list(range(len(arms))),'ordered_arm_rows');check(len({(r['assignment_id'],r['role']) for r in arms})==len(arms),'unique_arm_keys');check(np.isfinite(zs).all(),'all_masked_scores_finite');c=context(b,ids,src,ds['sources']);check({r['paper_id'] for r in bm}==set(ids) and len(bm)==len(ids),'reference_exact_cohort_IDs');lookup={x:i for i,x in enumerate(ids)}
  for r in bm:
   j=lookup[r['paper_id']];check(float(r['score'])==b[j],'exact_baseline_score');check((r['selected']=='True')==bool(c['old'][j]),'exact_baseline_selected_ID');check((r['in_band']=='True')==bool(c['band'][j]),'exact_baseline_band_ID');close(c['margin'][j],r['margin_sd'],2e-10,'baseline_margin',name+'/'+r['paper_id'])
  if ds['encoder']=='SPECTER2':
   close(c['sd'],refbase['sd'],2e-10,'baseline_sd',name);close(c['contrast'],refbase['source_contrast_raw'],2e-10,'baseline_contrast',name);check(set(refbase['selected_ids'])==set(np.array(ids)[c['old']]),'S2_saved_baseline_selected_IDs')
  else:
   close(c['sd'],refbase['baseline_population_sd'],2e-10,'baseline_sd',name);close(c['contrast'],refbase['baseline_group_contrast'],2e-10,'baseline_contrast',name)
  if ds['family_size'] not in critical_checks:critical_checks[ds['family_size']]=dict(recorded=ds['critical'],independently_verified=verify_critical(ds['family_size'],ds['critical']))
  check(c['k']==(156 if ds['study']=='ACL' else 148),'baseline_k');baseline_outputs.append(dict(dataset=name,sd=c['sd'],contrast=c['contrast'],contrast_sd=c['contrast']/c['sd'],k=c['k'],n=len(ids),band_size=int(c['band'].sum()),selected_in_band=int((c['old']&c['band']).sum()),cutoff_exact_tie=c['kth']==c['next'],rank_tied_papers=c['baseline_rank_tied_papers']))
  grouped=collections.defaultdict(list);values={};arm_index={};bincount=np.bincount(c['bins'][c['old']],minlength=10)
  for j,(arm,ref,z) in enumerate(zip(arms,expected,zs)):
   armkey=arm['assignment_id'],arm['role'];key=name+'/'+arm['assignment_id']+'/'+arm['role'];check((ref['assignment_id'],ref['role'],ref['mechanism'],float(ref['rate']))==(arm['assignment_id'],arm['role'],arm['mechanism'],float(arm['rate'])),'exact_arm_reference_ID_role_condition');check(int(masks[j].sum())==int(arm['missing_count'])==math.floor(float(arm['rate'])*len(h)+.5),'exact_mask_count');check(arm['pair_id']==(arm['assignment_id'] if arm['mechanism']!='random' else ''),'exact_pair_identifier')
   v,lost,selected=endpoint(z,c);check(set(np.array(ids)[lost])==set(ref['lost_ids']) and len(ref['lost_ids'])==len(set(ref['lost_ids'])),'exact_lost_IDs');check(set(np.array(ids)[lost&~c['band']])==set(ref['outside_ids']),'exact_outside_IDs');check(v['selected_lost_count']==ref['selected_lost_count'] and v['inside_count']==ref['inside_count'] and v['outside_count']==ref['outside_count'],'exact_cutoff_counts')
   for threshold,expectedids in ref['far_ids'].items():check(set(np.array(ids)[lost&(c['margin']>float(threshold))])==set(expectedids),'exact_strict_margin_IDs')
   for m in ['G','T','Q']:
    close(v[m],ref[m],2e-10,'readout_arm_'+m,key);close(v[m],ref['saved_scoring_endpoints'][m],manifest['documented_reconciliation_tolerances'][m],'saved_scoring_arm_'+m,key)
   numerator=np.bincount(c['bins'][lost],minlength=10);loss=np.divide(numerator,bincount,out=np.full(10,np.nan),where=bincount>0);check(abs(float(np.nansum(loss*bincount))-v['selected_lost_count'])<1e-10,'bin_weighted_loss_identity');check(np.isnan(loss[bincount==0]).all(),'empty_bins_undefined');values[armkey]=v;arm_index[armkey]=j;grouped[(arm['mechanism'],arm['role'],float(arm['rate']))].append(v);all_arms.append(dict(dataset=name,assignment_id=arm['assignment_id'],role=arm['role'],mechanism=arm['mechanism'],rate=float(arm['rate']),**v))
  # Counter audit forbids missing cells, extra deterministic policy arms, or new rates.
  rates=[.1,.25,.5] if name=='specter_acl' else ([.25,.5] if name=='specter_biomedical' else [.25]);mechs=['random','source','topic'] if name=='specter_acl' else (['random'] if name=='specter2_acl' else ['random','source']);expectedcells={(m,r,p) for m in mechs for p in rates for r in (['base'] if m=='random' else ['base','control'])};check(set(grouped)==expectedcells,'exact_condition_scope');check(all(len(v)==200 for v in grouped.values()),'200_arms_each_condition')
  paired=collections.defaultdict(list)
  for arm in arms:
   if arm['role']!='control':continue
   k=arm['assignment_id'];check((k,'base') in values,'exact_control_base_pair');base=values[k,'base'];control=values[k,'control'];paired[(arm['mechanism'],'paired',float(arm['rate']))].append({m:base[m]-control[m] for m in ['G','T','Q']})
  for (mech,role,rate),vs in grouped.items():
   source_means.append(dict(dataset=name,study=ds['study'],condition=mech if role=='base' else mech+'_matched_control',role=role,rate=rate,n_masks=len(vs),**{k:float(np.mean([v[k] for v in vs])) for k in ['baseline_contrast_sd','masked_contrast_sd','signed_change_sd','mean_drift_sd','source_A_shift_sd','source_B_shift_sd']},mean_G=float(np.mean([v['G'] for v in vs]))))
  for ref in refs:
   if ref['dataset']!=name:continue
   group=(ref['mechanism'],ref['role'],float(ref['rate']));v=(paired if group[1]=='paired' else grouped)[group];metric=ref['metric'];s=summarize([r[metric] for r in v],ds['critical']);key='/'.join(map(str,(name,*group,metric)))
   check(len(v)==int(ref['n']),'family_unit_count');check(ds['family_size'] in [45,18,12],'separate_family_identity')
   for quantity in ['mean','sd','mcse','ci_low','ci_high','p05','p95','median']:close(s[quantity],ref[quantity],2e-10,'family_'+quantity,key)
   summaries.append(dict(dataset=name,mechanism=group[0],role=group[1],rate=group[2],metric=metric,family=ds['family_size'],critical=ds['critical'],**s))
 check(sum(ds['n_arms'] for ds in manifest['datasets'])==5000,'total_5000_scored_arms');check(sum(ds['masked_scalar_values'] for ds in manifest['datasets'])==7630200,'total_7630200_masked_values');check(len(summaries)==75,'63_original_plus_12_S2_means')
 for n,x in [('baseline_reconstruction.csv',baseline_outputs),('per_arm_endpoints.csv',all_arms),('source_shift_direct_means.csv',source_means),('family_reconstruction.csv',summaries),('numeric_comparisons.csv',comparisons)]:csv_write(output/n,x)
 report=dict(status='PASS',scope='Independent endpoint reconstruction from saved complete scalars; no fresh scoring/encoding/acquisition',families={'original_ACL':45,'original_Biomedical':18,'SPECTER2':12},scored_arms=5000,masked_scalar_values=7630200,summary_rows=75,checks=dict(checks),check_assertions=sum(checks.values()),maximum_absolute_deviations=dict(maxerr),critical_checks=critical_checks,failures=failures,command=sys.argv,elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,environment={'python':platform.python_version(),'numpy':np.__version__,'mpmath':__import__('mpmath').__version__,'platform':sys.platform},input_sha256={str(p.relative_to(package)):sha(p) for p in package.rglob('*') if p.is_file() and output not in p.parents and '__pycache__' not in str(p)},output_sha256={str(p.relative_to(output)):sha(p) for p in output.iterdir() if p.is_file() and p.name!='verification_report.json'})
 json_write(output/'verification_report.json',report);print(json.dumps({k:report[k] for k in ['status','scored_arms','masked_scalar_values','summary_rows','check_assertions','elapsed_seconds']}));return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--package',type=pathlib.Path,default=pathlib.Path('.'));p.add_argument('--output',type=pathlib.Path,default=pathlib.Path('verification'));a=p.parse_args()
 try:run(a.package.resolve(),a.output.resolve())
 except Exception as exc:
  a.output.mkdir(parents=True,exist_ok=True);json_write(a.output/'failed_verification.json',dict(status='FAILED_CHECK',error=repr(exc),command=sys.argv));raise
if __name__=='__main__':main()
