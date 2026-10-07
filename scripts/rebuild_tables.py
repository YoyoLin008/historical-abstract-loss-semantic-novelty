"""Reconstruct reported views from released outcomes; no scoring/resampling/refit."""
from pathlib import Path
from collections import defaultdict
import argparse,csv,hashlib,json,math,time,platform,resource
import numpy as np

def rows(p):
 with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def output(p,rr):
 with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--out',type=Path);a=ap.parse_args();root=a.root.resolve();out=(a.out or root/'outputs/tables').resolve();out.mkdir(parents=True,exist_ok=True);start=time.monotonic();checks=[];primary=[];paired=[];cutoffs=[];cohorts=[]
 def check(v,label):
  if not v:raise ValueError(label)
  checks.append(label)
 assignment_summary=[];grid_views=[];allcutoffqualifiers=[];cohort_profile=[];nested=[]
 for study,k,n,family,config,critical in [('acl',156,1557,45,'confirmation_v1.json',3.3090595211511644),('biomedical',148,1471,18,'pubmed_replication_v1.json',3.030571298919818)]:
  directory=root/'data/derived'/('original_'+study);cfg=json.loads((root/'configs'/config).read_text());critical=cfg['inference']['t_critical'];groups=defaultdict(list)
  for row in rows(directory/'mask_primary_and_movement.csv'):groups[float(row['rate']),row['condition']].append(row)
  summaries=rows(directory/'table2_aggregate.csv')+rows(directory/'table3_rank_selection.csv');pairrows=rows(directory/'table4_paired_primary.csv');bc=json.loads((directory/'baseline_cutoff.json').read_text())
  check(bc['n']==n and bc['k']==k and bc['baseline_selected_in_band']==16,study+' baseline n,k,band selected')
  check(bc['baseline_selected_outside_band']==k-16,study+' outside conditional denominator')
  probabilities=rows(directory/'paper_selection_probabilities.csv');first_rate=probabilities[0]['rate'];first_cond=probabilities[0]['condition'];baseline=[x for x in probabilities if x['rate']==first_rate and x['condition']==first_cond and x['baseline_selected']=='True'];outside_ids={x['record_id'] for x in baseline if x['in_cutoff_band']=='False'};margin_ids={x['record_id'] for x in baseline if float(x['signed_margin_sd'])>.1}
  check(len(outside_ids)==k-16 and len(margin_ids)==(132 if study=='acl' else 133),study+' baseline nesting set sizes')
  check((margin_ids<outside_ids) if study=='acl' else (outside_ids<margin_ids),study+' verified cohort-specific δ.10 nesting')
  nested.append(dict(study=study,threshold_sd=.1,outside_count=len(outside_ids),margin_count=len(margin_ids),relationship='margin proper subset of outside' if study=='acl' else 'outside proper subset of margin'))
  # Every saved assignment row, including matched controls, respects the verified nesting.
  for rr_all in groups.values():
   for x in rr_all:
    outside=float(x['loss_outside_cutoff_band_per_k'])*k;far=int(x['far_loss_count_0.1']);check(far<=outside+1e-9 if study=='acl' else outside<=far+1e-9,study+' saved loss-count nesting '+x['assignment_id'])
  for cell,rr in sorted(groups.items()):
   rate,cond=cell
   if cond not in ['random','source','topic']:continue
   rr=sorted(rr,key=lambda x:x['assignment_id']);check(len(rr)==200,study+' '+str(cell)+' 200 original masks')
   for metric in ['G','T','Q']:
    target=[v for v in summaries if v['condition']==cond and float(v['rate'])==rate and v['policy']=='mixed' and v['metric']==metric];check(len(target)==1,study+' unique primary '+str((cell,metric)));v=target[0];xx=np.array([float(x[metric]) for x in rr]);mean=float(xx.mean());sd=float(xx.std(ddof=1));mcse=sd/math.sqrt(200);lo=mean-critical*mcse;hi=mean+critical*mcse
    for field,got in [('mean',mean),('sd_over_masks',sd),('mcse',mcse),('ci_low_raw',lo),('ci_high_raw',hi)]:check(abs(got-float(v[field]))<5e-12,study+' primary '+str((cell,metric,field)))
    check(v['interval']==f'approximate_95pct_FWER_{family}_Bonferroni_Student_mean',study+' original family retained')
    primary.append(dict(study=study,condition=cond,rate=rate,metric=metric,n_masks=200,mean=mean,mcse=mcse,ci_low=lo,ci_high=hi,family_size=family,mean_sd_over_masks=sd))
   for x in rr:
    check(int(x['selected_lost_count'])==int(x['selected_gained_count']),study+' fixed set loss/gain '+x['assignment_id'])
    check(abs(float(x['T'])*k-int(x['selected_lost_count']))<1e-9,study+' T*k '+x['assignment_id'])
    check(abs((float(x['loss_in_cutoff_band_per_k'])+float(x['loss_outside_cutoff_band_per_k']))*k-int(x['selected_lost_count']))<1e-9,study+' inside+outside '+x['assignment_id'])
    cc=[int(x['far_loss_count_'+str(d)]) for d in [.01,.025,.05,.1]];check(all(b<=a for a,b in zip(cc,cc[1:])),study+' margin counts monotone '+x['assignment_id'])
   inside=float(np.mean([float(x['loss_in_cutoff_band_per_k'])*k for x in rr]));outside=float(np.mean([float(x['loss_outside_cutoff_band_per_k'])*k for x in rr]));total=float(np.mean([int(x['selected_lost_count']) for x in rr]));far=float(np.mean([int(x['far_loss_count_0.1']) for x in rr]));cutoffs.append(dict(study=study,condition=cond,rate=rate,k=k,inside_count=inside,outside_count=outside,total_count=total,far_0_1_sd_count=far))
   qualifier=dict(study=study,condition=cond,rate=rate,total_count=total,outside_count=outside,conditional_outside_fraction=outside/(k-16))
   for delta in [.01,.025,.05,.1]:qualifier['far_loss_count_'+str(delta)]=float(np.mean([int(x['far_loss_count_'+str(delta)]) for x in rr]))
   allcutoffqualifiers.append(qualifier)
   cells=[v for v in rows(directory/'control_cell_counts.csv') if float(v['rate'])==rate and v['mechanism']==cond]
   assignment_summary.append(dict(study=study,condition=cond,rate=rate,missing_count=int(math.floor((3799 if study=='acl' else 3391)*rate+.5)),base_assignments=200,control_assignments=len(cells),matching='none' if cond=='random' else 'year;source' if cond=='topic' else 'year',zero_cell_instances=sum(int(x['zero_cells']) for x in cells),full_cell_instances=sum(int(x['full_cells']) for x in cells)))
   if cond=='random':continue
   controls={x['assignment_id']:x for x in groups[rate,cond+'_matched_control']};check(len(controls)==200 and set(controls)=={x['assignment_id'] for x in rr},study+' exact matched IDs')
   for metric in ['G','T','Q']:
    target=[v for v in pairrows if v['mechanism']==cond and float(v['rate'])==rate and v['metric']==metric];check(len(target)==1,study+' unique paired '+str((cell,metric)));v=target[0];xx=np.array([float(x[metric])-float(controls[x['assignment_id']][metric]) for x in rr]);mean=float(xx.mean());mcse=float(xx.std(ddof=1)/math.sqrt(200));lo=mean-critical*mcse;hi=mean+critical*mcse
    for field,got in [('mean',mean),('mcse',mcse),('ci_low_raw',lo),('ci_high_raw',hi)]:check(abs(got-float(v[field]))<5e-12,study+' paired '+str((cell,metric,field)))
    paired.append(dict(study=study,mechanism=cond,rate=rate,metric=metric,n_masks=200,mean=mean,mcse=mcse,ci_low=lo,ci_high=hi,family_size=family))
  for x in rows(directory/'table1_cohort_construction.csv'):cohorts.append(dict(study=study,cohort=x['cohort'],year=x['year'],source=x['source'],n=x['n'],baseline_top_decile_k=x['baseline_top_decile_k']))
  identifiers=rows(root/'data/identifiers'/f'{study}_cohort_identifiers.csv')
  check(len(identifiers)==n+(3799 if study=='acl' else 3391),study+' exact all cohort IDs')
  for x in rows(directory/'table1_cohort_construction.csv'):check(sum(r['year']==x['year'] and r['source']==x['source'] and r['cohort']==x['cohort'] for r in identifiers)==int(x['n']),study+' identifier cell '+str((x['year'],x['source'],x['cohort'])))
  for grid in rows(directory/'threshold_sensitivity.csv'):
   rate=float(grid['rate']);cond=grid['condition'];qq={v['metric']:v for v in primary if v['study']==study and v['rate']==rate and v['condition']==cond};check(set(qq)=={'G','T','Q'},study+' threshold inputs')
   classes={}
   for metric,label in [('G','aggregate'),('T','selection'),('Q','ranks')]:
    tau=float(grid['tau_'+metric]);v=qq[metric];classification='below_tolerance' if v['ci_high']<tau else 'above_tolerance' if v['ci_low']>tau else 'unresolved';check(classification==grid[label],study+' unchanged grid '+label);classes[metric]=classification
   joint=classes['G']=='below_tolerance' and classes['T']=='above_tolerance';check(str(joint)==grid['joint_mean_rule'] and str(joint and classes['Q']=='above_tolerance')==grid['joint_with_rank_rule'],study+' unchanged grid decisions');grid_views.append(dict(study=study,**grid))
  if study=='biomedical':
   cp=json.loads((directory/'cohort_profile.json').read_text());check(cp['publisher_frame_count']==4896 and sum(int(x['publisher_candidates']) for x in cp['profile'])==4896,'biomedical actual publisher frame');check(sum(int(x['eligible_final']) for x in cp['profile'])==4862,'biomedical retained original frame');cohort_profile=cp['profile']
 output(out/'original_primary_reconstructed.csv',primary);output(out/'original_paired_reconstructed.csv',paired);output(out/'cutoff_counts_reconstructed.csv',cutoffs);output(out/'collection_counts.csv',cohorts)
 output(out/'table_S2_assignment_counts.csv',assignment_summary);output(out/'table_S5_all_margin_counts.csv',allcutoffqualifiers);output(out/'table_S6_threshold_grid.csv',grid_views);output(out/'table_S1_publisher_frame.csv',cohort_profile);output(out/'verified_cutoff_set_nesting.csv',nested)
 check(len(primary)==39 and len(paired)==24 and len(cutoffs)==13,'all original reported means/conditions')
 # Existing posthoc summaries are faithfully rendered, never recomputed from new outcomes.
 for filename in ['baseline_reference_comparison.csv','encoder_reference_comparison.csv','primary_and_paired.csv']:output(out/('specter2_'+filename),rows(root/'data/derived/post_confirmation/specter2'/filename))
 states=rows(root/'data/derived/post_confirmation/openalex/field_states.csv');coverage=rows(root/'data/derived/post_confirmation/openalex/coverage.csv');check(len(states)==7190 and len({x['doi'] for x in states})==7190,'7190 historical uniqueDOIs')
 for c in coverage:
  rr=[r for r in states if r['collection']==c['collection'] and (c['source']=='ALL' or r['source']==c['source']) and (c['year']=='ALL' or r['year']==c['year'])];check(len(rr)==int(c['total']),'provider frame cell')
  for state in ['valid_present','explicit_null','unmatched','empty_or_malformed','field_omitted_unknown','ambiguous_duplicate']:check(sum(r['state']==state for r in rr)==int(c[state]),'provider field '+state)
 check(sum(x['state']=='valid_present' for x in states)==7176 and sum(x['state']=='ambiguous_duplicate' for x in states)==14,'OpenAlex failed calibration state totals');output(out/'openalex_coverage.csv',coverage)
 for fn in ['interval_sensitivity.csv','joint_sensitivity.csv']:output(out/('bootstrap_'+fn),rows(root/'data/derived/post_confirmation/bootstrap'/fn))
 pilot=json.loads((root/'data/derived/pilot/diagnostic.json').read_text());diagnostic=[]
 for model,v in pilot['models'].items():
  target=np.array(v['heldout_targets']);pred=np.array(v['heldout_predictions']);check(len(target)==60 and len(pred)==60,'pilot fixed60 heldout');mae=float(abs(target-pred).mean());rmse=float(np.sqrt(np.mean((target-pred)**2)));check(abs(mae-v['test_mae_sd_units'])<1e-12 and abs(rmse-v['test_rmse_sd_units'])<1e-12,'pilot preserved prediction errors');diagnostic.append(dict(model=model,n_train=v['n_train_masks'],n_test=60,mae_sd=mae,rmse_sd=rmse))
 output(out/'pilot_heldout_errors.csv',diagnostic)
 result=dict(status='PASS',scope='Reporting-level derived-data reconstruction only; no scoring/encoding/resampling/refit',original_primary_means=39,original_paired_means=24,original_conditions=13,original_controls='assignment-ID matched200 each',assertions=len(checks),elapsed_seconds=time.monotonic()-start,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,peak_rss_unit='bytes on macOS;KiB on Linux',python=platform.python_version(),numpy=np.__version__,files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.csv')},checks=checks)
 (out/'reporting_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['files','checks']}))
if __name__=='__main__':main()
