"""Generate pilot summaries and a prespecified, whole-mechanism diagnostic holdout."""
from pathlib import Path
import csv,hashlib,json,os
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'results/pilot';summary=json.loads((p/'summary.json').read_text());baseline=json.loads((p/'baseline.json').read_text());raw=[json.loads(x) for x in (p/'mask_results.jsonl').read_text().splitlines()]
 manifest=json.loads((p/'run_manifest.json').read_text());assert manifest['status']=='complete','Pilot not complete'
 for name,expected in manifest['output_sha256'].items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==expected, 'Pilot output checksum mismatch: '+name
 assert hashlib.sha256((ROOT/'data/processed/records.jsonl').read_bytes()).hexdigest()==manifest['provenance']['inputs']['records'],'Records changed since pilot'
 assert baseline['run_fingerprint']==summary['run_fingerprint']==manifest['provenance']['run_fingerprint']
 assert all(r['run_fingerprint']==manifest['provenance']['run_fingerprint'] for r in raw)
 rows=[]
 for r in summary['conditions']:
  if r['condition'] in ['random','source','topic']:
   m=r['metrics'];key='standardized_mean_absolute_group_distortion';q=m[key];rows.append({'mask':r['condition'],'rate':r['rate'],'policy':r['policy'],'mean_absolute_group_distortion_sd':q['mean'],'mask_sd':q['sd_over_masks'],'monte_carlo_se':q['monte_carlo_se_mean'],'top_decile_replacement':m['mean_baseline_top_decile_replacement_fraction']['mean'],'n_masks':q['n_masks']})
 out=ROOT/'results/pilot_analysis';out.mkdir(parents=True,exist_ok=True)
 with (out/'conditions.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 records={r['id']:r for r in map(json.loads,(ROOT/'data/processed/records.jsonl').read_text().splitlines())};sources=np.array([records[i]['source'] for i in baseline['scope']['focal_ids']]);A=sources=='ACL';B=sources=='EMNLP'
 features=[];targets=[];mechanisms=[]
 for r in raw:
  a=r['structured_or_random'];local=np.array(a['deployment_available_features']['title_neighborhood_missing_fraction']);comp=a['composition'];sourcegap=comp['source']['ACL']['missing']/comp['source']['ACL']['total']-comp['source']['EMNLP']['missing']/comp['source']['EMNLP']['total'];yeargap=comp['year']['2018']['missing']/comp['year']['2018']['total']-comp['year']['2019']['missing']/comp['year']['2019']['total']
  features.append([a['realized_global_missing_rate'],abs(sourcegap),abs(yeargap),abs(local[A].mean()-local[B].mean()),local.mean()]);targets.append(a['evaluations']['mixed']['outcomes']['standardized_mean_absolute_group_distortion']);mechanisms.append(r['mechanism'])
 X=np.asarray(features);y=np.asarray(targets);train=np.array(mechanisms)!='topic';test=~train
 fits={}
 for name,ix in [('global_coverage',[0]),('composition',[0,1,2]),('title_local_coverage',[0,1,2,3,4])]:
  mean=X[train][:,ix].mean(0);scale=X[train][:,ix].std(0);scale[scale==0]=1
  xtr=np.column_stack([np.ones(train.sum()),(X[train][:,ix]-mean)/scale]);xte=np.column_stack([np.ones(test.sum()),(X[test][:,ix]-mean)/scale]);coef=np.linalg.lstsq(xtr,y[train],rcond=None)[0];pred=xte@coef
  fits[name]={'training_conditions':['random','source'],'heldout_condition':'topic','n_train_masks':int(train.sum()),'n_test_masks':int(test.sum()),'features':[ ['global_missing_fraction','abs_source_coverage_gap','abs_year_coverage_gap','abs_focal_group_local_coverage_gap','mean_local_coverage'][i] for i in ix],'train_feature_mean':mean.tolist(),'train_feature_sd':scale.tolist(),'intercept_and_scaled_coefficients':coef.tolist(),'test_mae_sd_units':float(np.mean(abs(pred-y[test]))),'test_rmse_sd_units':float(np.sqrt(np.mean((pred-y[test])**2))),'heldout_targets':y[test].tolist(),'heldout_predictions':pred.tolist(),'negative_predictions':int((pred<0).sum())}
 diagnostic={'status':'exploratory condition holdout only; no independent corpus/cohort validation','target':'absolute ACL-minus-EMNLP contrast distortion in frozen baseline SD units','method':'ordinary least squares; feature sets specified before outcomes, no outcome-selected tuning','shared_history_limitation':'All assignments share the selected historical corpus and focal sample; test transfers mask mechanism within this finite corpus only.','models':fits,'inputs_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [p/'mask_results.jsonl',p/'baseline.json',p/'summary.json',p/'run_manifest.json',ROOT/'data/processed/records.jsonl']},'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 (out/'diagnostic.json').write_text(json.dumps(diagnostic,indent=2))
 sd=baseline['evaluations']['restored']['outcomes']['baseline_population_sd'];paired=[]
 for r in summary['paired_primary']:
  if r['policy']=='mixed':
   q=r['excess_absolute_group_distortion'];paired.append({'mask':r['mechanism'],'rate':r['rate'],'mean_excess_sd':q['mean']/sd,'mc_se_sd':q['monte_carlo_se_mean']/sd,'n_masks':q['n_masks']})
 (out/'paired_primary.json').write_text(json.dumps(paired,indent=2))
 lines=['# Exploratory pilot readout','',f"Executed {len(raw)} base assignments plus matched controls. History={baseline['scope']['history_count']}; focal={baseline['scope']['focal_count']}. This is not confirmation or a claim about true novelty.",'','## Mixed historical representation','', '|Mask|Missing rate|Mean absolute group distortion (baseline SD)|MC SE|Top-decile replacement|','|---|---:|---:|---:|---:|']
 for r in rows:
  if r['policy']=='mixed':lines.append(f"|{r['mask']}|{r['rate']:.0%}|{r['mean_absolute_group_distortion_sd']:.4f}|{r['monte_carlo_se']:.4f}|{r['top_decile_replacement']:.1%}|")
 lines+=['','Uncertainty here is Monte Carlo error of a mean over independent complete masks, conditional on this selected corpus/sample. It does not quantify generalization across fields. Replacement denominator is the baseline top decile (30 papers), not all 300 focal papers. 50% is a stress test.','', '## Matched-control comparison','', '|Mask|Rate|Structured minus year-matched random absolute distortion (SD)|MC SE|','|---|---:|---:|---:|']
 for r in paired:lines.append(f"|{r['mask']}|{r['rate']:.0%}|{r['mean_excess_sd']:.4f}|{r['mc_se_sd']:.4f}|")
 lines+=['','## Deployment-available diagnostic, whole-mechanism holdout','', 'Train on random/source masks; test only topic masks. No hidden abstracts, baseline scores or induced drift are features. Baseline values are evaluation targets only. Linear model is deliberately simple; no tuning after looking at test results.','']
 for name,v in fits.items():lines.append(f"- {name}: held-out mean absolute prediction error={v['test_mae_sd_units']:.4f} SD; RMSE={v['test_rmse_sd_units']:.4f} SD; negative predictions={v['negative_predictions']}.")
 lines+=['','This one exploratory holdout does not establish deployable calibration. A later cohort or independent collection and frozen decision threshold remain essential. All-title invariance and restoration are pipeline checks, not contributions. Policy deviations are in conditions.csv. Full baseline contrast and signed values are in results/pilot/baseline.json and raw assignments.','', 'Sources: pinned ACL XML and SPECTER manifests; generating code analysis/pilot_readout.py; raw output results/pilot/mask_results.jsonl.']
 (ROOT/'docs/pilot_readout.md').write_text('\n'.join(lines)+'\n')
 os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'results/matplotlib_cache'))
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,ax=plt.subplots(figsize=(7,4.4))
 for mechanism,color,marker in [('random','#0072B2','o'),('source','#D55E00','s'),('topic','#009E73','^')]:
  sub=[r for r in rows if r['mask']==mechanism and r['policy']=='mixed'];ax.errorbar([r['rate']*100 for r in sub],[r['mean_absolute_group_distortion_sd'] for r in sub],yerr=[r['monte_carlo_se'] for r in sub],label=mechanism+' mask',color=color,marker=marker,capsize=4)
 ax.axhline(.1,color='#666666',linestyle='--',linewidth=.8,label='provisional 0.1 SD tolerance');ax.set(xlabel='Historical abstracts masked (%)',ylabel='Absolute group-contrast distortion\n(baseline SD units)',title='Exploratory pilot: equal coverage, different mask patterns');ax.legend(fontsize=8);fig.text(.02,.01,'Points: means over 20 masks; bars: ±1 Monte Carlo SE. Single corpus/cohort. 50% is a stress test.',fontsize=8);fig.tight_layout(rect=(0,.055,1,1));fig.savefig(ROOT/'figures/pilot_group_distortion.png',dpi=180);fig.savefig(ROOT/'figures/pilot_group_distortion.pdf');plt.close(fig)
 print(json.dumps({'analysis':'complete','diagnostic_mae':{k:v['test_mae_sd_units'] for k,v in fits.items()},'paired_primary':paired},indent=2))
if __name__=='__main__':main()
