"""Frozen, streaming confirmation readout. Never called by the scoring stage.

Run only after the root authorizes post-freeze analysis of a complete confirmation.
Requires numpy, matplotlib, and mpmath==1.3.0; does not depend on SciPy.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import numpy as np

STAGE = 'confirmation_v1'
FAMILY = 45
N_MASKS = 200
CRITICAL = 3.3090595211511644
# The scoring stage aggregates float32 vectors; this independent readout uses
# float64 accumulation. T/Q are discrete/rank identities; G permits only a
# prespecified small numerical reconciliation, never outcome-based tolerance.
RECONCILIATION_TOLERANCE = {'G': 2e-5, 'T': 2e-10, 'Q': 2e-10}
PRIMARY = {'G': 'standardized_mean_absolute_group_distortion',
           'T': 'mean_baseline_top_decile_replacement_fraction',
           'Q': 'mean_normalized_within_year_rank_movement'}
EDGES = np.array([-np.inf, -.1, -.05, -.025, -.01, 0, .01, .025, .05, .1, np.inf])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + '\n')


def save_csv(path, rows):
    require(bool(rows), 'No rows for ' + str(path))
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(clean(r) for r in rows)


def verify_critical():
    import mpmath as mp
    require(mp.__version__ == '1.3.0', 'Pin mpmath==1.3.0 for quantile verification')
    with mp.workdps(50):
        df = mp.mpf(199)
        tail = mp.mpf('0.05') / (2 * FAMILY)
        def survival(t):
            return mp.betainc(df/2, mp.mpf('.5'), 0, df/(df+t*t), regularized=True)/2
        computed = mp.findroot(lambda t: survival(t)-tail, (3, 4))
        require(abs(float(computed)-CRITICAL) < 1e-10 and CRITICAL >= float(computed), 'Student critical value mismatch')
        return {'mpmath_version': mp.__version__, 'df': 199, 'family': FAMILY,
                'alpha': .05, 'critical': CRITICAL, 'verified_decimal': str(computed)}


def average_ranks(values):
    """Zero-based ascending average ranks, exact equality only."""
    x = np.asarray(values, dtype=float)
    order = np.argsort(x, kind='stable')
    ranks = np.empty(len(x), dtype=float)
    starts = np.r_[0, np.flatnonzero(np.diff(x[order]) != 0) + 1]
    stops = np.r_[starts[1:], len(x)]
    for a, b in zip(starts, stops):
        ranks[order[a:b]] = (a+b-1)/2
    return ranks


def selected(scores, ids, fraction=.1):
    k = math.ceil(fraction * len(scores))
    result = np.zeros(len(scores), dtype=bool)
    result[np.lexsort((ids, -scores))[:k]] = True
    return result


def baseline_context(scores, ids, sources):
    b = np.asarray(scores, dtype=float)
    require(len(b) > 1 and np.isfinite(b).all(), 'Invalid baseline')
    ids, sources = np.asarray(ids), np.asarray(sources)
    sd = float(b.std(ddof=0))
    require(sd > 0, 'Zero baseline variance blocks confirmation')
    a, e = sources == 'ACL', sources == 'EMNLP'
    require(a.any() and e.any(), 'Both fixed source groups required')
    k = math.ceil(.1*len(b)); order = np.lexsort((ids, -b))
    kth, next_score = float(b[order[k-1]]), float(b[order[k]])
    cut = (kth+next_score)/2
    # One-based descending ordinal positions, percentile at each position's
    # center. Include [.09,.11] then expand both boundary exact-score ties.
    ordinal_percentile = (np.arange(1, len(b)+1)-.5)/len(b)
    band_positions = np.flatnonzero((ordinal_percentile >= .09) & (ordinal_percentile <= .11))
    lo = int(band_positions[0]) if len(band_positions) else None
    hi = int(band_positions[-1]) if len(band_positions) else None
    band = ((b <= b[order[lo]]) & (b >= b[order[hi]])) if lo is not None else np.zeros(len(b),dtype=bool)
    margin = (b-cut)/sd
    tie = b == kth if kth == next_score else np.zeros(len(b), dtype=bool)
    old = selected(b, ids)
    return dict(b=b, ids=ids, sources=sources, A=a, B=e, sd=sd, k=k, old=old,
        ranks=average_ranks(b), band=band, margin=margin,
        bins=np.searchsorted(EDGES, margin, side='right')-1,
        original_contrast=float(b[a].mean()-b[e].mean()),
        cutoff=dict(k=k, n=len(b), kth_score=kth, next_score=next_score,
            midpoint=cut, gap=kth-next_score, gap_baseline_sd=(kth-next_score)/sd,
            tied_block_size=int(tie.sum()), tied_block_selected=int((tie & old).sum()),
            rank_band_index_low=lo, rank_band_index_high=hi,
            rank_band_size=int(band.sum()), baseline_selected_in_band=int((old & band).sum()),
            baseline_selected_outside_band=int((old & ~band).sum()),
            band_rule='Descending ordinal percentile (r-0.5)/n, r=1..n; inclusive [0.09,0.11], expanded exact ties at both edges'))


def mask_metrics(scores, ctx):
    z = np.asarray(scores, dtype=float)
    require(z.shape == ctx['b'].shape, 'Focal score vector length/order mismatch')
    require(np.isfinite(z).all(), 'Undefined scores cannot change cohort')
    b, old, k = ctx['b'], ctx['old'], ctx['k']
    new = selected(z, ctx['ids']); lost = old & ~new; gained = ~old & new
    rank = average_ranks(z); movement = abs(rank-ctx['ranks'])/(len(z)-1)
    diff = float(z[ctx['A']].mean()-z[ctx['B']].mean())
    signed = (diff-ctx['original_contrast'])/ctx['sd']
    sp = float(np.corrcoef(rank, ctx['ranks'])[0, 1]) if np.std(rank) else None
    outside = old & ~ctx['band']
    result = dict(G=abs(signed), T=float(lost.sum()/k), Q=float(movement.mean()),
        signed_group_distortion_sd=signed, group_contrast=diff,
        group_contrast_sd=diff/ctx['sd'], mean_score_drift=float(np.mean(z-b)),
        spearman=sp, rank_movement_median=float(np.median(movement)),
        rank_movement_p90=float(np.quantile(movement, .9)),
        fraction_moving_more_than_5_percentiles=float(np.mean(movement > .05)),
        selected_lost_count=int(lost.sum()), selected_gained_count=int(gained.sum()),
        loss_in_cutoff_band_per_k=float((lost & ctx['band']).sum()/k),
        loss_outside_cutoff_band_per_k=float((lost & ~ctx['band']).sum()/k),
        loss_conditional_selected_outside_band=float((lost & ~ctx['band']).sum()/outside.sum()) if outside.any() else None)
    for delta in (.01, .025, .05, .1):
        count = int((lost & (ctx['margin'] > delta)).sum())
        result[f'far_loss_count_{delta}'] = count
        result[f'far_loss_per_k_{delta}'] = count/k
    for source in ('ACL', 'EMNLP'):
        ix = ctx['sources'] == source
        for name, mask in [('lost', lost), ('gained', gained)]:
            result[f'{source}_{name}_count'] = int((mask & ix).sum())
            result[f'{source}_{name}_per_k'] = float((mask & ix).sum()/k)
    for q in (.05, .2):
        s0 = selected(b, ctx['ids'], q); sm = selected(z, ctx['ids'], q)
        result[f'top_{q}_replacement'] = float((s0 & ~sm).sum()/s0.sum())
    bins = []
    for j in range(len(EDGES)-1):
        for status, target, event in [('baseline_selected', old, new), ('baseline_unselected', ~old, new)]:
            ix = (ctx['bins'] == j) & target
            bins.append((j, status, int(ix.sum()), float(event[ix].mean()) if ix.any() else None))
    return result, new, bins


def summarize(values, simultaneous=False):
    finite = [v for v in values if v is not None and math.isfinite(v)]
    n = len(finite)
    if simultaneous:
        require(n == N_MASKS and n == len(values), 'Primary family requires exactly 200 defined masks')
    if not n:
        return dict(n_masks=0, n_undefined=len(values), mean=None, median=None,
                    sd_over_masks=None, mcse=None, p05=None, p95=None,
                    ci_low_raw=None, ci_high_raw=None, interval='undefined')
    x = np.asarray(finite, dtype=float); sd = float(x.std(ddof=1)) if n > 1 else None
    se = sd/math.sqrt(n) if sd is not None else None
    half = CRITICAL*se if simultaneous else None
    return dict(n_masks=n, n_undefined=len(values)-n, mean=float(x.mean()), median=float(np.median(x)),
                sd_over_masks=sd, mcse=se, p05=float(np.quantile(x,.05)), p95=float(np.quantile(x,.95)),
                ci_low_raw=float(x.mean()-half) if half is not None else None,
                ci_high_raw=float(x.mean()+half) if half is not None else None,
                interval='approximate_95pct_FWER_45_Bonferroni_Student_mean' if simultaneous else 'descriptive_no_family_interval')


def wilson(count, n):
    z = 1.959963984540054; p = count/n; denom = 1+z*z/n
    center = (p+z*z/(2*n))/denom
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return p, center-half, center+half


def classify(g, t, q, tg=.1, tt=.05, tq=.01):
    stable = g['ci_high_raw'] < tg
    return dict(aggregate='below_tolerance' if stable else ('above_tolerance' if g['ci_low_raw'] > tg else 'unresolved'),
        selection='above_tolerance' if t['ci_low_raw'] > tt else ('below_tolerance' if t['ci_high_raw'] < tt else 'unresolved'),
        ranks='above_tolerance' if q['ci_low_raw'] > tq else ('below_tolerance' if q['ci_high_raw'] < tq else 'unresolved'),
        joint_mean_rule=bool(stable and t['ci_low_raw'] > tt),
        joint_with_rank_rule=bool(stable and t['ci_low_raw'] > tt and q['ci_low_raw'] > tq))


def provenance_gate(args):
    root, results = Path(args.root), Path(args.results)
    run = read_json(results/'run_manifest.json'); freeze = read_json(args.freeze_manifest)
    require(run.get('status') == 'complete' and run.get('stage') == STAGE, 'Complete confirmation required')
    require(freeze.get('status') == 'frozen' and freeze.get('stage') == STAGE, 'Frozen confirmation required')
    p = run['provenance']
    require(p['freeze_manifest_sha256'] == sha(args.freeze_manifest), 'Freeze manifest changed')
    require(p['freeze_git_commit'] == freeze['git_commit'], 'Freeze commit mismatch')
    require(freeze.get('analysis_code_sha256', {}).get('analysis/confirmatory_readout.py') == sha(__file__), 'Analysis code not frozen or changed')
    inputs = {name: sha(getattr(args, name)) for name in ('records','config','seed_manifest','protocol','cohort_manifest','encoding_manifest')}
    for name, key in [('records','records'),('config','config'),('seed_manifest','seeds'),('protocol','protocol')]:
        require(inputs[name] == p['inputs'][key] == freeze['sha256'][key], 'Frozen input changed: '+name)
    require(inputs['cohort_manifest'] == freeze['sha256']['cohort_manifest'], 'Cohort manifest not frozen')
    require(inputs['encoding_manifest'] == p['encoding_provenance']['manifest_sha256'], 'Encoding manifest changed')
    for name, expected in p['source_code_sha256'].items():
        require(sha(root/'src'/name) == expected, 'Scoring source changed: '+name)
    for name in ('baseline.json','mask_results.jsonl','summary.json'):
        require(sha(results/name) == run['output_sha256'][name], 'Output checksum mismatch: '+name)
        inputs[name] = run['output_sha256'][name]
    inputs['run_manifest'] = sha(results/'run_manifest.json'); inputs['freeze_manifest'] = sha(args.freeze_manifest)
    baseline = read_json(results/'baseline.json'); summary = read_json(results/'summary.json')
    require(baseline['run_fingerprint'] == summary['run_fingerprint'] == p['run_fingerprint'], 'Output fingerprints differ')
    config = read_json(args.config); seeds = read_json(args.seed_manifest); cohort = read_json(args.cohort_manifest)
    require(config['masks_per_condition'] == 200 and config['rates'] == [.1,.25,.5], 'Wrong confirmation design')
    require(cohort['quality_pass'] and cohort['records_sha256'] == inputs['records'], 'Cohort audit mismatch')
    records = {}
    with Path(args.records).open() as f:
        for line in f:
            row = json.loads(line); require(row['id'] not in records,'Duplicate record ID'); records[row['id']] = row
    focal = baseline['scope']['focal_ids']; history = baseline['scope']['history_ids']
    require(focal == run['scope']['focal_ids'] and history == run['scope']['history_ids'], 'Cohort order mismatch')
    require(set(focal) == {r['id'] for r in cohort['focal']} and set(history) == {r['id'] for r in cohort['history']}, 'Cohort manifest IDs differ')
    require(len(focal) == 1557 and len(history) == 3799 and len(set(focal)) == len(focal), 'Unexpected empirical cohort')
    require(all(records[i]['year'] == 2021 for i in focal), 'Focal chronology mismatch')
    require(all(records[i]['year'] in [2018,2019,2020] for i in history), 'History chronology mismatch')
    require(not set(focal)&set(history), 'Focal/history overlap')
    expected = {r['assignment_id']: r for r in seeds['assignments']}
    require(len(expected) == len(seeds['assignments']) == 1800, 'Seed assignment cardinality mismatch')
    return run, freeze, inputs, baseline, records, expected, cohort


def plot_figures(out, grouped, summaries, margin_rows):
    os.environ.setdefault('MPLCONFIGDIR', str(out/'mpl_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    labels = {'random':'Random','source':'Source-preferred','topic':'Lexical-proxy-preferred'}
    colors = {'random':'#226688','source':'#b54b32','topic':'#7765a8'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for metric, threshold, name, ylabel in [('G',.1,'figure1_aggregate','Absolute group-contrast distortion (baseline SD)'),('T',.05,'figure2_selection','Replacement among baseline top decile (k = 156)')]:
        fig, ax = plt.subplots(figsize=(7.2,4.5))
        for mechanism,marker,offset in zip(labels,['o','^','s'],[-.6,0,.6]):
            xs=[]; ys=[]; errors=[]
            for rate in [.1,.25,.5]:
                s=summaries[(rate,mechanism,metric)];xs.append(rate*100+offset);ys.append(s['mean']);errors.append(s['mean']-s['ci_low_raw'])
            ax.errorbar(xs,ys,yerr=errors,marker=marker,ls='none',capsize=3,label=labels[mechanism],color=colors[mechanism])
        ax.axhline(threshold,color='black',ls='--',lw=1,label='Prespecified tolerance')
        ax.set(xlabel='Synthetic historical abstract loss (%)',ylabel=ylabel,xticks=[10,25,50])
        ax.set_ylim(bottom=min(0,ax.get_ylim()[0]))
        if metric=='T':ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(fontsize=8);fig.text(.1,.01,'Bars: approximate 95% familywise mean intervals (45 Bonferroni outcomes). 50% is stress.',fontsize=8)
        fig.tight_layout(rect=(0,.04,1,1));fig.savefig(out/(name+'.png'),dpi=200);fig.savefig(out/(name+'.pdf'));plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(11.8,3.9),sharex=True,sharey=True)
    for ax, mechanism in zip(axes,labels):
        for rate, marker, color in zip([.1,.25,.5],['o','^','s'],['#226688','#b54b32','#7765a8']):
            rows=grouped[(rate,mechanism,'mixed')]
            ax.scatter([r['G'] for r in rows],[r['T'] for r in rows],s=9,alpha=.35,marker=marker,color=color,label=f'{rate:.0%}')
        ax.axvline(.1,color='black',ls='--',lw=.8);ax.axhline(.05,color='black',ls='--',lw=.8)
        ax.set_title(labels[mechanism]);ax.set_xlabel('G (baseline SD units)')
    axes[0].set_ylabel('T (fraction of baseline k = 156)');axes[-1].legend(title='Loss rate',fontsize=8)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    fig.text(.06,.01,'Each point is one complete mask. Dashed lines define descriptive per-mask events, not mean-level inference.',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(out/'figure3_joint.png',dpi=200);fig.savefig(out/'figure3_joint.pdf');plt.close(fig)
    # Descriptive errors use complete-mask MCSE, not independent-paper errors.
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True)
    bin_labels=['<−.10','−.10:−.05','−.05:−.025','−.025:−.01','−.01:0','0:.01','.01:.025','.025:.05','.05:.10','≥.10']
    for col,mechanism in enumerate(labels):
        for row_index,status in enumerate(['baseline_selected','baseline_unselected']):
            ax=axes[row_index,col]
            for rate,offset,color,marker in zip([.1,.25,.5],[-.15,0,.15],['#226688','#b54b32','#7765a8'],['o','^','s']):
                subset=[r for r in margin_rows if r['condition']==mechanism and r['rate']==rate and r['baseline_status']==status]
                labelled=False
                for j,r in enumerate(subset):
                    if r['mean'] is None:continue
                    ax.errorbar(j+offset,r['mean'],yerr=1.96*r['mcse'],fmt=marker,ms=3,capsize=2,color=color,
                        label=f'{rate:.0%}' if not labelled else None)
                    labelled=True
            ax.set_title(labels[mechanism]+' — '+('retention' if row_index==0 else 'entry'),fontsize=10)
            ax.set_xticks(range(10),bin_labels,rotation=60,ha='right');ax.set_ylim(-.03,1.03)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            if col==0:ax.set_ylabel('Probability over masks')
            if row_index==0:ax.legend(fontsize=7,title='Loss rate')
    fig.text(.04,.01,'Signed baseline margin / baseline SD. Empty bins have no points; bin sizes are in cutoff_margin_bins.csv. Bars: ±1.96 whole-mask MCSE (descriptive).',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(out/'figure4_cutoff_margins.png',dpi=200);fig.savefig(out/'figure4_cutoff_margins.pdf');plt.close(fig)


def run(args):
    # All integrity checks precede reading score vectors or emitting inferential results.
    manifest, freeze, inputs, baseline, records, seeds, cohort = provenance_gate(args)
    quantile = verify_critical(); fingerprint = manifest['provenance']['run_fingerprint']
    ids = baseline['scope']['focal_ids']; sources = [str(records[i]['source']).upper() for i in ids]
    ctx = baseline_context(baseline['evaluations']['restored']['novelty_scores'],ids,sources)
    require(ctx['k'] == 156 and Counter(sources) == {'ACL':710,'EMNLP':847}, 'Source selection contract failed')
    out = Path(args.output);out.mkdir(parents=True,exist_ok=True)
    grouped=defaultdict(list);paired=defaultdict(list);bin_values=defaultdict(list);selection_counts=defaultdict(lambda:np.zeros(len(ids),dtype=int))
    seen=set(); mask_rows=[]; control_rows=[]; reconciliation=defaultdict(float)
    baseline_title=np.asarray(baseline['evaluations']['all_title']['novelty_scores'],dtype=float)
    with (Path(args.results)/'mask_results.jsonl').open() as f:
        for line in f:
            row=json.loads(line); name=row['assignment_id']
            require(name in seeds and name not in seen, 'Unknown/duplicate assignment')
            seen.add(name)
            require(row['run_fingerprint'] == fingerprint and row['stage']==STAGE,'Row provenance mismatch')
            content={k:v for k,v in row.items() if k!='content_sha256'}
            computed=hashlib.sha256(json.dumps(content,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            require(computed == row['content_sha256'],'Raw assignment checksum mismatch')
            for key,value in seeds[name].items():require(row[key]==value,'Seed assignment mismatch: '+key)
            rate, mechanism = row['rate'],row['mechanism'];role_metrics={}
            for role in ['structured_or_random','matched_random_control']:
                if role not in row:
                    require(role!='matched_random_control' or mechanism=='random','Missing matched control');continue
                condition=mechanism if role=='structured_or_random' else mechanism+'_matched_control'
                assignment=row[role]
                require(assignment['missing_count']==math.floor(len(baseline['scope']['history_ids'])*rate+.5),'Mask count mismatch')
                missing_ids=assignment['missing_history_ids']
                require(len(missing_ids)==len(set(missing_ids))==assignment['missing_count'] and
                    set(missing_ids)<=set(baseline['scope']['history_ids']),'Masked ID/count mismatch')
                require(set(assignment['evaluations'])=={'mixed','complete_case','all_title','restored'},'Missing/extra representation policy')
                for policy, evaluation in assignment['evaluations'].items():
                    scores=np.asarray(evaluation['novelty_scores'],dtype=float)
                    require(scores.shape==ctx['b'].shape,'Focal score vector length/order mismatch')
                    if policy=='restored':require(np.array_equal(scores,ctx['b']),'Restoration identity failed')
                    if policy=='all_title':require(np.array_equal(scores,baseline_title),'All-title identity failed')
                    if not np.isfinite(scores).all():
                        require(policy=='complete_case','Undefined non-ablation score')
                        metrics={k:None for k in ['G','T','Q']};metrics['n_undefined_scores']=int((~np.isfinite(scores)).sum())
                    else:
                        metrics,new,bins=mask_metrics(scores,ctx)
                        for endpoint,key in PRIMARY.items():
                            discrepancy=abs(metrics[endpoint]-evaluation['outcomes'][key])
                            reconciliation[endpoint]=max(reconciliation[endpoint],discrepancy)
                            require(discrepancy<RECONCILIATION_TOLERANCE[endpoint],'Independent '+endpoint+' mismatch')
                        if policy=='mixed':
                            selection_counts[(rate,condition)] += new
                            for j,status,n,prob in bins:bin_values[(rate,condition,j,status,n)].append(prob)
                    grouped[(rate,condition,policy)].append(metrics)
                    if policy=='mixed':
                        role_metrics[role]=metrics
                        mask_rows.append(dict(assignment_id=name,condition=condition,rate=rate,replicate=row['replicate'],**metrics))
                if role=='matched_random_control':
                    wanted=['year'] if mechanism=='source' else ['year','source']
                    require(row['control_matching']==wanted,'Control matching mismatch')
                    def cell(record_id):
                        return tuple(records[record_id][field] if field=='year' else str(records[record_id][field]).upper() for field in wanted)
                    totals=Counter(cell(i) for i in baseline['scope']['history_ids'])
                    structured_counts=Counter(cell(i) for i in row['structured_or_random']['missing_history_ids'])
                    control_counts=Counter(cell(i) for i in missing_ids)
                    require(structured_counts==control_counts,'Control cell counts not preserved')
                    declared={tuple(c['cell'][field] for field in wanted):c for c in row['control_cells']}
                    require(set(declared)==set(totals) and len(declared)==len(row['control_cells']),'Control cells missing/duplicated')
                    for key,n in totals.items():
                        require(declared[key]['total']==n and declared[key]['missing']==structured_counts[key],'Control cell ledger differs from IDs')
                    control_rows.append(dict(assignment_id=name,mechanism=mechanism,rate=rate,matching=';'.join(wanted),
                        zero_cells=sum(c['missing']==0 for c in row['control_cells']),
                        full_cells=sum(c['missing']==c['total'] for c in row['control_cells']),
                        nondegenerate_cells=sum(0<c['missing']<c['total'] for c in row['control_cells'])))
            if mechanism!='random':
                for metric in PRIMARY:
                    value=role_metrics['structured_or_random'][metric]-role_metrics['matched_random_control'][metric]
                    discrepancy=abs(value-row['paired_primary_differences'][metric])
                    reconciliation['paired_'+metric]=max(reconciliation['paired_'+metric],discrepancy)
                    require(discrepancy<2*RECONCILIATION_TOLERANCE[metric],'Paired outcome mismatch')
                    paired[(rate,mechanism,metric)].append(value)
    require(seen==set(seeds),'Incomplete mask assignments')
    require(len(control_rows)==1200,'Incomplete matched controls')
    table1=[]
    for scope, rids in [('history',baseline['scope']['history_ids']),('focal',ids)]:
        for (year,source),n in sorted(Counter((records[i]['year'],records[i]['source']) for i in rids).items()):
            table1.append(dict(cohort=scope,year=year,source=source,n=n,availability='complete title/abstract by frozen audit',
                exclusions_new=len(cohort['new_duplicate_exclusions']) if scope=='focal' else None,
                baseline_top_decile_k=ctx['k'] if scope=='focal' else None))
    table2=[];table3=[];table4=[];summaries={};joint=[];threshold=[]
    for (rate,condition,policy),values in sorted(grouped.items()):
        primary=policy=='mixed' and condition in ['random','source','topic']
        for metric in dict.fromkeys(k for v in values for k in v):
            stat=summarize([v.get(metric) for v in values],simultaneous=primary and metric in PRIMARY)
            row=dict(rate=rate,condition=condition,policy=policy,metric=metric,**stat)
            (table2 if metric in ['G','signed_group_distortion_sd','group_contrast','group_contrast_sd','mean_score_drift'] else table3).append(row)
            if primary and metric in PRIMARY:summaries[(rate,condition,metric)]=stat
        if primary:
            state=classify(*(summaries[(rate,condition,m)] for m in PRIMARY))
            count=sum(v['G']<.1 and v['T']>.05 for v in values);freq,lo,hi=wilson(count,len(values))
            joint.append(dict(rate=rate,condition=condition,**state,descriptive_joint_masks=count,n_masks=len(values),
                descriptive_joint_fraction=freq,wilson95_low=lo,wilson95_high=hi,
                wilson_scope='pointwise descriptive; outside 45-mean family'))
            for tg in [.05,.1,.15]:
                for tt in [.025,.05,.1]:
                    for tq in [.005,.01,.02]:threshold.append(dict(rate=rate,condition=condition,tau_G=tg,tau_T=tt,tau_Q=tq,
                        **classify(*(summaries[(rate,condition,m)] for m in PRIMARY),tg,tt,tq)))
    for (rate,mechanism,metric),values in sorted(paired.items()):
        table4.append(dict(rate=rate,mechanism=mechanism,metric=metric,contrast='structured minus matched random',
            matching='year' if mechanism=='source' else 'year;source',**summarize(values,True)))
    require(len(summaries)==27 and len(table4)==18,'45-mean family incomplete')
    margin_rows=[]
    for (rate,condition,j,status,n),values in sorted(bin_values.items()):
        margin_rows.append(dict(rate=rate,condition=condition,
            bin_left='-inf' if np.isneginf(EDGES[j]) else EDGES[j],
            bin_right='+inf' if np.isposinf(EDGES[j+1]) else EDGES[j+1],
            bin_convention='left-inclusive right-exclusive; exact zero in [0,.01)',baseline_status=status,n_papers=n,
            endpoint='retention' if status=='baseline_selected' else 'entry',**summarize(values)))
    individual=[]
    for (rate,condition),counts in sorted(selection_counts.items()):
        n=len(grouped[(rate,condition,'mixed')])
        for i,record_id in enumerate(ids):individual.append(dict(record_id=record_id,source=sources[i],rate=rate,condition=condition,
            baseline_selected=bool(ctx['old'][i]),signed_margin_sd=float(ctx['margin'][i]),in_cutoff_band=bool(ctx['band'][i]),
            selected_probability=float(counts[i]/n),n_masks=n))
    for filename,rows in [('table1_cohort_construction.csv',table1),('table2_aggregate.csv',table2),('table3_rank_selection.csv',table3),
        ('table4_paired_primary.csv',table4),('joint_classification.csv',joint),('threshold_sensitivity.csv',threshold),
        ('cutoff_margin_bins.csv',margin_rows),('paper_selection_probabilities.csv',individual),('mask_primary_and_movement.csv',mask_rows),
        ('control_cell_counts.csv',control_rows)]:save_csv(out/filename,rows)
    save_json(out/'baseline_cutoff.json',dict(**ctx['cutoff'],baseline_population_sd=ctx['sd'],
        baseline_group_contrast=ctx['original_contrast'],baseline_group_contrast_sd=ctx['original_contrast']/ctx['sd'],
        far_loss_denominator=ctx['k'],rank_normalization_denominator=len(ids)-1))
    plot_figures(out,grouped,summaries,margin_rows)
    lines=['# Confirmation v1 readout','',f'All {len(ids):,} focal papers; {len(baseline["scope"]["history_ids"]):,} historical records. '
        f'Top-decile denominator: **k = {ctx["k"]}**, not all focal papers. Each condition has 200 complete independent mask assignments.',
        '',f'Baseline ACL-minus-EMNLP contrast: {ctx["original_contrast"]:.6g} raw; {ctx["original_contrast"]/ctx["sd"]:.6g} baseline SD. '
        f'Cutoff gap: {ctx["cutoff"]["gap_baseline_sd"]:.6g} baseline SD; cutoff tied block: {ctx["cutoff"]["tied_block_size"]}.','',
        '| Mechanism | Loss rate | G mean [simultaneous interval] | T mean [interval] | Q mean [interval] | Joint mean rule |',
        '|---|---:|---|---|---|---|']
    for j in joint:
        cells=[]
        for m in PRIMARY:
            s=summaries[(j['rate'],j['condition'],m)];cells.append(f'{s["mean"]:.4f} [{s["ci_low_raw"]:.4f}, {s["ci_high_raw"]:.4f}]')
        lines.append(f'| {j["condition"]} | {j["rate"]:.0%} | '+ ' | '.join(cells)+f' | {j["joint_mean_rule"]} |')
    lines += ['', 'The joint rule requires the upper G interval below 0.10 and the lower T interval above 0.05. '
        'An additional Q lower bound above 0.01 is reported separately. Touching a threshold is unresolved. '
        'Threshold sensitivity, raw mask distributions and pointwise descriptive joint-event Wilson intervals are separate outputs.',
        '', '## Cutoff interpretation', '',
        'Inspect Table 3 and cutoff_margin_bins.csv before interpreting replacement as broad ranking disorder. '
        'Loss outside the fixed 9th–11th-percentile band and losses beyond margins of 0.01, 0.025, 0.05 and 0.10 baseline SD '
        'all retain k as denominator; the conditional outside-band denominator is separately recorded in baseline_cutoff.json. '
        'Bin MCSEs summarize complete-mask probabilities, not independent-paper errors. Empty bins remain visible. '
        'Individual selection probabilities are descriptive, not separately tested claims.',
        '', 'The fixed band uses descending ordinal percentiles (r−0.5)/n in inclusive [0.09,0.11], with all exact-score ties at either edge included. '
        'Signed-margin bins are left-inclusive and right-exclusive; exact zero is in [0,0.01). '
        f'The band contains {ctx["cutoff"]["rank_band_size"]} papers, including {ctx["cutoff"]["baseline_selected_in_band"]} baseline selections; '
        f'{ctx["cutoff"]["baseline_selected_outside_band"]} baseline selections lie outside it.',
        '', '| Mechanism | Loss rate | Mean selected losses | Mean losses outside band / k | Mean far loss (>0.10 SD) / k | Mean per-mask rank-movement median | Mean per-mask rank-movement p90 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for j in joint:
        values=grouped[(j['rate'],j['condition'],'mixed')]
        selected_metrics=['selected_lost_count','loss_outside_cutoff_band_per_k','far_loss_per_k_0.1','rank_movement_median','rank_movement_p90']
        cells=[f'{np.mean([v[m] for v in values]):.4f}' for m in selected_metrics]
        lines.append(f'| {j["condition"]} | {j["rate"]:.0%} | '+' | '.join(cells)+' |')
    lines += [
        '', '## Limits', '',
        'This is a prospective confirmation after an exploratory pilot, not a preregistration. Inference is conditional on this selected '
        'ACL/EMNLP collection, fixed encoder, focal cohort and synthetic mask process. The full-data reference is not true scientific novelty. '
        'All loss rates are synthetic; 50% is a stress condition. Tolerances are design choices without external policy validation. '
        'The 45 Bonferroni Student mean intervals use a large-sample approximation for bounded, possibly skewed outcomes; '
        'they do not guarantee exact finite-sample coverage. Mean-level coexistence need not occur in most individual masks. '
        'Near-tie exchanges can change selections without substantial substantive differences. Structured-control contrasts retain '
        'correlated content effects within matched cells and do not recreate real database omissions. Shared corpus uncertainty and '
        'generalization to other fields are unmeasured. The failed pilot diagnostic remains failed; no new diagnostic is validated here. '
        'Unresolved closest-paper methods and a second public corpus remain publication-readiness limitations.',
        '', '## Artifacts', '', 'Four primary CSV tables, cutoff/movement supplements, threshold grid, three primary figures and a cutoff-margin figure, and the analysis '
        'manifest are generated from verified score vectors. No favorable condition is selected for reporting.']
    report=Path(args.report);report.parent.mkdir(parents=True,exist_ok=True);report.write_text('\n'.join(lines)+'\n')
    try: git_head=subprocess.check_output(['git','-C',str(args.root),'rev-parse','HEAD'],text=True).strip()
    except subprocess.CalledProcessError:git_head=None
    files={f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name!='analysis_manifest.json'}
    save_json(out/'analysis_manifest.json',dict(stage=STAGE,status='complete',created_utc=datetime.now(timezone.utc).isoformat(),
        analysis_code_sha256=sha(__file__),inputs_sha256=inputs,scoring_provenance=manifest['provenance'],freeze_git_commit=freeze['git_commit'],
        analysis_git_commit=git_head,quantile_verification=quantile,python=platform.python_version(),numpy=np.__version__,
        numerical_reconciliation=dict(max_abs_discrepancies=dict(reconciliation),
            endpoint_tolerances=RECONCILIATION_TOLERANCE,paired_tolerance_multiplier=2,
            note='Independent float64 reductions versus scoring-stage float32 reductions; reported analysis uses float64'),
        n_assignments=len(seen),n_matched_controls=len(control_rows),family_size=FAMILY,output_sha256=files,
        report_path=str(report),report_sha256=sha(report)))
    return {'status':'complete','output':str(out),'report':str(report),'family_size':45}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['root','results','records','config','seed-manifest','freeze-manifest','protocol','cohort-manifest','encoding-manifest','output','report']:
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args();print(json.dumps(run(args)))


if __name__=='__main__':main()
