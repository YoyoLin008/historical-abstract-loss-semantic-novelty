#!/usr/bin/env python3
"""Revision-stage descriptive summaries; original experiment and inference unmodified.

Portable: reads only frozen numeric projections from ESM_2_contents/data/derived.
No original endpoint code imports. See POST_HOC_ANALYSIS_SPEC.md for prior scope.
"""
import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

EDGES = np.array([-np.inf, -.10, -.05, -.025, -.01, 0., .01, .025, .05, .10, np.inf])
SPECS = {'ACL': ('acl', 'ACL', 'EMNLP', 1557, 710, 156),
         'Biomedical': ('biomedical', 'PLOSCB', 'PLOSGEN', 1471, 934, 148)}
LABELS = {'random': 'Random', 'source': 'Source', 'topic': 'Lexical',
          'source_matched_control': 'Source control', 'topic_matched_control': 'Lexical control'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def boolean(x):
    require(str(x) in ('True', 'False'), 'Invalid boolean: ' + str(x))
    return str(x) == 'True'


def average_ranks(x):
    """Ascending one-based average ranks, preserving all exact-score ordering."""
    x = np.asarray(x, dtype=np.float64)
    require(x.ndim == 1 and len(x) > 0 and np.isfinite(x).all(), 'Invalid ranks input')
    order = np.argsort(x, kind='stable')
    out = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        out[order[i:j]] = (i + 1 + j) / 2
        i = j
    return out


def selected(x, ids):
    x = np.asarray(x, dtype=np.float64)
    ids = np.asarray(ids, dtype=str)
    require(len(x) == len(ids) == len(set(ids)) and np.isfinite(x).all(), 'Invalid top-k IDs/scores')
    k = math.ceil(.1 * len(x))
    ix = np.lexsort((ids, -x))[:k]
    return set(ids[ix])


def aligned_scores(left, right):
    require(len(left) == len({r['paper_id'] for r in left}), 'Duplicate left ID')
    require(len(right) == len({r['paper_id'] for r in right}), 'Duplicate right ID')
    a = {r['paper_id']: float(r['score']) for r in left}
    b = {r['paper_id']: float(r['score']) for r in right}
    require(set(a) == set(b), 'Encoder focal ID sets differ; refusing silent join drops')
    ids = sorted(a)
    return ids, np.array([a[i] for i in ids]), np.array([b[i] for i in ids])


def rank_agreement(a, b):
    ra, rb = average_ranks(a), average_ranks(b)
    rho = None if np.std(ra) == 0 or np.std(rb) == 0 else float(np.corrcoef(ra, rb)[0, 1])
    return rho, float(np.mean(np.abs(ra-rb))/(len(a)-1)), ra, rb


def source_shifts(d, m, w):
    require(0 < w < 1, 'Invalid source weight')
    return np.asarray(m)+(1-w)*np.asarray(d), np.asarray(m)-w*np.asarray(d)


def mixed_only(data):
    result = [r for r in data if r['policy'] == 'mixed']
    keys = [(r['condition'], float(r['rate']), r['metric']) for r in result]
    require(len(keys) == len(set(keys)), 'Duplicate mixed aggregate row')
    return result


def one_source_map(probabilities):
    out = {}
    for r in probabilities:
        i, source = r['record_id'], r['source']
        require(i not in out or out[i] == source, 'Inconsistent source across probability rows')
        out[i] = source
    return out


def capacity(lost, baseline, band):
    require(lost <= baseline, 'Lost ID was not selected at baseline')
    inside, outside = len(lost & band), len(lost-band)
    n, cap = len(lost), len(baseline & band)
    require(n == inside + outside and 0 <= inside <= cap, 'Capacity partition failed')
    bound = max(0, n-cap)
    require(outside >= bound, 'Capacity lower bound failed')
    return n, inside, outside, bound


def nested_sets(outside, far):
    intersection = len(outside & far)
    return dict(outside_n=len(outside), margin_gt_point1_n=len(far), intersection_n=intersection,
                margin_subset_outside=far <= outside, outside_subset_margin=outside <= far,
                outside_only_n=len(outside-far), margin_only_n=len(far-outside))


def bins_for(margins):
    return np.searchsorted(EDGES, np.asarray(margins), side='right')-1


def conditional(losses, denominator):
    return None if denominator == 0 else float(np.mean(losses)/denominator)


def floatcheck(a, b, label, atol=2e-12):
    require(np.allclose(a, b, rtol=0, atol=atol), label + ' differs')
    return float(np.max(np.abs(np.asarray(a)-np.asarray(b))))


def edge_label(x):
    return '-inf' if np.isneginf(x) else '+inf' if np.isposinf(x) else format(float(x), '.12g')


class Outputs:
    def __init__(self, out):
        self.out = out
        self.provenance = []
        self.row_counts = {}
        out.mkdir(parents=True, exist_ok=True)

    def csv(self, name, data, inputs, filters, formula):
        require(len(data) > 0, 'No output rows for '+name)
        keys = list(dict.fromkeys(k for r in data for k in r))
        with (self.out/name).open('w', newline='') as f:
            writer = csv.DictWriter(f, keys)
            writer.writeheader()
            writer.writerows(data)
        self.row_counts[name] = len(data)
        for n, r in enumerate(data, start=2):
            self.provenance.append(dict(output_file=name, csv_line=n, output_key=json.dumps({k: r[k] for k in ('study','encoder','source','condition','rate','role','assignment_id','paper_id','bin_index') if k in r}, sort_keys=True),
                source_files=';'.join(inputs), source_filter=filters, formula=formula, generating_code='analysis/revision_descriptive_outputs.py',
                generated_columns=';'.join(keys)))


def run(data, output):
    out = Outputs(output)
    inputs = {}
    errors = defaultdict(float)
    def load(rel, json_file=False):
        path = data/rel
        inputs[str(rel)] = sha(path)
        return json.loads(path.read_text()) if json_file else rows(path)
    membership = load('cutoff_verification/baseline_membership.csv')
    expected_baselines = load('post_confirmation/specter2/baseline_reference_comparison.csv')
    shift_rows, shift_masks, capacity_rows, capacity_masks, conditional_rows = [], [], [], [], []
    agreement_rows, composition_rows, aligned_rows, nesting_rows = [], [], [], []
    baseline_rows = []
    for study, (slug, source_a, source_b, expected_n, expected_a, expected_k) in SPECS.items():
        prefix = f'original_{slug}'
        agg = load(f'{prefix}/table2_aggregate.csv')
        masks = load(f'{prefix}/mask_primary_and_movement.csv')
        probs = load(f'{prefix}/paper_selection_probabilities.csv')
        original_bins = load(f'{prefix}/cutoff_margin_bins.csv')
        cutoff = load(f'{prefix}/baseline_cutoff.json', True)
        sources = one_source_map(probs)
        require(len(sources) == expected_n and Counter(sources.values()) == {source_a: expected_a, source_b: expected_n-expected_a}, 'Original cohort/source counts differ')
        baseline = {encoder: [r for r in membership if r['study'] == study and r['encoder'] == encoder] for encoder in ('SPECTER','SPECTER2')}
        ids, b, b2 = aligned_scores(baseline['SPECTER'], baseline['SPECTER2'])
        require(len(ids) == expected_n and set(ids) == set(sources), 'Baseline/source exact IDs differ')
        source_flags = np.array([sources[i] == source_a for i in ids])
        require(np.isfinite(b).all() and np.isfinite(b2).all(), 'Nonfinite baseline')
        s0 = float(np.std(b, ddof=0)); c0 = float((b[source_flags].mean()-b[~source_flags].mean())/s0)
        errors['baseline_sd'] = max(errors['baseline_sd'], floatcheck(s0, cutoff['baseline_population_sd'], 'Baseline SD'))
        floatcheck(c0, cutoff['baseline_group_contrast_sd'], 'Baseline source contrast')
        baseline_sets = {}
        for encoder, values in [('SPECTER', b), ('SPECTER2', b2)]:
            s = selected(values, ids); baseline_sets[encoder] = s
            require(len(s) == expected_k, 'Wrong top-k')
            declared = {r['paper_id'] for r in baseline[encoder] if boolean(r['selected'])}
            require(s == declared, 'Recomputed baseline selections differ from saved projection')
            refs = [r for r in expected_baselines if r['study'] == slug and r['encoder'] == encoder]
            require(len(refs) == 1, 'Reference baseline not unique')
            sd = float(np.std(values, ddof=0)); contrast = float(values[source_flags].mean()-values[~source_flags].mean())
            floatcheck(sd, float(refs[0]['pop_sd']), 'Encoder own population SD')
            floatcheck(contrast, float(refs[0]['source_contrast_raw']), 'Encoder own source contrast')
            baseline_rows.append(dict(study=study, encoder=encoder, n=expected_n, n_A=expected_a, n_B=expected_n-expected_a, k=expected_k, source_A=source_a, source_B=source_b, population_sd=sd, contrast_raw=contrast, contrast_sd=contrast/sd))
            for source in [source_a, source_b]:
                selected_n = sum(sources[i] == source for i in s)
                source_n = sum(v == source for v in sources.values())
                composition_rows.append(dict(study=study, encoder=encoder, source=source, selected_n=selected_n, k=expected_k, selected_composition=selected_n/expected_k, source_cohort_n=source_n, within_source_selection_rate=selected_n/source_n))
        rho, qcross, ranks, ranks2 = rank_agreement(b, b2)
        overlap = len(baseline_sets['SPECTER'] & baseline_sets['SPECTER2'])
        agreement_rows.append(dict(study=study, n=expected_n, k=expected_k, rho_cross=rho, Q_cross=qcross, overlap_n=overlap, overlap_fraction=overlap/expected_k, replacement_fraction=1-overlap/expected_k))
        for i, paper_id in enumerate(ids):
            aligned_rows.append(dict(study=study, paper_id=paper_id, source=sources[paper_id], specter_score=float(b[i]), specter2_score=float(b2[i]), specter_average_rank=float(ranks[i]), specter2_average_rank=float(ranks2[i]), specter_selected=paper_id in baseline_sets['SPECTER'], specter2_selected=paper_id in baseline_sets['SPECTER2']))
        base = baseline_sets['SPECTER']
        # Reconstruct original inclusive rank band with exact tie expansion.
        order = np.lexsort((np.array(ids), -b)); percentiles = (np.arange(1, expected_n+1)-.5)/expected_n
        band_positions = np.flatnonzero((percentiles >= .09) & (percentiles <= .11))
        band = {ids[j] for j in range(expected_n) if b[j] <= b[order[band_positions[0]]] and b[j] >= b[order[band_positions[-1]]]}
        declared_band = {r['paper_id'] for r in baseline['SPECTER'] if boolean(r['in_band'])}
        require(band == declared_band, 'Tie-expanded band differs')
        midpoint = float((b[order[expected_k-1]]+b[order[expected_k]])/2)
        margin = {i: float((x-midpoint)/s0) for i, x in zip(ids,b)}
        for r in baseline['SPECTER']:
            floatcheck(margin[r['paper_id']], float(r['margin_sd']), 'Margin projection')
        far = {i for i in base if margin[i] > .1}
        outside = base-band
        nesting_rows.append(dict(study=study, k=expected_k, band_capacity=len(base & band), **nested_sets(outside,far)))
        require(far <= outside or outside <= far, 'Empirical nesting changed')
        metrics = {(r['condition'], float(r['rate']), r['metric']):r for r in mixed_only(agg)}
        conditions = sorted({(r['condition'], float(r['rate'])) for r in mixed_only(agg)})
        expected_conditions = {(c, rate) for c in (['random','source','topic','source_matched_control','topic_matched_control'] if study=='ACL' else ['random','source','source_matched_control']) for rate in ([.1,.25,.5] if study=='ACL' else [.25,.5])}
        require(set(conditions)==expected_conditions, 'Condition coverage incomplete/expanded')
        # Frozen endpoint table reuses pair ID for controls; lost projection names arms.
        def arm_id(r):
            return r['assignment_id'] + ('_matched_random' if 'matched_control' in r['condition'] else '')
        mask_by_id = {arm_id(r):r for r in masks}
        require(len(mask_by_id)==len(masks), 'Duplicate mask ID')
        lost_path = data/f'cutoff_verification/{slug}_lost_ids.jsonl'
        inputs[f'cutoff_verification/{slug}_lost_ids.jsonl'] = sha(lost_path)
        lost_records = [json.loads(line) for line in lost_path.read_text().splitlines()]
        require({r['assignment_id'] for r in lost_records} == set(mask_by_id), 'Lost-record IDs do not exactly match original masks')
        require(len(lost_records)==len(mask_by_id), 'Duplicate lost record')
        lost_lookup = {r['assignment_id']:r for r in lost_records}
        for condition, rate in conditions:
            role = 'control' if 'matched_control' in condition else 'base'
            def mean(metric):
                row = metrics[(condition,rate,metric)]
                require(int(row['n_masks'])==200 and int(row['n_undefined'])==0, 'Bad aggregate denominator')
                return float(row['mean'])
            d, m = mean('signed_group_distortion_sd'), mean('mean_score_drift')/s0
            a, bb = source_shifts(d,m,expected_a/expected_n)
            c = mean('group_contrast_sd'); g = mean('G')
            floatcheck(c,c0+d,'C=C0+D'); floatcheck(mean('group_contrast')/s0,c,'Raw and SD contrast')
            floatcheck(a-bb,d,'D=A-B');floatcheck((expected_a*a+(expected_n-expected_a)*bb)/expected_n,m,'M weighted sources')
            shift_rows.append(dict(study=study, condition=condition, rate=rate, role=role, n_masks=200, source_A=source_a, source_B=source_b, source_A_weight=expected_a/expected_n, baseline_sd=s0, baseline_contrast_sd=c0, masked_contrast_sd=c, signed_change_sd=d, mean_G=g, abs_mean_D=abs(d), mean_drift_sd=m, source_A_shift_sd=float(a), source_B_shift_sd=float(bb), contrast_change_fraction_of_baseline=d/c0, derivation='algebraic reconstruction from full-precision frozen means', analysis_label='revision-stage descriptive summary of frozen results'))
            mm = [r for r in masks if r['condition']==condition and float(r['rate'])==rate]
            require(len(mm)==200,'Not 200 original masks')
            ds=np.array([float(r['signed_group_distortion_sd']) for r in mm]); ms=np.array([float(r['mean_score_drift'])/s0 for r in mm])
            aa,bbb=source_shifts(ds,ms,expected_a/expected_n)
            floatcheck(ds.mean(),d,'Mean signed distortion');floatcheck(np.abs(ds).mean(),g,'Mean absolute distortion');floatcheck(ms.mean(),m,'Mean drift');floatcheck(aa.mean(),a,'Source A algebra');floatcheck(bbb.mean(),bb,'Source B algebra')
            cap_values=[]; bin_counts=np.zeros((200,len(EDGES)-1),dtype=int)
            for ix,r in enumerate(mm):
                shift_masks.append(dict(study=study,condition=condition,rate=rate,role=role,assignment_id=arm_id(r),original_pair_assignment_id=r['assignment_id'],signed_change_sd=float(ds[ix]),mean_drift_sd=float(ms[ix]),source_A_shift_sd=float(aa[ix]),source_B_shift_sd=float(bbb[ix]), derivation='algebraic reconstruction from saved per-mask endpoints'))
                lr=lost_lookup[arm_id(r)]; lost=set(lr['lost_ids'])
                require(len(lost)==len(lr['lost_ids']), 'Duplicate lost ID')
                require(lr['role']==role and lr['mechanism']==condition.replace('_matched_control','') and float(lr['rate'])==rate,'Lost metadata mismatch')
                n, inside, outside_n, bound=capacity(lost,base,band)
                require(n==int(r['selected_lost_count'])==lr['selected_lost_count'] and inside==lr['inside_count'] and outside_n==lr['outside_count'],'Lost counts differ')
                require(lost-band==set(lr['outside_ids']), 'Outside IDs differ')
                require(lost & far==set(lr['far_ids']['0.1']), 'Far IDs differ')
                floatcheck(n/expected_k,float(r['T']),'Lost/k equals T')
                cap_values.append((n,inside,outside_n,bound))
                capacity_masks.append(dict(study=study,condition=condition,rate=rate,role=role,assignment_id=arm_id(r),original_pair_assignment_id=r['assignment_id'],k=expected_k,band_capacity=len(base&band),total_lost=n,inside_lost=inside,outside_lost=outside_n,per_mask_capacity_lower_bound=bound))
                for pid in lost:
                    j=int(bins_for([margin[pid]])[0]);bin_counts[ix,j]+=1
            cv=np.array(cap_values,dtype=float); means=cv.mean(axis=0)
            simple=max(0.,means[0]-len(base&band))
            require(means[2]+1e-12>=means[3]>=simple-1e-12,'Mean capacity/Jensen inequalities failed')
            capacity_rows.append(dict(study=study,condition=condition,rate=rate,role=role,n_masks=200,k=expected_k,band_capacity=len(base&band),mean_total_lost=float(means[0]),mean_inside_lost=float(means[1]),mean_outside_lost=float(means[2]),mean_per_mask_capacity_lower_bound=float(means[3]),simple_mean_capacity_lower_bound=float(simple)))
            pp=[r for r in probs if r['condition']==condition and float(r['rate'])==rate]
            require(len(pp)==len({r['record_id'] for r in pp})==expected_n and {r['record_id'] for r in pp}==set(ids),'Probability cohort mismatch')
            require(all(int(r['n_masks'])==200 for r in pp),'Probability mask denominator mismatch')
            prob_by_id={r['record_id']:r for r in pp}
            recovered=0.
            for j in range(len(EDGES)-1):
                members={i for i in base if EDGES[j] <= margin[i] < EDGES[j+1]}
                den=len(members); prob=conditional(bin_counts[:,j],den)
                bin_reference=[r for r in original_bins if r['condition']==condition and float(r['rate'])==rate and r['baseline_status']=='baseline_selected' and float(r['bin_left'])==EDGES[j] and float(r['bin_right'])==EDGES[j+1]]
                require(len(bin_reference)==1 and int(bin_reference[0]['n_papers'])==den,'Original bin denominator mismatch')
                for i in members:
                    require(boolean(prob_by_id[i]['baseline_selected']), 'Unselected paper in conditional-loss denominator')
                if den:
                    selection_prob=float(np.mean([float(prob_by_id[i]['selected_probability']) for i in members]))
                    floatcheck(1-selection_prob,prob,'Per-paper probability route versus lost IDs')
                    floatcheck(1-float(bin_reference[0]['mean']),prob,'Original retention-bin route versus lost IDs')
                    recovered+=den*prob
                else:
                    require(not bin_reference[0]['mean'],'Empty original bin should be undefined')
                conditional_rows.append(dict(study=study,condition=condition,rate=rate,role=role,bin_index=j,bin_left=edge_label(EDGES[j]),bin_right=edge_label(EDGES[j+1]),bin_label=f'[{edge_label(EDGES[j])}, {edge_label(EDGES[j+1])})',baseline_selected_n=den,n_masks=200,defined_masks=200 if den else 0,total_loss_events=int(bin_counts[:,j].sum()),mean_loss_count=float(bin_counts[:,j].mean()),mean_loss_probability=prob,mean_retention_probability=None if prob is None else 1-prob,bin_rule='left closed, right open; zero belongs to [0,.01)',analysis_label='revision-stage descriptive summary; no confidence interval'))
            floatcheck(recovered,means[0],'Denominator weighted conditional losses recover total')
    require(len([r for r in shift_rows if r['role']=='base'])==13,'13 base conditions not complete')
    require(len([r for r in shift_rows if r['role']=='control'])==8,'8 controls not separate')
    shared=['original_acl/table2_aggregate.csv','original_biomedical/table2_aggregate.csv','original_acl/baseline_cutoff.json','original_biomedical/baseline_cutoff.json','cutoff_verification/baseline_membership.csv']
    out.csv('source_shifts.csv',shift_rows,shared,'policy=mixed; exact output study/condition/rate; metric columns named in formula','s0=std(b,ddof=0); C0=(mean_A(b)-mean_B(b))/s0; C=mean(group_contrast_sd); D=mean(signed_group_distortion_sd); G=mean(abs(D_m)); M=mean(mean_score_drift)/s0; w=n_A/n; A=M+(1-w)D; B=M-wD; relative=D/C0')
    out.csv('source_shifts_per_mask.csv',shift_masks,['original_acl/mask_primary_and_movement.csv','original_biomedical/mask_primary_and_movement.csv'],'condition and rate; original_pair_assignment_id equals original endpoint assignment_id; output assignment_id adds _matched_random for control arms only','A_m=M_m+(1-w)D_m; B_m=M_m-wD_m; saved mixed-policy endpoints, fixed source denominators')
    cutoff_inputs=['cutoff_verification/baseline_membership.csv','cutoff_verification/acl_lost_ids.jsonl','cutoff_verification/biomedical_lost_ids.jsonl']
    out.csv('cutoff_capacity.csv',capacity_rows,cutoff_inputs,'study/condition/rate; base or control kept separate; all 200 assignments','L=|lost|; I=|lost intersect band|; O=L-I; per-mask lower=max(0,L-|S0 intersect band|); mean across 200; simple=max(0,mean(L)-capacity)')
    out.csv('cutoff_capacity_per_mask.csv',capacity_masks,cutoff_inputs,'exact assignment_id','L=|lost|; I=|lost intersect band|; O=L-I; lower=max(0,L-|S0 intersect band|)')
    out.csv('conditional_loss_by_margin.csv',conditional_rows,cutoff_inputs+['original_acl/paper_selection_probabilities.csv','original_biomedical/paper_selection_probabilities.csv','original_acl/cutoff_margin_bins.csv','original_biomedical/cutoff_margin_bins.csv'],'study/condition/rate; baseline_selected only; bin_left<=baseline_margin<bin_right; all 200 assignments','denominator=|S0 intersect bin|; total events=sum_m|lost_m intersect bin|; mean count=total/200; loss probability=total/(200*denominator); undefined if denominator=0; retention=1-loss')
    out.csv('cutoff_nesting.csv',nesting_rows,cutoff_inputs[:1],'encoder=SPECTER; study; original top ceil(.1n); original inclusive tie-expanded band; strict margin>.1','outside=S0 minus band; far=S0 intersect {margin>.1}; explicit intersection and set differences')
    out.csv('baseline_encoder_context.csv',baseline_rows,['cutoff_verification/baseline_membership.csv','post_confirmation/specter2/baseline_reference_comparison.csv'],'exact study/encoder; source labels by exact ID from original paper_selection_probabilities','population_sd=std(scores,ddof=0); contrast=mean_A(scores)-mean_B(scores); contrast_sd=contrast/population_sd; k=ceil(.1n)')
    out.csv('cross_encoder_agreement.csv',agreement_rows,['cutoff_verification/baseline_membership.csv'],'exact study; join on paper_id; same complete-history cohort','rho=Pearson(average_ranks(SPECTER),average_ranks(SPECTER2)); Q=mean(abs(rank1-rank2))/(n-1); overlap=|topk1 intersect topk2|; overlap_fraction=overlap/k; replacement=1-overlap/k')
    out.csv('cross_encoder_source_composition.csv',composition_rows,['cutoff_verification/baseline_membership.csv','original_acl/paper_selection_probabilities.csv','original_biomedical/paper_selection_probabilities.csv'],'exact study/encoder/source; IDs joined with no drops','selected_n=count(source in encoder topk); composition=selected_n/k; within_source_rate=selected_n/source_cohort_n')
    out.csv('cross_encoder_aligned_membership.csv',aligned_rows,['cutoff_verification/baseline_membership.csv','original_acl/paper_selection_probabilities.csv','original_biomedical/paper_selection_probabilities.csv'],'exact study/paper_id and encoder; source consistent across original conditions','scores unchanged; exact-score average ranks (one-based ascending); selected=first ceil(.1n) sorted by descending score, then lexical ID')
    build_tex(output,shift_rows,capacity_rows,conditional_rows,agreement_rows,composition_rows)
    # Provenance is row-level and maps every generated scientific column to source, filter and formula.
    with (output/'claim_provenance.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,list(out.provenance[0]));writer.writeheader();writer.writerows(out.provenance)
    outputs={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name not in ('verification.json','README.md')}
    report=dict(status='VERIFIED',executed_at_utc=datetime.now(timezone.utc).isoformat(),analysis_label='R1/R2 revision-stage descriptive summaries of frozen outputs; R4 post hoc descriptive analysis',script_sha256=sha(__file__),python=sys.version,numpy=np.__version__,platform=platform.platform(),command=['python', 'analysis/revision_descriptive_outputs.py','--data-dir','02_supplement/ESM_2_contents/data/derived','--output-dir','outputs/revision_descriptive/core'],input_sha256=inputs,output_sha256=outputs,row_counts=out.row_counts,provenance_rows=len(out.provenance),original_masks_verified=4200,original_base_conditions=13,original_control_conditions=8,baseline_id_pairs=3028,max_absolute_baseline_sd_difference=errors['baseline_sd'],checks=['mixed policy only; 200 masks per condition','baseline scores/population SD/source means match original full-precision baseline table','all five source-shift identities; per-mask algebraic means agree','all original lost-ID sets, counts, inside/outside partitions, and strict margins agree','per-mask capacity and both average lower bounds','empty bins undefined; lost-ID, per-paper probability, and original bin-retention routes agree','weighted bin totals recover total losses','exact ID cohorts; average ranks and lexical topk; source composition totals'],limitations=['Source shifts algebraically reconstructed from saved full-precision endpoints; direct scalar validation is a separately executed companion check.','R4 source labels are mapped by canonical ID from the original verified source projection; encoder baseline file has no separate source-label field.','No new inferential intervals or significance tests; original family results untouched.','No fresh source acquisition or encoding.'])
    (output/'verification.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'row_counts':out.row_counts,'findings':agreement_rows,'main_cases':[r for r in shift_rows if (r['study'],r['condition'],r['rate']) in {('ACL','random',.25),('Biomedical','random',.25),('Biomedical','source',.25),('Biomedical','source',.5)}]},indent=2))


def build_tex(path, shifts, capacities, bins, agreements, compositions):
    def save(name,lines):
        (path/name).write_text('\n'.join(lines)+'\n')
    def cond(r):
        return f"{LABELS[r['condition']]} {100*r['rate']:.0f}\\%"
    selected_cases={('ACL','random',.25),('Biomedical','random',.25),('Biomedical','source',.25),('Biomedical','source',.5)}
    lines=[r'\begin{tabular}{llrrrr}',r'\toprule',r'Collection & Condition & $C_0$ & $\overline C$ & $\overline G$ & $\overline M$ \\',r'\midrule']
    for r in shifts:
        if (r['study'],r['condition'],r['rate']) in selected_cases:
            lines.append(f"{r['study']} & {cond(r)} & {r['baseline_contrast_sd']:+.3f} & {r['masked_contrast_sd']:+.3f} & {r['mean_G']:.3f} & {r['mean_drift_sd']:+.3f} \\\\")
    save('main_signed_drift.tex',lines+[r'\bottomrule',r'\end{tabular}'])
    for role in ['base','control']:
        lines=[r'\begin{tabular}{llrrrrrr}',r'\toprule',r'Collection & Condition & $\overline C$ & $\overline D$ & $\overline G$ & $\overline M$ & $\overline A$ & $\overline B$ \\',r'\midrule']
        for r in shifts:
            if r['role']==role:
                numbers=' & '.join(f'{r[k]:+.4f}' for k in ['masked_contrast_sd','signed_change_sd','mean_G','mean_drift_sd','source_A_shift_sd','source_B_shift_sd'])
                lines.append(f"{r['study']} & {cond(r)} & {numbers} \\\\")
        save(f'source_shifts_{role}.tex',lines+[r'\bottomrule',r'\end{tabular}'])
    lines=[r'\begin{tabular}{lrrrrr}',r'\toprule',r'Collection & $\rho_{\rm cross}$ & $Q_{\rm cross}$ & Overlap / $k$ & Retained & Replaced \\',r'\midrule']
    for r in agreements:
        lines.append(f"{r['study']} & {r['rho_cross']:.3f} & {r['Q_cross']:.3f} & {r['overlap_n']}/{r['k']} & {100*r['overlap_fraction']:.1f}\\% & {100*r['replacement_fraction']:.1f}\\% \\\\")
    save('cross_encoder_agreement.tex',lines+[r'\bottomrule',r'\end{tabular}'])
    lines=[r'\begin{tabular}{lllrr}',r'\toprule',r'Collection & Encoder & Source & Selected & Share of top $k$ \\',r'\midrule']
    source_names={'ACL':'ACL','EMNLP':'EMNLP','PLOSCB':'PLOS Comput. Biol.','PLOSGEN':'PLOS Genetics'}
    for r in compositions:
        lines.append(f"{r['study']} & {r['encoder']} & {source_names[r['source']]} & {r['selected_n']} & {100*r['selected_composition']:.1f}\\% \\\\")
    save('cross_encoder_composition.tex',lines+[r'\bottomrule',r'\end{tabular}'])
    lines=[r'\begin{tabular}{llrrrr}',r'\toprule',r'Collection & Condition & $\overline L$ & $\overline O$ & $\overline{(L-16)_+}$ & $(\overline L-16)_+$ \\',r'\midrule']
    for r in capacities:
        if r['role']=='base':
            numbers=' & '.join(f'{r[k]:.1f}' for k in ['mean_total_lost','mean_outside_lost','mean_per_mask_capacity_lower_bound','simple_mean_capacity_lower_bound'])
            lines.append(f"{r['study']} & {cond(r)} & {numbers} \\\\")
    save('cutoff_capacity.tex',lines+[r'\bottomrule',r'\end{tabular}'])
    lines=[r'\begin{tabular}{lrrrrr}',r'\toprule',r'Collection & $[0,.01)$ & $[.01,.025)$ & $[.025,.05)$ & $[.05,.10)$ & $[.10,\infty)$ \\',r'\midrule']
    for study in SPECS:
        rr=[r for r in bins if r['study']==study and r['condition']=='random' and r['rate']==.25 and r['bin_index']>=5]
        lines.append(study+' & '+' & '.join(str(r['baseline_selected_n']) for r in rr)+r' \\')
    save('conditional_bin_denominators.tex',lines+[r'\bottomrule',r'\end{tabular}'])
    macro=[]
    for r in shifts:
        if (r['study'],r['condition'],r['rate']) in selected_cases:
            prefix='Rev'+('Acl' if r['study']=='ACL' else 'Bio')+r['condition'].title()+('Quarter' if r['rate']==.25 else 'Half')
            for label,col in [('Contrast','masked_contrast_sd'),('Drift','mean_drift_sd'),('SourceA','source_A_shift_sd'),('SourceB','source_B_shift_sd')]:
                macro.append('\\newcommand{\\'+prefix+label+'}{'+f'{r[col]:.3f}'+'}')
    save('revision_values.tex',macro)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    run(args.data_dir,args.output_dir)
