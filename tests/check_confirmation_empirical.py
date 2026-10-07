"""Independent post-completion confirmation checks; no production-code imports.

--self-test uses fabricated arrays only. Empirical verification requires a frozen
manifest and a completed run; do not invoke it until the root authorizes it.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np

PRIMARY = {"G": "standardized_mean_absolute_group_distortion",
    "T": "mean_baseline_top_decile_replacement_fraction",
    "Q": "mean_normalized_within_year_rank_movement"}
POLICIES = ("mixed", "complete_case", "all_title", "restored")


def require(value, message):
    if not value:
        raise AssertionError(message)


def sha(path):
    state = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            state.update(chunk)
    return state.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def ranks_independent(scores):
    """Average 1-based ranks from Python sorting/grouping; ties ignore IDs."""
    groups = defaultdict(list)
    for i, value in enumerate(scores):
        groups[float(value)].append(i)
    ranks = np.empty(len(scores), dtype=float)
    start = 1
    for value in sorted(groups):
        indices = groups[value]
        ranks[indices] = start + (len(indices) - 1) / 2
        start += len(indices)
    return ranks


def order_independent(scores, ids):
    return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), str(ids[i])))


def metrics_independent(baseline, changed, ids, sources):
    b, z = np.asarray(baseline, dtype=float), np.asarray(changed, dtype=float)
    a, e = np.asarray(sources) == "ACL", np.asarray(sources) == "EMNLP"
    sd = float(np.std(b, ddof=0))
    require(sd > 0 and a.any() and e.any(), "Invalid baseline/groups")
    k = math.ceil(len(ids) / 10)
    old = set(order_independent(b, ids)[:k])
    new = set(order_independent(z, ids)[:k])
    signed = float((z[a].mean()-z[e].mean()) - (b[a].mean()-b[e].mean()))
    return {"G": abs(signed)/sd, "T": len(old-new)/k,
        "Q": float(np.abs(ranks_independent(z)-ranks_independent(b)).mean()/(len(b)-1)),
        "k": k, "lost": len(old-new)}


def margins_independent(scores, ids):
    b = np.asarray(scores, dtype=float)
    order = order_independent(b, ids)
    n, k = len(b), math.ceil(len(b)/10)
    high, low = float(b[order[k-1]]), float(b[order[k]])
    midpoint = (high+low)/2
    inside = [i for position, i in enumerate(order, start=1) if .09 <= (position-.5)/n <= .11]
    # Expand every edge-score tie intersecting the fixed ordinal percentile band.
    values = {float(b[i]) for i in inside}
    band = np.array([float(value) in values for value in b])
    return {"k": k, "gap": high-low, "midpoint": midpoint,
        "margin": (b-midpoint)/float(b.std(ddof=0)), "band": band,
        "cutoff_tied_block": int(np.sum(b == high)) if high == low else 0}


def critical_independent():
    """Quadrature of t-density; independent of readout's incomplete-beta formula."""
    df, target = 199, 1-.05/(2*45)
    nodes, weights = np.polynomial.legendre.leggauss(384)
    factor = math.exp(math.lgamma((df+1)/2)-math.lgamma(df/2)) / math.sqrt(df*math.pi)
    def cdf(t):
        points = (nodes+1)*t/2
        density = factor*np.exp(-(df+1)/2*np.log1p(points*points/df))
        return .5+t/2*float(np.dot(weights, density))
    lo, hi = 0., 8.
    for _ in range(90):
        mid = (lo+hi)/2
        if cdf(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


def scalar_score(records, full, title, fi, history_indices, missing, policy):
    vector = title[fi] if policy == "all_title" else full[fi]
    best = None
    for hi in history_indices:
        historical = records[hi]
        if historical["id"] == records[fi]["id"] or historical["year"] >= records[fi]["year"]:
            continue
        absent = historical["id"] in missing
        if policy == "complete_case" and absent:
            continue
        candidate = title[hi] if policy == "all_title" or (policy == "mixed" and absent) else full[hi]
        cosine = float(np.dot(vector, candidate))
        cosine = max(-1., min(1., cosine))
        best = cosine if best is None else max(best, cosine)
    return None if best is None else 1-best


def normalize64(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 2 and np.isfinite(values).all(), "Nonfinite/malformed embedding")
    lengths = np.sqrt(np.sum(values*values, axis=1, dtype=np.float64))
    require(bool((lengths > 0).all()), "Zero embedding")
    return values/lengths[:, None]


def self_test():
    # The top-decile denominator must be2 here, not the20-paper population.
    ids = [f"id{i:02}" for i in range(20)]
    sources = ["ACL"]*10 + ["EMNLP"]*10
    baseline = np.arange(20, dtype=float)
    moved = baseline.copy();moved[0], moved[19] = moved[19], moved[0]
    m = metrics_independent(baseline, moved, ids, sources)
    require(m["k"] == 2 and m["lost"] == 1 and m["T"] == .5, "Selection denominator check")
    require(abs(m["G"] - 3.8/baseline.std()) < 1e-14, "Absolute group distortion check")
    require(abs(m["Q"]-.1) < 1e-14, "Rank-range normalization check")
    tied = np.array([1., 1., 0., 0.])
    require(order_independent(tied, ["b", "a", "c", "d"])[0] == 1, "Lexical tie break check")
    require(np.array_equal(ranks_independent(tied), [3.5,3.5,1.5,1.5]), "Average tie-rank check")
    test_scores = -np.arange(100,dtype=float)
    context = margins_independent(test_scores, [str(i) for i in range(100)])
    require(set(np.flatnonzero(context["band"])) == {9,10}, "Ordinal midpoint percentile band")
    test_scores[8:12] = -9.5
    context = margins_independent(test_scores, [str(i) for i in range(100)])
    require(set(np.flatnonzero(context["band"])) == {8,9,10,11}, "Band boundary tie expansion")
    critical = critical_independent()
    require(abs(critical-3.3090595211511644) < 2e-11, "Independent45-family t critical")
    result = {"status": "passed", "scope": "synthetic arrays only; no empirical inputs read",
              "independent_t_critical": critical}
    print(json.dumps(result, indent=2))
    return result


def empirical(args):
    root = Path(args.root).resolve()
    results = Path(args.results).resolve() if args.results else root/"results/confirmation_v1"
    run = read_json(results/"run_manifest.json")
    # These checks occur before opening baseline scores or any embeddings.
    require(run.get("status") == "complete" and run.get("stage") == "confirmation_v1",
            "Do not calculate empirical outcomes before complete frozen confirmation")
    freeze_path = Path(args.freeze_manifest) if args.freeze_manifest else root/"configs/confirmation_freeze_v1.json"
    freeze = read_json(freeze_path)
    require(freeze.get("status") == "frozen", "Protocol is not frozen")
    provenance = run["provenance"]
    require(provenance["freeze_manifest_sha256"] == sha(freeze_path), "Freeze hash mismatch")
    paths = {"records": root/"data/confirmation_v1/records.jsonl",
        "full": root/"data/confirmation_v1/embeddings/full.npy",
        "title": root/"data/confirmation_v1/embeddings/title.npy"}
    for name, path in paths.items():
        require(sha(path) == provenance["inputs"][name], "Input checksum mismatch: "+name)
    for name, expected in run["output_sha256"].items():
        require(sha(results/name) == expected, "Result checksum mismatch: "+name)
    baseline = read_json(results/"baseline.json")
    summary = read_json(results/"summary.json")
    fingerprint = provenance["run_fingerprint"]
    require(baseline["run_fingerprint"] == summary["run_fingerprint"] == fingerprint, "Result fingerprint mismatch")
    records = [json.loads(line) for line in paths["records"].read_text().splitlines() if line.strip()]
    by_id = {row["id"]: i for i, row in enumerate(records)}
    require(len(by_id) == len(records), "Duplicate records")
    ids = baseline["scope"]["focal_ids"]
    history_ids = baseline["scope"]["history_ids"]
    sources = [records[by_id[i]]["source"] for i in ids]
    require(len(ids) == 1557 and len(history_ids) == 3799 and Counter(sources) == {"ACL":710,"EMNLP":847}, "Unexpected cohort")
    b = np.array(baseline["evaluations"]["restored"]["novelty_scores"],dtype=float)
    cutoff = margins_independent(b, ids)
    require(cutoff["k"] == 156, "Wrong selected-set denominator")
    checked_scores = 0;largest_score_error = 0.;metric_errors = {k:0. for k in PRIMARY}
    base_values, paired_values = defaultdict(list), defaultdict(list)
    saved_scalar_assignments = {"zero_mask": baseline}
    seen = set();mask_count = 0;control_count = 0
    with (results/"mask_results.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            require(row["run_fingerprint"] == fingerprint, "Raw row fingerprint mismatch")
            require(row["assignment_id"] not in seen, "Duplicate mask")
            seen.add(row["assignment_id"])
            canonical = {k:v for k,v in row.items() if k != "content_sha256"}
            require(hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest() == row["content_sha256"], "Raw content hash mismatch")
            role_metrics = {};missing_by_role = {}
            for role in ("structured_or_random", "matched_random_control"):
                if role not in row:
                    continue
                assignment = row[role];mask_count += 1
                missing = set(assignment["missing_history_ids"])
                missing_by_role[role] = missing
                require(len(missing) == len(assignment["missing_history_ids"]) == assignment["missing_count"] == math.floor(len(history_ids)*row["rate"]+.5), "Exact mask count mismatch")
                require(missing.issubset(set(history_ids)), "Mask ID outside eligible history")
                for policy in POLICIES:
                    ev = assignment["evaluations"][policy]
                    z = np.asarray(ev["novelty_scores"],dtype=float)
                    if not np.isfinite(z).all():
                        require(policy == "complete_case", "Undefined primary score")
                        continue
                    calculated = metrics_independent(b,z,ids,sources)
                    for endpoint,key in PRIMARY.items():
                        error = abs(calculated[endpoint]-ev["outcomes"][key])
                        metric_errors[endpoint] = max(metric_errors[endpoint],error)
                        require(error < (2e-5 if endpoint == "G" else 2e-12), "Saved outcome mismatch: "+endpoint)
                    if policy == "mixed":
                        role_metrics[role] = calculated
                        if role == "structured_or_random":
                            for endpoint in PRIMARY:
                                base_values[(row["rate"],row["mechanism"],endpoint)].append(calculated[endpoint])
                if row["rate"] == .25 and row["replicate"] == 0 and row["mechanism"] in ("source","topic"):
                    saved_scalar_assignments[row["mechanism"]+"/"+role] = assignment
            if row["mechanism"] != "random":
                control_count += 1
                fields = ("year",) if row["mechanism"] == "source" else ("year","source")
                require(row["control_matching"] == list(fields), "Wrong matching type")
                counts = []
                for role in ("structured_or_random","matched_random_control"):
                    counts.append(Counter(tuple(records[by_id[i]][key] for key in fields) for i in missing_by_role[role]))
                require(counts[0] == counts[1], "Actual matched cell counts differ")
                for endpoint in PRIMARY:
                    paired_values[(row["rate"],row["mechanism"],endpoint)].append(role_metrics["structured_or_random"][endpoint]-role_metrics["matched_random_control"][endpoint])
    require(len(seen)==1800 and mask_count==3000 and control_count==1200, "Incomplete confirmation design")
    require(len(base_values)==27 and len(paired_values)==18, "Wrong primary family size")
    critical = critical_independent()
    require(abs(critical-3.3090595211511644) < 2e-11, "Critical mismatch")
    # Scalar dots cover both source groups and both kinds of structured controls.
    full = normalize64(np.load(paths["full"],allow_pickle=False))
    title = normalize64(np.load(paths["title"],allow_pickle=False))
    history_indices = [by_id[i] for i in history_ids]
    focal_positions = [i for source in ("ACL","EMNLP") for i in [j for j,s in enumerate(sources) if s==source][:args.focals_per_source]]
    for assignment in saved_scalar_assignments.values():
        missing = set(assignment["missing_history_ids"])
        for policy in POLICIES:
            for position in focal_positions:
                actual = scalar_score(records,full,title,by_id[ids[position]],history_indices,missing,policy)
                expected = assignment["evaluations"][policy]["novelty_scores"][position]
                require(actual is not None and expected is not None, "Undefined scalar check")
                error = abs(actual-expected);largest_score_error=max(largest_score_error,error)
                require(error < args.atol, "Independent scalar score mismatch")
                checked_scores += 1
    report = {"status":"passed","run_fingerprint":fingerprint,"method":"standalone float64 scalar cosine and independent Python rank/group checks; no src or analysis imports",
        "scalar_scores_checked":checked_scores,"max_scalar_error":largest_score_error,
        "mask_assignments_checked":mask_count,"matched_controls_checked":control_count,
        "max_primary_metric_discrepancy":metric_errors,"primary_mean_family_size":len(base_values)+len(paired_values),
        "independent_t_critical":critical,"focal_count":len(ids),"top_decile_denominator":cutoff["k"],
        "baseline_cutoff_gap":cutoff["gap"],"cutoff_tied_block":cutoff["cutoff_tied_block"],
        "cutoff_band_size":int(cutoff["band"].sum()),"script_sha256":sha(__file__)}
    analysis = Path(args.analysis) if args.analysis else root/"results/confirmation_v1_analysis"
    if (analysis/"baseline_cutoff.json").exists():
        saved_cutoff=read_json(analysis/"baseline_cutoff.json")
        require(abs(saved_cutoff["gap"]-cutoff["gap"]) < 1e-13,"Analysis cutoff gap mismatch")
        require(saved_cutoff["rank_band_size"]==int(cutoff["band"].sum()),"Analysis percentile band mismatch")
        checks=0;largest_interval_error=0.
        for filename,paired in (("table2_aggregate.csv",False),("table3_rank_selection.csv",False),("table4_paired_primary.csv",True)):
            with (analysis/filename).open() as stream:
                for row in csv.DictReader(stream):
                    if row["metric"] not in PRIMARY or (not paired and (row["condition"] not in ("random","source","topic") or row["policy"]!="mixed")):
                        continue
                    key=(float(row["rate"]),row["mechanism"] if paired else row["condition"],row["metric"])
                    values=np.asarray((paired_values if paired else base_values)[key]);se=float(values.std(ddof=1)/math.sqrt(len(values)))
                    for field,expected in (("mean",float(values.mean())),("mcse",se),("ci_low_raw",float(values.mean()-critical*se)),("ci_high_raw",float(values.mean()+critical*se))):
                        error=abs(float(row[field])-expected);largest_interval_error=max(largest_interval_error,error)
                        require(error<2e-10,"Independent simultaneous interval mismatch: "+field)
                    checks+=1
        require(checks==45,"Analysis interval family incomplete")
        report.update(analysis_intervals_checked=checks,max_interval_discrepancy=largest_interval_error,analysis_cutoff_checked=True)
    else:
        report["analysis_interval_check"]="not yet available; rerun after analysis completes"
    if args.output:
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--results")
    parser.add_argument("--analysis")
    parser.add_argument("--freeze-manifest")
    parser.add_argument("--output")
    parser.add_argument("--atol",type=float,default=2e-6)
    parser.add_argument("--focals-per-source",type=int,default=4)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    require(args.atol>0 and args.focals_per_source>0,"Positive checking parameters required")
    self_test() if args.self_test else empirical(args)


if __name__ == "__main__":
    main()
