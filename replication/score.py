"""Independent PubMed/PLOS replication; frozen ACL source files are imported unchanged."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import signal
import sys
import time
import numpy as np
from src.masks import exact_count, make_mask
from src.measure import build_cache, score, title_neighbors
from src.pilot import (POLICIES, atomic_json, atomic_text, clean_json, evaluate_assignment,
    file_sha256, fingerprints, mc_summary, read_records, row_content_hash, run_lock,
    select_cohorts, utc_now)

STAGE = "pubmed_replication_v1"
PRIMARY_METRICS = {"G": "standardized_mean_absolute_group_distortion",
    "T": "mean_baseline_top_decile_replacement_fraction",
    "Q": "mean_normalized_within_year_rank_movement"}
CONTROL_FIELDS = {"source": ("year",)}


def assignment_specs(config):
    for rate in config["rates"]:
        for mechanism in ("random", "source"):
            for replicate in range(config["masks_per_condition"]):
                name = f"pubmed_v1_{mechanism}_r{int(round(rate * 1_000_000)):06d}_m{replicate:03d}"
                yield name, float(rate), mechanism, replicate


def create_seed_manifest(config, master_seed=2785196430,
                         namespace="pubmed-2021-replication-v1"):
    """Create an explicit pre-outcome seed artifact; never called implicitly to run."""
    if not isinstance(master_seed, int) or master_seed in (20261004, 3216741901) or master_seed < 0:
        raise ValueError("Confirmation master seed must be new and nonnegative")
    if not namespace:
        raise ValueError("Seed namespace is required")
    assignments = []
    for name, rate, mechanism, replicate in assignment_specs(config):
        item = {"assignment_id": name, "rate": rate, "mechanism": mechanism,
                "replicate": replicate}
        for role in ("mask", "control") if mechanism != "random" else ("mask",):
            payload = json.dumps([namespace, master_seed, name, role], separators=(",", ":"))
            digest = hashlib.sha256(payload.encode()).digest()
            item[role + "_seed"] = [int.from_bytes(digest[i:i+4], "big") for i in (0, 4, 8, 12)]
        assignments.append(item)
    return {"schema_version": 1, "stage": STAGE, "master_seed": master_seed,
        "namespace": namespace, "excluded_previous_seeds": [20261004, 3216741901],
        "derivation": "first128bits_SHA256_of_JSON(namespace,master_seed,assignment_id,role); four big-endian uint32 values",
        "assignments": assignments}


def validate_seed_manifest(manifest, config):
    if manifest.get("stage") != STAGE or manifest.get("schema_version") != 1:
        raise ValueError("Invalid confirmation seed manifest stage/schema")
    expected = create_seed_manifest(config, manifest.get("master_seed"), manifest.get("namespace"))
    if manifest.get("assignments") != expected["assignments"]:
        raise ValueError("Explicit seed assignments do not match the frozen configuration/derivation")
    vectors = [tuple(row[key]) for row in manifest["assignments"]
               for key in ("mask_seed", "control_seed") if key in row]
    if len(set(vectors)) != len(vectors):
        raise ValueError("Duplicate mask/control seeds")
    if manifest.get("excluded_previous_seeds") != [20261004, 3216741901]:
        raise ValueError("Pilot seed exclusion must be explicit")
    return manifest["assignments"]


def validate_config(config, synthetic_test=False):
    if config.get("stage") != STAGE:
        raise ValueError("Confirmation stage must be explicit")
    if config.get("focal_selection") != "all_eligible" or config.get("focal_sample_size") is not None:
        raise ValueError("Confirmation must retain all eligible focal papers")
    if config.get("control_matching") != {k: list(v) for k, v in CONTROL_FIELDS.items()}:
        raise ValueError("Source masks require exact historical-year matched controls")
    if config.get("history_years") != [2018, 2019, 2020] or config.get("focal_years") != [2021]:
        raise ValueError("Confirmation v1 is frozen to 2018–2020 history and 2021 focals")
    if config.get("sources") != ["PLOSCB", "PLOSGEN"] or config.get("source_contrast") != ["PLOSCB", "PLOSGEN"]:
        raise ValueError("Replication source frame/contrast must be PLOSCB versus PLOSGEN")
    if config.get("mask_preferred_source") != "PLOSCB" or config.get("structured_weight") != 8:
        raise ValueError("The pilot's outcome-independent mechanism preferences must be retained")
    if not synthetic_test and (config.get("rates") != [0.25, 0.5] or config.get("masks_per_condition") != 200):
        raise ValueError("Empirical confirmation requires both prespecified rates and exactly 200 assignments per condition")
    if config.get("masks_per_condition", 0) < 1 or not config.get("rates") or len(set(config["rates"])) != len(config["rates"]):
        raise ValueError("Invalid replicate count or repeated/empty rates")
    if any(not 0 < r < 1 for r in config["rates"]):
        raise ValueError("Stochastic rates must lie strictly between zero and one")


def select_all_cohorts(records, config):
    # Existing validated selector is reused with a nonbinding ceiling, never sampling.
    selection_config = dict(config, focal_sample_size=len(records), seed=0)
    focal, history = select_cohorts(records, selection_config)
    expected = [i for i, r in enumerate(records) if r["year"] in config["focal_years"]
                and r["source"].upper() in config["sources"]]
    if set(focal) != set(expected):
        raise AssertionError("All-focal cohort contract failed")
    return focal, history


def matched_control(history, structured_mask, rng, fields):
    if tuple(fields) not in CONTROL_FIELDS.values():
        raise ValueError("Unrecognized matching cells")
    structured_mask = np.asarray(structured_mask, dtype=bool)
    if structured_mask.shape != (len(history),):
        raise ValueError("Structured mask/history alignment mismatch")
    cells = defaultdict(list)
    for i, record in enumerate(history):
        key = tuple(int(record[field]) if field == "year" else str(record[field]).upper() for field in fields)
        cells[key].append(i)
    control = np.zeros(len(history), dtype=bool)
    metadata = []
    for key in sorted(cells):
        ix = np.asarray(cells[key], dtype=int)
        count = int(structured_mask[ix].sum())
        control[rng.choice(ix, size=count, replace=False)] = True
        metadata.append({"cell": dict(zip(fields, key)), "total": len(ix), "missing": count,
            "nondegenerate": bool(0 < count < len(ix))})
    return control, metadata


def replication_provenance(args):
    root = Path(getattr(args, "root", None) or Path(__file__).resolve().parents[1]).resolve()
    provenance = fingerprints(args)
    provenance.pop("run_fingerprint", None)
    provenance["stage"] = STAGE
    provenance["replication_code_sha256"] = {
        "replication/" + str(p.relative_to(Path(__file__).parent)): file_sha256(p)
        for p in sorted(Path(__file__).parent.rglob("*.py"))}
    provenance["inputs"].update(seeds=file_sha256(args.seed_manifest), protocol=file_sha256(args.protocol))
    provenance["inputs"].update(cohort_manifest=file_sha256(args.cohort_manifest),
        acquisition_manifest=file_sha256(args.acquisition_manifest))
    acquisition = json.loads(Path(args.acquisition_manifest).read_text())
    acquisition_code = root / "replication/acquire.py"
    rules = root / "docs/replication_cohort_rules_v1.md"
    if not acquisition_code.is_file() or acquisition.get("code_sha256") != file_sha256(acquisition_code):
        raise ValueError("Acquisition manifest does not bind the actual acquisition code")
    if not rules.is_file() or acquisition.get("rules_sha256") != file_sha256(rules):
        raise ValueError("Acquisition inclusion rules changed or are not bound")
    for name, expected in acquisition.get("data_sha256", {}).items():
        if Path(name).name != name or not (Path(args.acquisition_manifest).parent / name).is_file():
            raise ValueError("Acquisition data inventory contains an invalid or missing file")
        if file_sha256(Path(args.acquisition_manifest).parent / name) != expected:
            raise ValueError("Acquisition-derived artifact changed: " + name)
    for name, key in (("records.jsonl", "records"), ("cohort_manifest.json", "cohort_manifest")):
        if acquisition.get("data_sha256", {}).get(name) != provenance["inputs"][key]:
            raise ValueError("Acquisition manifest does not derive the supplied " + key)
    encoding_path = Path(getattr(args, "encoding_manifest", None) or Path(args.full).parent / "encoding_manifest.json")
    if not encoding_path.is_file() or not (root / "requirements.lock").is_file():
        raise ValueError("Confirmation requires encoding-manifest v2 and the actual requirements lock")
    encoding = json.loads(encoding_path.read_text())
    encoder = encoding.get("fingerprint", {})
    if encoder.get("schema") != 2 or not encoder.get("model_files"):
        raise ValueError("Encoding-manifest v2 must bind model-file hashes")
    if any(not re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in encoder["model_files"].values()):
        raise ValueError("Invalid model-file hash in encoding manifest")
    provenance["inputs"].update(encoding_manifest=file_sha256(encoding_path),
        requirements_lock=file_sha256(root / "requirements.lock"))
    if encoder.get("requirements_lock_sha256") != provenance["inputs"]["requirements_lock"]:
        raise ValueError("Actual requirements lock differs from the encoder's bound lock")
    freeze = json.loads(Path(args.freeze_manifest).read_text())
    if freeze.get("status") != "frozen" or freeze.get("stage") != STAGE:
        raise ValueError("Confirmation cannot execute before a frozen protocol manifest")
    if freeze.get("originality_gate") not in ("PASS — proceed with Candidate 1", "PASS WITH NARROWER CLAIM"):
        raise ValueError("A passing originality gate is required before confirmation")
    if not re.fullmatch(r"[0-9a-f]{40}", freeze.get("git_commit", "")) or not freeze.get("frozen_utc"):
        raise ValueError("Freeze must record a real commit identifier and timestamp")
    for name in ("records", "config", "seeds", "protocol", "full", "title", "encoding_manifest", "requirements_lock", "cohort_manifest", "acquisition_manifest"):
        if freeze.get("sha256", {}).get(name) != provenance["inputs"][name]:
            raise ValueError("Frozen artifact hash mismatch: " + name)
    if freeze.get("source_code_sha256") != provenance["source_code_sha256"]:
        raise ValueError("Source code changed after protocol freeze")
    if freeze.get("replication_code_sha256") != provenance["replication_code_sha256"]:
        raise ValueError("Replication acquisition/scoring/analysis code changed after freeze")
    analysis = freeze.get("analysis_code_sha256", {})
    if not isinstance(analysis, dict) or "replication/analysis.py" not in analysis:
        raise ValueError("Confirmation analysis code must be bound in the freeze")
    for relative, expected in analysis.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts or not relative_path.parts or relative_path.parts[0] != "replication":
            raise ValueError("Analysis bindings must be project-relative paths under replication/")
        actual = root / relative_path
        if not actual.is_file() or file_sha256(actual) != expected:
            raise ValueError("Frozen analysis file missing or changed: " + relative)
    provenance["analysis_code_sha256"] = analysis
    provenance["freeze_manifest_sha256"] = file_sha256(args.freeze_manifest)
    provenance["freeze_git_commit"] = freeze["git_commit"]
    provenance["freeze_time_utc"] = freeze["frozen_utc"]
    provenance["run_fingerprint"] = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    return provenance


def aggregate_confirmation(rows, baseline, fingerprint, config):
    grouped, paired = defaultdict(list), defaultdict(list)
    for row in rows:
        for role in ("structured_or_random", "matched_random_control"):
            if role not in row:
                continue
            condition = row["mechanism"] if role == "structured_or_random" else (
                "source_year_matched_control")
            for policy, evaluation in row[role]["evaluations"].items():
                grouped[(row["rate"], condition, policy)].append(evaluation["outcomes"])
        if "matched_random_control" in row:
            for policy in POLICIES:
                a = row["structured_or_random"]["evaluations"][policy]["outcomes"]
                b = row["matched_random_control"]["evaluations"][policy]["outcomes"]
                paired[(row["rate"], row["mechanism"], policy)].append({
                    metric: a[key] - b[key] if a[key] is not None and b[key] is not None else None
                    for metric, key in PRIMARY_METRICS.items()})
    metrics = tuple(PRIMARY_METRICS.values()) + ("mean_absolute_group_distortion",
        "mean_signed_group_distortion", "mean_score_drift", "rms_standardized_score_drift")
    return {"stage": STAGE, "run_fingerprint": fingerprint,
        "uncertainty_target": "Monte Carlo variation over complete independent masks conditional on frozen corpus/cohort; paired controls retained within unit",
        "baseline_zero_mask": baseline,
        "primary_inference": {"family_size": 18, "alpha": 0.05,
            "replicates_per_condition": config["masks_per_condition"],
            "outcomes": PRIMARY_METRICS, "policy": "mixed",
            "note": "Simultaneous intervals and mandatory cutoff-margin analysis are generated by the separately frozen analysis stage; these are raw Monte Carlo summaries."},
        "conditions": [{"rate": rate, "condition": condition, "policy": policy,
            "metrics": {key: mc_summary([r[key] for r in outcomes]) for key in metrics}}
            for (rate, condition, policy), outcomes in sorted(grouped.items())],
        "paired_primary": [{"rate": rate, "mechanism": mechanism, "policy": policy,
            "matching": list(CONTROL_FIELDS[mechanism]),
            "outcomes": {metric: mc_summary([r[metric] for r in outcomes]) for metric in PRIMARY_METRICS}}
            for (rate, mechanism, policy), outcomes in sorted(paired.items())]}


def validate_cohort_manifest(records, focal, history, args):
    cohort = json.loads(Path(args.cohort_manifest).read_text())
    if not cohort.get("quality_pass") or cohort.get("records_sha256") != file_sha256(args.records):
        raise ValueError("Replication cohort audit/records checksum mismatch")
    if not cohort.get("date_definition"):
        raise ValueError("Publication-year definition must be frozen before biomedical scoring")
    by_id = {record["id"]: record for record in records}
    for key, indices in (("focal", focal), ("history", history)):
        declared = cohort.get(key, [])
        ids = [r["id"] for r in declared]
        if len(set(ids)) != len(ids) or set(ids) != {records[i]["id"] for i in indices}:
            raise ValueError("Frozen cohort ID set mismatch: " + key)
        if cohort.get(key + "_count") != len(indices):
            raise ValueError("Frozen cohort count mismatch: " + key)
        for entry in declared:
            record = by_id[entry["id"]]
            if entry.get("year") != record["year"] or entry.get("source") != record["source"]:
                raise ValueError("Frozen cohort date/source mismatch: " + entry["id"])
            for field in ("title_sha256", "abstract_sha256"):
                if field in entry and entry[field] != record.get(field):
                    raise ValueError("Frozen cohort text hash mismatch: " + entry["id"])
    return cohort


def run(args):
    started = time.perf_counter()
    config = json.loads(Path(args.config).read_text())
    validate_config(config, getattr(args, "synthetic_test", False))
    seed_manifest = json.loads(Path(args.seed_manifest).read_text())
    assignments = validate_seed_manifest(seed_manifest, config)
    records = read_records(args.records)
    focal, history_indices = select_all_cohorts(records, config)
    cohort = validate_cohort_manifest(records, focal, history_indices, args)
    full = np.load(args.full, mmap_mode="r", allow_pickle=False)
    title = np.load(args.title, mmap_mode="r", allow_pickle=False)
    if full.shape != title.shape or len(full) != len(records):
        raise ValueError("Embedding/record alignment mismatch")
    provenance = replication_provenance(args)
    fingerprint = provenance["run_fingerprint"]
    scope = {"focal_count": len(focal), "history_count": len(history_indices),
        "focal_selection": "all_eligible", "focal_ids": [records[i]["id"] for i in focal],
        "history_ids": [records[i]["id"] for i in history_indices],
        "exact_missing_counts": {str(r): exact_count(len(history_indices), r) for r in config["rates"]},
        "base_assignments": len(assignments), "matched_controls": sum(r["mechanism"] != "random" for r in assignments),
        "control_matching": config["control_matching"]}
    scope["date_definition"] = cohort["date_definition"]
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "stage": STAGE, "scope": scope,
                          "run_fingerprint": fingerprint}, indent=2))
        return {"status": "dry_run", "stage": STAGE, "scope": scope, "provenance": provenance}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        manifest_path = output / "run_manifest.json"
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text())
            if old["provenance"]["run_fingerprint"] != fingerprint:
                raise ValueError("Resume rejected: frozen confirmation fingerprint changed")
            if not args.resume:
                raise ValueError("Existing confirmation output requires --resume")
            if old["status"] == "complete":
                for name, expected in old["output_sha256"].items():
                    if file_sha256(output / name) != expected:
                        raise ValueError("Completed output checksum mismatch: " + name)
                return old
        manifest = {"status": "running", "stage": STAGE, "started_utc": utc_now(),
            "config": config, "scope": scope, "provenance": provenance}
        atomic_json(manifest_path, manifest)
        raw_dir = output / "raw"
        raw_dir.mkdir(exist_ok=True)
        try:
            cache = build_cache(records, full, title, focal, history_indices,
                cache_dir=output / "similarity_cache", block_size=config.get("block_size", 128))
            if not cache.eligible.any(axis=1).all():
                raise ValueError("Confirmation focal has empty eligible history")
            baseline = score(cache, np.zeros(len(history_indices), dtype=bool), "restored")
            if not np.isfinite(baseline).all() or np.std(baseline) <= 0:
                raise ValueError("Undefined or zero-variance confirmation baseline")
            neighbors = title_neighbors(cache, config.get("top_k", 20))
            history = [records[i] for i in history_indices]
            baseline_row = clean_json(evaluate_assignment(records, cache, baseline,
                np.zeros(len(history_indices), dtype=bool), neighbors, config, "zero_mask"))
            atomic_json(output / "baseline.json", {"stage": STAGE,
                "run_fingerprint": fingerprint, "scope": scope, **baseline_row})
            rows, computed = [], 0
            for seed_row in assignments:
                name, rate, mechanism = (seed_row[k] for k in ("assignment_id", "rate", "mechanism"))
                shard = raw_dir / (name + ".json")
                if shard.exists():
                    row = json.loads(shard.read_text())
                    if row.get("run_fingerprint") != fingerprint or row.get("assignment_id") != name or row.get("content_sha256") != row_content_hash(row):
                        raise ValueError("Confirmation shard fingerprint/content mismatch")
                    rows.append(row)
                    continue
                if args.max_masks is not None and computed >= args.max_masks:
                    manifest.update(status="partial", completed_assignments=len(rows),
                        checkpoint_utc=utc_now(), elapsed_seconds=time.perf_counter()-started)
                    atomic_json(manifest_path, manifest)
                    return manifest
                missing = make_mask(history, rate, mechanism,
                    np.random.default_rng(np.random.SeedSequence(seed_row["mask_seed"])),
                    preferred_source=config["mask_preferred_source"], preferred_topic=config.get("mask_preferred_topic", 0),
                    weight=config["structured_weight"], topic_buckets=config.get("topic_buckets", 8))
                row = {"stage": STAGE, "run_fingerprint": fingerprint, **seed_row,
                    "structured_or_random": evaluate_assignment(records, cache, baseline,
                        missing, neighbors, config, name)}
                if mechanism != "random":
                    control, cells = matched_control(history, missing,
                        np.random.default_rng(np.random.SeedSequence(seed_row["control_seed"])),
                        CONTROL_FIELDS[mechanism])
                    row["control_matching"] = list(CONTROL_FIELDS[mechanism])
                    row["control_cells"] = cells
                    row["matched_random_control"] = evaluate_assignment(records, cache,
                        baseline, control, neighbors, config, name + "_matched_random")
                    row["paired_primary_differences"] = {}
                    for metric, key in PRIMARY_METRICS.items():
                        a = row["structured_or_random"]["evaluations"]["mixed"]["outcomes"][key]
                        b = row["matched_random_control"]["evaluations"]["mixed"]["outcomes"][key]
                        row["paired_primary_differences"][metric] = a-b if a is not None and b is not None else None
                row["content_sha256"] = row_content_hash(row)
                atomic_json(shard, row)
                rows.append(clean_json(row))
                computed += 1
                manifest.update(completed_assignments=len(rows), checkpoint_utc=utc_now())
                atomic_json(manifest_path, manifest)
            atomic_text(output / "mask_results.jsonl", "".join(json.dumps(row, sort_keys=True, allow_nan=False)+"\n" for row in rows))
            atomic_json(output / "summary.json", aggregate_confirmation(rows, baseline_row, fingerprint, config))
            manifest.update(status="complete", finished_utc=utc_now(), completed_assignments=len(rows),
                elapsed_seconds=time.perf_counter()-started,
                output_sha256={name: file_sha256(output / name) for name in ("baseline.json", "mask_results.jsonl", "summary.json")})
            atomic_json(manifest_path, manifest)
            return manifest
        except BaseException as exc:
            manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                checkpoint_utc=utc_now(), error_type=type(exc).__name__, error=str(exc),
                elapsed_seconds=time.perf_counter()-started)
            atomic_json(manifest_path, manifest)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("records", "full", "title", "config", "seed-manifest", "freeze-manifest", "protocol", "cohort-manifest", "acquisition-manifest", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--encoding-manifest")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]),
                        help="Project root containing the actual requirements lock and frozen analysis files")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--synthetic-test", action="store_true", help="Only synthetic tests may reduce replication and omit an encoding manifest")
    parser.add_argument("--max-masks", type=int)
    args = parser.parse_args()
    if args.max_masks is not None and args.max_masks < 1:
        parser.error("--max-masks must be positive")
    def stop(signum, frame):
        raise KeyboardInterrupt("Termination received; committed confirmation shards are resumable")
    signal.signal(signal.SIGTERM, stop)
    try:
        result = run(args)
        print(json.dumps({"stage": STAGE, "status": result["status"],
            "completed_assignments": result.get("completed_assignments"), "output": args.output}))
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
