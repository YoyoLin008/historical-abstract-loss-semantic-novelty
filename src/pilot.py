"""Resumable exploratory Candidate 1 experiment with exact cached similarities."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import socket
import sys
import tempfile
import time
import numpy as np
from .masks import make_mask, mask_rng, year_matched_control, exact_count
from .measure import build_cache, score, summarize_scores, title_neighbors, local_exposure

POLICIES = ("mixed", "complete_case", "all_title", "restored")
SCHEMA_VERSION = 1


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_json(value):
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(x) for x in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    atomic_text(path, json.dumps(clean_json(value), indent=2, sort_keys=True,
                                 allow_nan=False) + "\n")


@contextmanager
def run_lock(output):
    """One writer per output; reclaim only a demonstrably dead same-host process."""
    path = Path(output) / ".run.lock"
    body = {"pid": os.getpid(), "host": socket.gethostname(), "created_utc": utc_now()}
    for attempt in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "w") as stream:
                json.dump(body, stream)
            break
        except FileExistsError:
            try:
                previous = json.loads(path.read_text())
                if previous.get("host") != socket.gethostname():
                    raise RuntimeError("Output is locked by another host; inspect before resuming")
                os.kill(int(previous["pid"]), 0)
            except ProcessLookupError:
                path.unlink()
                continue
            except (ValueError, KeyError, json.JSONDecodeError):
                raise RuntimeError("Malformed lock requires inspection: " + str(path))
            raise RuntimeError("An active process already owns output: " + str(path))
    else:
        raise RuntimeError("Unable to obtain output lock")
    try:
        yield
    finally:
        if path.exists():
            path.unlink()


def read_records(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError("Empty records file")
    ids = [r.get("id") for r in records]
    if any(not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("Each record must have a unique nonempty identifier")
    for r in records:
        for key in ("year", "source", "title", "abstract", "topic"):
            if key not in r:
                raise ValueError("Missing required record field: " + key)
        if not str(r["title"]).strip() or not str(r["abstract"]).strip():
            raise ValueError("Experimental records require complete authentic title/abstract")
        if not isinstance(r["year"], int) or not isinstance(r["topic"], int):
            raise ValueError("year and preassigned lexical-proxy topic must be integers")
    return records


def select_cohorts(records, config):
    sources = {s.upper() for s in config["sources"]}
    histories = [i for i, r in enumerate(records)
        if r["year"] in config["history_years"] and r["source"].upper() in sources]
    focals = [i for i, r in enumerate(records)
        if r["year"] in config["focal_years"] and r["source"].upper() in sources]
    histories.sort(key=lambda i: records[i]["id"])
    focals.sort(key=lambda i: records[i]["id"])
    if not histories or not focals:
        raise ValueError("Configured historical/focal collection is empty")
    if set(config["history_years"]) & set(config["focal_years"]):
        raise ValueError("This pilot requires disjoint historical and focal years")
    sample_size = config.get("focal_sample_size", len(focals))
    if sample_size <= 0:
        raise ValueError("Focal sample size must be positive")
    if len(focals) > sample_size:
        rng = np.random.default_rng(config["seed"])
        focals = sorted(rng.choice(focals, size=sample_size, replace=False).tolist(),
                        key=lambda i: records[i]["id"])
    for year in config["focal_years"]:
        for source in config["source_contrast"]:
            if not any(records[i]["year"] == year and records[i]["source"].upper() == source.upper() for i in focals):
                raise ValueError(f"Sample lacks predefined contrast group {year}/{source}")
    return np.asarray(focals), np.asarray(histories)


def fingerprints(args):
    source_dir = Path(__file__).parent
    code = {str(p.relative_to(source_dir)): file_sha256(p)
            for p in sorted(source_dir.rglob("*.py"))}
    payload = {"schema_version": SCHEMA_VERSION,
        "inputs": {name: file_sha256(getattr(args, name)) for name in ("records", "full", "title", "config")},
        "source_code_sha256": code,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__}}
    payload["encoding_provenance"] = validate_encoding_manifest(args, payload["inputs"])
    payload["run_fingerprint"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def validate_encoding_manifest(args, input_hashes):
    path = Path(getattr(args, "encoding_manifest", None) or Path(args.full).parent / "encoding_manifest.json")
    if not path.exists():
        if getattr(args, "synthetic_test", False):
            return {"status": "unverified_synthetic_test", "model": None, "revision": None}
        raise ValueError("Encoding manifest required; cannot verify embedding/record alignment: " + str(path))
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Encoding manifest is not complete")
    fingerprint = manifest.get("fingerprint", {})
    if fingerprint.get("data_sha256") != input_hashes["records"]:
        raise ValueError("Encoding manifest records hash/order does not match supplied records")
    for representation in ("full", "title"):
        if manifest.get("output_sha256", {}).get(representation) != input_hashes[representation]:
            raise ValueError("Encoding manifest output hash mismatch: " + representation)
    if not fingerprint.get("model") or not fingerprint.get("revision"):
        raise ValueError("Encoding model and exact revision must be recorded")
    return {"status": "verified", "manifest_sha256": file_sha256(path),
        "model": fingerprint["model"], "revision": fingerprint["revision"],
        "encoder_fingerprint": fingerprint}


def row_content_hash(row):
    payload = {key: value for key, value in row.items() if key != "content_sha256"}
    return hashlib.sha256(json.dumps(clean_json(payload), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def assignment_specs(config):
    for rate in config["rates"]:
        for mechanism in ("random", "source", "topic"):
            for replicate in range(config["masks_per_condition"]):
                name = f"{mechanism}_r{int(round(rate * 1_000_000)):06d}_m{replicate:03d}"
                yield name, float(rate), mechanism, replicate


def composition_counts(history, missing):
    counts = {}
    for field in ("year", "source", "topic"):
        values = {}
        for r, absent in zip(history, missing):
            key = str(r[field])
            if key not in values:
                values[key] = {"total": 0, "missing": 0}
            values[key]["total"] += 1
            values[key]["missing"] += int(absent)
        counts[field] = values
    return counts


def evaluate_assignment(records, cache, baseline, missing, neighbors, config, name):
    history = [records[i] for i in cache.history_indices]
    eligible_rate, local_rate = local_exposure(cache, missing, neighbors)
    evaluations = {}
    for policy in POLICIES:
        scores = score(cache, missing, policy)
        evaluations[policy] = {
            "outcomes": summarize_scores(records, cache.focal_indices, baseline,
                scores, tuple(config["source_contrast"])),
            "novelty_scores": scores,
        }
    return {"assignment_id": name, "missing_count": int(missing.sum()),
        "realized_global_missing_rate": float(missing.mean()),
        "missing_history_ids": [records[i]["id"] for i in cache.history_indices[missing]],
        "composition": composition_counts(history, missing),
        "deployment_available_features": {
            "eligible_history_missing_fraction": eligible_rate,
            "title_neighborhood_missing_fraction": local_rate,
        },
        "evaluations": evaluations}


def mc_summary(values):
    values = [x for x in values if x is not None]
    n = len(values)
    sd = float(np.std(values, ddof=1)) if n > 1 else None
    return {"n_masks": n, "mean": float(np.mean(values)) if n else None,
        "sd_over_masks": sd, "monte_carlo_se_mean": sd / np.sqrt(n) if sd is not None else None,
        "minimum": float(min(values)) if n else None, "maximum": float(max(values)) if n else None}


def aggregate_results(rows, baseline_row, run_fingerprint):
    grouped = defaultdict(list)
    paired = defaultdict(list)
    for row in rows:
        for role in ("structured_or_random", "year_matched_random_control"):
            if role not in row:
                continue
            condition = row["mechanism"] if role == "structured_or_random" else row["mechanism"] + "_year_matched_control"
            assignment = row[role]
            for policy, evaluation in assignment["evaluations"].items():
                grouped[(row["rate"], condition, policy)].append(evaluation["outcomes"])
        if "paired_excess_absolute_group_distortion" in row:
            for policy, difference in row["paired_excess_absolute_group_distortion"].items():
                paired[(row["rate"], row["mechanism"], policy)].append(difference)
    metrics = ("mean_absolute_group_distortion", "standardized_mean_absolute_group_distortion",
        "mean_signed_group_distortion", "mean_score_drift", "rms_standardized_score_drift",
        "mean_normalized_within_year_rank_movement", "mean_baseline_top_decile_replacement_fraction")
    return {"run_fingerprint": run_fingerprint, "stage": "exploratory_pilot",
        "uncertainty_target": "Monte Carlo variability over complete mask assignments conditional on this frozen corpus and focal sample; not population generalization",
        "baseline_zero_mask": baseline_row,
        "conditions": [{"rate": rate, "condition": condition, "policy": policy,
            "metrics": {metric: mc_summary([x[metric] for x in outcomes]) for metric in metrics}}
            for (rate, condition, policy), outcomes in sorted(grouped.items())],
        "paired_primary": [{"rate": rate, "mechanism": mechanism, "policy": policy,
            "excess_absolute_group_distortion": mc_summary(values)}
            for (rate, mechanism, policy), values in sorted(paired.items())]}


def run(args):
    started = time.perf_counter()
    config = json.loads(Path(args.config).read_text())
    if len(set(config["rates"])) != len(config["rates"]) or any(not 0 < r < 1 for r in config["rates"]):
        raise ValueError("Pilot rates must be unique values strictly between zero and one")
    if config["masks_per_condition"] < 1:
        raise ValueError("At least one mask is required")
    records = read_records(args.records)
    focal, history_indices = select_cohorts(records, config)
    full = np.load(args.full, mmap_mode="r", allow_pickle=False)
    title = np.load(args.title, mmap_mode="r", allow_pickle=False)
    if full.shape != title.shape or len(full) != len(records):
        raise ValueError("Embeddings do not align with the supplied records")
    scope = {"history_count": len(history_indices), "focal_count": len(focal),
        "focal_ids": [records[i]["id"] for i in focal],
        "history_ids": [records[i]["id"] for i in history_indices],
        "exact_missing_counts": {str(rate): exact_count(len(history_indices), rate) for rate in config["rates"]},
        "assignment_pairs_or_random": len(list(assignment_specs(config))),
        "policy_count": len(POLICIES)}
    provenance = fingerprints(args)
    if args.dry_run:
        print(json.dumps(scope, indent=2))
        return {"status": "dry_run", "scope": scope, "provenance": provenance}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = provenance["run_fingerprint"]
    with run_lock(output):
        manifest_path = output / "run_manifest.json"
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text())
            if old["provenance"]["run_fingerprint"] != fingerprint:
                raise ValueError("Resume rejected: input/config/code/runtime fingerprint changed; use a new output directory")
            if not args.resume:
                raise ValueError("Output already has a run; use --resume or a new output directory")
            if old["status"] == "complete":
                for name, expected in old["output_sha256"].items():
                    if file_sha256(output / name) != expected:
                        raise ValueError("Completed output checksum mismatch: " + name)
                print(json.dumps({"status": "already_complete", "output": str(output), "run_fingerprint": fingerprint}))
                return old
        manifest = {"status": "running", "stage": config["stage"], "started_utc": utc_now(),
            "provenance": provenance, "scope": scope, "config": config,
            "pid": os.getpid(), "host": socket.gethostname()}
        atomic_json(manifest_path, manifest)
        raw_dir = output / "raw"
        raw_dir.mkdir(exist_ok=True)
        try:
            cache_started = time.perf_counter()
            cache = build_cache(records, full, title, focal, history_indices,
                cache_dir=output / "similarity_cache", block_size=config.get("block_size", 128))
            if not cache.eligible.any(axis=1).all():
                raise ValueError("Frozen focal cohort contains papers with empty eligible histories")
            baseline = score(cache, np.zeros(len(history_indices), dtype=bool), "restored")
            neighbors = title_neighbors(cache, config.get("top_k", 20))
            history = [records[i] for i in history_indices]
            baseline_row = evaluate_assignment(records, cache, baseline,
                np.zeros(len(history_indices), dtype=bool), neighbors, config, "zero_mask")
            atomic_json(output / "baseline.json", {"run_fingerprint": fingerprint,
                "scope": scope, **baseline_row})
            cache_seconds = time.perf_counter() - cache_started
            rows, newly_computed = [], 0
            for name, rate, mechanism, replicate in assignment_specs(config):
                shard = raw_dir / (name + ".json")
                if shard.exists():
                    row = json.loads(shard.read_text())
                    if row["run_fingerprint"] != fingerprint or row["assignment_id"] != name:
                        raise ValueError("Corrupt or mismatched result shard: " + str(shard))
                    if row.get("content_sha256") != row_content_hash(row):
                        raise ValueError("Result shard content checksum mismatch: " + str(shard))
                    rows.append(row)
                    continue
                if args.max_masks is not None and newly_computed >= args.max_masks:
                    manifest.update(status="partial", completed_assignments=len(rows),
                        checkpoint_utc=utc_now(), elapsed_seconds=time.perf_counter() - started)
                    atomic_json(manifest_path, manifest)
                    print(json.dumps({"status": "partial", "completed_assignments": len(rows), "output": str(output)}))
                    return manifest
                missing = make_mask(history, rate, mechanism,
                    mask_rng(config["seed"], rate, mechanism, replicate),
                    preferred_source=config["mask_preferred_source"],
                    preferred_topic=config["mask_preferred_topic"],
                    weight=config["structured_weight"], topic_buckets=config["topic_buckets"])
                row = {"run_fingerprint": fingerprint, "assignment_id": name,
                    "rate": rate, "mechanism": mechanism, "replicate": replicate,
                    "structured_or_random": evaluate_assignment(records, cache, baseline,
                        missing, neighbors, config, name)}
                if mechanism != "random":
                    control = year_matched_control(history, missing,
                        mask_rng(config["seed"], rate, mechanism, replicate, control=True))
                    row["year_matched_random_control"] = evaluate_assignment(records, cache,
                        baseline, control, neighbors, config, name + "_year_matched_random")
                    row["paired_excess_absolute_group_distortion"] = {}
                    for policy in POLICIES:
                        a = row["structured_or_random"]["evaluations"][policy]["outcomes"]["mean_absolute_group_distortion"]
                        b = row["year_matched_random_control"]["evaluations"][policy]["outcomes"]["mean_absolute_group_distortion"]
                        row["paired_excess_absolute_group_distortion"][policy] = a - b if a is not None and b is not None else None
                row["content_sha256"] = row_content_hash(row)
                atomic_json(shard, row)
                rows.append(clean_json(row))
                newly_computed += 1
                manifest.update(completed_assignments=len(rows), checkpoint_utc=utc_now())
                atomic_json(manifest_path, manifest)
            atomic_text(output / "mask_results.jsonl", "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))
            atomic_json(output / "summary.json", aggregate_results(rows, clean_json(baseline_row), fingerprint))
            manifest.update(status="complete", finished_utc=utc_now(),
                completed_assignments=len(rows), cache_seconds=cache_seconds,
                elapsed_seconds=time.perf_counter() - started,
                output_sha256={name: file_sha256(output / name)
                    for name in ("baseline.json", "mask_results.jsonl", "summary.json")})
            atomic_json(manifest_path, manifest)
            print(json.dumps({"status": "complete", "output": str(output), "run_fingerprint": fingerprint,
                "completed_assignments": len(rows), "elapsed_seconds": manifest["elapsed_seconds"]}))
            return manifest
        except BaseException as exc:
            manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                checkpoint_utc=utc_now(), error_type=type(exc).__name__, error=str(exc),
                elapsed_seconds=time.perf_counter() - started)
            atomic_json(manifest_path, manifest)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--full", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--config", default="configs/pilot.json")
    parser.add_argument("--output", default="results/pilot")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--encoding-manifest", help="Default: encoding_manifest.json beside full.npy")
    parser.add_argument("--synthetic-test", action="store_true", help="Allow absent encoding manifest only for synthetic implementation tests; never scientific results")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-masks", type=int, help="Bound this invocation; resume remaining assignments later")
    args = parser.parse_args()
    if args.max_masks is not None and args.max_masks < 1:
        parser.error("--max-masks must be positive")
    def stop(signum, frame):
        raise KeyboardInterrupt("Received termination signal; committed shards are resumable")
    signal.signal(signal.SIGTERM, stop)
    try:
        run(args)
    except KeyboardInterrupt:
        print("Interrupted; resume with the same command plus --resume", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
