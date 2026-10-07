"""Independent float64 scalar-dot verification of saved empirical pilot results.

This intentionally imports no project measurement or masking implementation.
It verifies only selected saved scores, not the scientific validity of an encoder.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("mixed", "complete_case", "all_title", "restored")


def digest(path):
    state = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            state.update(chunk)
    return state.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def normalize_independently(array):
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Nonfinite or malformed embeddings")
    # Accumulation in float64 is independent of production float32 normalization.
    lengths = np.sqrt(np.sum(array * array, axis=1, dtype=np.float64))
    if (lengths <= 0).any():
        raise ValueError("Zero embedding encountered")
    return array / lengths[:, None]


def direct_one(records, full, title, focal_index, history_indices, missing_ids, policy):
    focal_vector = title[focal_index] if policy == "all_title" else full[focal_index]
    best = None
    for historical_index in history_indices:
        record = records[historical_index]
        if record["id"] == records[focal_index]["id"]:
            continue
        if int(record["year"]) >= int(records[focal_index]["year"]):
            continue
        missing = record["id"] in missing_ids
        if policy == "complete_case" and missing:
            continue
        vector = title[historical_index] if policy == "all_title" or (policy == "mixed" and missing) else full[historical_index]
        # Each pair is computed as a scalar dot product; no production cache used.
        similarity = min(1.0, max(-1.0, float(np.dot(focal_vector, vector))))
        if best is None or similarity > best:
            best = similarity
    return None if best is None else 1.0 - best


def verify(args):
    project = Path(args.project).resolve()
    output = Path(args.pilot).resolve() if args.pilot else project / "results/pilot"
    manifest = read_json(output / "run_manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("Empirical verification requires a completed pilot")
    fingerprint = manifest["provenance"]["run_fingerprint"]
    paths = {"records": project / "data/processed/records.jsonl",
        "full": project / "data/embeddings/full.npy", "title": project / "data/embeddings/title.npy"}
    for name, path in paths.items():
        if digest(path) != manifest["provenance"]["inputs"][name]:
            raise ValueError("Input provenance mismatch: " + name)
    for name, expected in manifest["output_sha256"].items():
        if digest(output / name) != expected:
            raise ValueError("Saved output integrity mismatch: " + name)
    baseline = read_json(output / "baseline.json")
    if baseline["run_fingerprint"] != fingerprint:
        raise ValueError("Baseline belongs to another run")
    shard = Path(args.assignment) if args.assignment else output / "raw/source_r250000_m000.json"
    if not shard.exists():
        choices = sorted((output / "raw").glob("*.json"))
        if not choices:
            raise ValueError("No saved masks found")
        shard = choices[0]
    row = read_json(shard)
    if row["run_fingerprint"] != fingerprint:
        raise ValueError("Saved mask belongs to another run")
    content = {k: v for k, v in row.items() if k != "content_sha256"}
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if hashlib.sha256(canonical.encode()).hexdigest() != row.get("content_sha256"):
        raise ValueError("Raw mask content checksum mismatch")
    records = [json.loads(line) for line in paths["records"].read_text().splitlines() if line.strip()]
    index = {r["id"]: i for i, r in enumerate(records)}
    if len(index) != len(records):
        raise ValueError("Duplicate IDs in empirical inputs")
    full = normalize_independently(np.load(paths["full"], allow_pickle=False))
    title = normalize_independently(np.load(paths["title"], allow_pickle=False))
    if full.shape != title.shape or len(full) != len(records):
        raise ValueError("Embedding shape mismatch")
    focal_ids = baseline["scope"]["focal_ids"][:args.focals]
    history_ids = baseline["scope"]["history_ids"]
    history_indices = [index[record_id] for record_id in history_ids]
    checks = []
    assignments = {"zero_mask": baseline, "structured_or_random": row["structured_or_random"]}
    if "year_matched_random_control" in row:
        assignments["year_matched_random_control"] = row["year_matched_random_control"]
        a = set(row["structured_or_random"]["missing_history_ids"])
        b = set(row["year_matched_random_control"]["missing_history_ids"])
        for year in sorted({int(records[i]["year"]) for i in history_indices}):
            group = {records[i]["id"] for i in history_indices if int(records[i]["year"]) == year}
            if len(a & group) != len(b & group):
                raise AssertionError("Matched control failed exact year counts")
    for name, assignment in assignments.items():
        missing_ids = set(assignment["missing_history_ids"])
        if not missing_ids.issubset(set(history_ids)):
            raise ValueError("Mask contains a record outside the declared history")
        for policy in POLICIES:
            saved = assignment["evaluations"][policy]["novelty_scores"][:len(focal_ids)]
            calculated = [direct_one(records, full, title, index[record_id],
                history_indices, missing_ids, policy) for record_id in focal_ids]
            errors = []
            for identifier, expected, actual in zip(focal_ids, calculated, saved):
                if expected is None or actual is None:
                    if expected != actual:
                        raise AssertionError(f"Undefinedness mismatch: {name}/{policy}/{identifier}")
                else:
                    error = abs(expected - actual)
                    if error > args.atol:
                        raise AssertionError(f"Score mismatch {name}/{policy}/{identifier}: scalar64={expected}, saved={actual}, error={error}")
                    errors.append(error)
            checks.append({"assignment": name, "policy": policy,
                "n_focals": len(focal_ids), "max_absolute_error": max(errors) if errors else None})
    report = {"status": "passed", "method": "independent float64 scalar cosine dots with strict prior-year eligibility; no src imports or similarity caches",
        "run_fingerprint": fingerprint, "assignment_id": row["assignment_id"],
        "focal_ids": focal_ids, "atol": args.atol, "checks": checks,
        "script_sha256": digest(__file__)}
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(ROOT))
    parser.add_argument("--pilot")
    parser.add_argument("--assignment")
    parser.add_argument("--focals", type=int, default=7)
    parser.add_argument("--atol", type=float, default=2e-6)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.focals < 1 or args.atol <= 0:
        parser.error("--focals and --atol must be positive")
    verify(args)


if __name__ == "__main__":
    main()
