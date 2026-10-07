"""Exact nearest-prior measurement; no approximate search or learned outcomes."""
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np


def unit_rows(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Embeddings must be a finite two-dimensional array")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("Zero-length or numerically overflowing embeddings are undefined")
    return values / norms


@dataclass
class SimilarityCache:
    full_full: np.ndarray
    full_title: np.ndarray
    title_title: np.ndarray
    eligible: np.ndarray
    focal_indices: np.ndarray
    history_indices: np.ndarray


def build_cache(records, full, title, focal_indices, history_indices, cache_dir=None,
                block_size=128):
    """Normalize once; bound each exact matrix multiplication by focal blocks.

    Arrays have one row per *frozen* focal and one column per historical record.
    Invalid temporal/self comparisons are excluded before all neighbor selections.
    """
    ids = [str(r["id"]) for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate record IDs could create duplicate/self comparisons")
    f = np.asarray(focal_indices, dtype=int)
    h = np.asarray(history_indices, dtype=int)
    if not len(f) or not len(h):
        raise ValueError("A nonempty focal set and historical collection are required")
    if len(set(f)) != len(f) or len(set(h)) != len(h):
        raise ValueError("Repeated focal/history indices are not allowed")
    if min(f.min(), h.min()) < 0 or max(f.max(), h.max()) >= len(records):
        raise ValueError("Record index outside the frozen collection")
    full, title = unit_rows(full), unit_rows(title)
    if full.shape != title.shape or len(full) != len(records):
        raise ValueError("Representations must align exactly with records")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    fy = np.asarray([int(records[i]["year"]) for i in f])
    hy = np.asarray([int(records[i]["year"]) for i in h])
    eligible = (hy[None, :] < fy[:, None]) & (f[:, None] != h[None, :])
    arrays = []
    for name, left, right in (("full_full", full, full),
                              ("full_title", full, title),
                              ("title_title", title, title)):
        if cache_dir is None:
            matrix = np.empty((len(f), len(h)), dtype=np.float32)
        else:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            matrix = np.lib.format.open_memmap(Path(cache_dir) / (name + ".npy"),
                mode="w+", dtype=np.float32, shape=(len(f), len(h)))
        for start in range(0, len(f), block_size):
            stop = min(start + block_size, len(f))
            block = np.clip(left[f[start:stop]] @ right[h].T, -1.0, 1.0)
            block[~eligible[start:stop]] = -np.inf
            matrix[start:stop] = block
        if hasattr(matrix, "flush"):
            matrix.flush()
        arrays.append(matrix)
    return SimilarityCache(*arrays, eligible, f, h)


def score(cache, missing, policy="mixed"):
    """Return 1−maximum cosine; empty eligible subsets give NaN, never defaults."""
    missing = np.asarray(missing, dtype=bool)
    if missing.shape != (len(cache.history_indices),):
        raise ValueError("Missingness mask must align with historical records")
    if policy == "mixed":
        sims = np.where(missing[None, :], cache.full_title, cache.full_full)
    elif policy == "complete_case":
        sims = np.where(missing[None, :], -np.inf, cache.full_full)
    elif policy == "all_title":
        sims = cache.title_title
    elif policy == "restored":
        sims = cache.full_full
    else:
        raise ValueError("Unknown representation policy: " + policy)
    maxima = np.max(sims, axis=1)
    return np.where(np.isfinite(maxima), 1.0 - maxima, np.nan)


def direct_score(records, full, title, focal_indices, history_indices, missing,
                 policy="mixed"):
    """Simple independent loop implementation used as a numerical oracle."""
    out = []
    for fi in focal_indices:
        similarities = []
        focal = title[fi] if policy == "all_title" else full[fi]
        for j, hi in enumerate(history_indices):
            if fi == hi or records[hi]["year"] >= records[fi]["year"]:
                continue
            if policy == "complete_case" and missing[j]:
                continue
            use_title = policy == "all_title" or (policy == "mixed" and missing[j])
            historical = title[hi] if use_title else full[hi]
            similarity = float(np.dot(np.asarray(focal, dtype=np.float64), historical)
                / (np.linalg.norm(focal) * np.linalg.norm(historical)))
            similarities.append(min(1.0, max(-1.0, similarity)))
        out.append(1.0 - max(similarities) if similarities else float("nan"))
    return np.asarray(out)


def title_neighbors(cache, k=20):
    """Deployment-available title/title neighbors; input order breaks ties."""
    if k < 1:
        raise ValueError("k must be positive")
    return [np.flatnonzero(cache.eligible[i])[
        np.argsort(-cache.title_title[i, cache.eligible[i]], kind="stable")[:k]]
        for i in range(len(cache.focal_indices))]


def local_exposure(cache, missing, neighbors):
    missing = np.asarray(missing, dtype=bool)
    eligible_count = cache.eligible.sum(axis=1)
    eligible_missing = (cache.eligible & missing[None, :]).sum(axis=1)
    global_rates = np.divide(eligible_missing, eligible_count,
        out=np.full(len(eligible_count), np.nan), where=eligible_count > 0)
    local = np.asarray([float(missing[ix].mean()) if len(ix) else np.nan
                        for ix in neighbors])
    return global_rates, local


def average_ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def summarize_scores(records, focal_indices, baseline, scores,
                     source_contrast=("ACL", "EMNLP")):
    """Finite-corpus outcomes; never treat individual documents as mask replicates.

    Contrasts and ranking outcomes are computed within year, then equally weighted
    over the predeclared years. An undefined score makes that cohort outcome
    undefined rather than changing the focal composition silently.
    """
    baseline, scores = np.asarray(baseline), np.asarray(scores)
    years = np.asarray([int(records[i]["year"]) for i in focal_indices])
    sources = np.asarray([str(records[i]["source"]).upper() for i in focal_indices])
    ids = np.asarray([str(records[i]["id"]) for i in focal_indices])
    valid = np.isfinite(baseline) & np.isfinite(scores)
    base_valid = np.isfinite(baseline)
    sd = float(np.std(baseline[base_valid], ddof=0)) if base_valid.any() else np.nan
    sd_ok = sd > 0 and math.isfinite(sd)
    contrasts, ranks, replacements = [], [], []
    for year in sorted(set(years)):
        ix = np.flatnonzero(years == year)
        a = ix[sources[ix] == source_contrast[0].upper()]
        b = ix[sources[ix] == source_contrast[1].upper()]
        complete = bool(valid[ix].all())
        if len(a) and len(b) and complete:
            original = float(baseline[a].mean() - baseline[b].mean())
            changed = float(scores[a].mean() - scores[b].mean())
            distortion = changed - original
            contrasts.append({"year": int(year), "n_A": len(a), "n_B": len(b),
                "baseline_difference": original, "masked_difference": changed,
                "signed_distortion": distortion, "absolute_distortion": abs(distortion),
                "standardized_absolute_distortion": abs(distortion) / sd if sd_ok else None,
                "sign_reversal": bool(original * changed < 0)})
        else:
            contrasts.append({"year": int(year), "n_A": len(a), "n_B": len(b),
                "baseline_difference": None, "masked_difference": None,
                "signed_distortion": None, "absolute_distortion": None,
                "standardized_absolute_distortion": None, "sign_reversal": None})
        if complete and len(ix) > 1:
            ranks.append(float(np.abs(average_ranks(scores[ix]) - average_ranks(baseline[ix])).mean() / (len(ix) - 1)))
            decile_n = max(1, math.ceil(0.1 * len(ix)))
            old = set(ix[np.lexsort((ids[ix], -baseline[ix]))[:decile_n]])
            new = set(ix[np.lexsort((ids[ix], -scores[ix]))[:decile_n]])
            replacements.append(len(old - new) / decile_n)
    signed = [c["signed_distortion"] for c in contrasts]
    absolute = [c["absolute_distortion"] for c in contrasts]
    contrast_ok = len(contrasts) > 0 and all(x is not None for x in signed)
    drift = scores - baseline
    all_valid = bool(valid.all())
    return {
        "n_focal": len(scores), "n_undefined": int((~valid).sum()),
        "baseline_population_sd": sd if math.isfinite(sd) else None,
        "contrasts_by_year": contrasts,
        "mean_signed_group_distortion": float(np.mean(signed)) if contrast_ok else None,
        "mean_absolute_group_distortion": float(np.mean(absolute)) if contrast_ok else None,
        "standardized_mean_absolute_group_distortion": float(np.mean(absolute)) / sd if contrast_ok and sd_ok else None,
        "mean_score_drift": float(drift.mean()) if all_valid else None,
        "rms_standardized_score_drift": float(np.sqrt(np.mean(drift ** 2))) / sd if all_valid and sd_ok else None,
        "mean_normalized_within_year_rank_movement": float(np.mean(ranks)) if all_valid and ranks else None,
        "mean_baseline_top_decile_replacement_fraction": float(np.mean(replacements)) if all_valid and replacements else None,
    }
