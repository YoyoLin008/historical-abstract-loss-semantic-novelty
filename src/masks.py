"""Outcome-independent, exact-count stochastic masking and paired controls."""
import math
import numpy as np


def exact_count(n, rate):
    if not 0 <= rate <= 1 or n < 0:
        raise ValueError("Rate must be in [0,1] and collection size nonnegative")
    return min(n, int(math.floor(n * rate + 0.5)))


def mask_rng(seed, rate, mechanism, replicate, control=False):
    codes = {"random": 1, "source": 2, "topic": 3}
    if mechanism not in codes or replicate < 0:
        raise ValueError("Invalid masking condition")
    return np.random.default_rng(np.random.SeedSequence([
        int(seed), int(round(rate * 1_000_000)), codes[mechanism],
        int(replicate), int(control)]))


def make_mask(history, rate, mechanism, rng, preferred_source="ACL",
              preferred_topic=0, weight=8.0, topic_buckets=8):
    """Weighted sampling without replacement; preference is fixed before scoring."""
    n, count = len(history), exact_count(len(history), rate)
    mask = np.zeros(n, dtype=bool)
    if not count:
        return mask
    if weight <= 0 or not math.isfinite(weight):
        raise ValueError("Structured sampling weight must be finite and positive")
    if mechanism == "random":
        indices = rng.choice(n, size=count, replace=False)
    else:
        if mechanism == "source":
            preferred = np.asarray([r["source"].upper() == preferred_source.upper() for r in history])
        elif mechanism == "topic":
            if any("topic" not in r for r in history):
                raise ValueError("Preassigned title-keyword lexical-proxy strata are required")
            preferred = np.asarray([r["topic"] == preferred_topic for r in history])
        else:
            raise ValueError("Unknown masking mechanism")
        weights = np.where(preferred, weight, 1.0)
        # Independent exponential clocks produce weighted sampling without replacement.
        priority = rng.exponential(scale=1.0 / weights)
        indices = np.argsort(priority, kind="stable")[:count]
    mask[indices] = True
    return mask


def year_matched_control(history, structured_mask, rng):
    """Randomize within each historical year, preserving every year's count."""
    structured_mask = np.asarray(structured_mask, dtype=bool)
    if structured_mask.shape != (len(history),):
        raise ValueError("Mask length does not match historical collection")
    years = np.asarray([int(r["year"]) for r in history])
    out = np.zeros(len(history), dtype=bool)
    for year in sorted(set(years)):
        ix = np.flatnonzero(years == year)
        count = int(structured_mask[ix].sum())
        out[rng.choice(ix, size=count, replace=False)] = True
    return out
