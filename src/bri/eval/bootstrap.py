"""Bootstrap confidence intervals over evaluation cases (customers).

The resampling unit is the evaluation case, i.e. one customer within one task and
split. Rankings within a case are never resampled independently. For model
comparisons, the *same* resampled case indices are applied to both models (paired
bootstrap), so customer-to-customer variation cancels in the difference.

Intervals are percentile intervals. `share_not_better` is the fraction of resamples in
which model A's mean is <= model B's. It is a descriptive stability measure, not a
p-value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_CHUNK = 200  # resamples per batch; bounds memory to _CHUNK * n_cases indices


def _resampled_means(values: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Means of n_boot resamples (with replacement) of the rows of `values`.

    `values` is (n_cases,) or (n_cases, m); identical seeds give identical index draws,
    which is what makes paired comparisons possible.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    n = len(values)
    if n == 0:
        raise ValueError("cannot bootstrap an empty set of cases")
    rng = np.random.default_rng(seed)
    out = []
    for start in range(0, n_boot, _CHUNK):
        idx = rng.integers(0, n, size=(min(_CHUNK, n_boot - start), n))
        out.append(values[idx].mean(axis=1))
    return np.concatenate(out)


def _interval(samples: np.ndarray, confidence: float) -> tuple[float, float]:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    lo, hi = np.quantile(samples, [(1 - confidence) / 2, (1 + confidence) / 2])
    return float(lo), float(hi)


def bootstrap_ci(values, n_boot: int = 1000, seed: int = 0, confidence: float = 0.95) -> dict:
    """Mean and percentile CI of a per-case metric."""
    v = np.asarray(values, dtype=float)
    lo, hi = _interval(_resampled_means(v, n_boot, seed), confidence)
    return {"mean": float(v.mean()), "ci_low": lo, "ci_high": hi}


def paired_bootstrap_diff(
    a: pd.Series, b: pd.Series, n_boot: int = 1000, seed: int = 0, confidence: float = 0.95
) -> dict:
    """CI for mean(a) - mean(b), resampling cases jointly. a and b must share one case index."""
    if not a.index.equals(b.index):
        raise ValueError("paired bootstrap needs both models scored on the same cases in the same order")
    pair = np.column_stack([a.to_numpy(dtype=float), b.to_numpy(dtype=float)])
    means = _resampled_means(pair, n_boot, seed)
    diffs = means[:, 0] - means[:, 1]
    lo, hi = _interval(diffs, confidence)
    return {
        "mean_diff": float(pair[:, 0].mean() - pair[:, 1].mean()),
        "ci_low": lo,
        "ci_high": hi,
        "share_not_better": float((diffs <= 0).mean()),
    }


def _cluster_resampled_means(values: np.ndarray, clusters, n_boot: int, seed: int) -> np.ndarray:
    """Means over cases after resampling whole clusters (customers) with replacement.

    Each resample draws as many clusters as there are; its mean is the pooled mean over
    all cases of the drawn clusters (a ratio estimator), matching the case-level mean
    reported for the full sample.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    v = values if values.ndim == 2 else values[:, None]
    if len(v) == 0:
        raise ValueError("cannot bootstrap an empty set of cases")
    codes, uniques = pd.factorize(pd.Series(clusters).to_numpy(), sort=True)
    g = len(uniques)
    sums = np.zeros((g, v.shape[1]))
    np.add.at(sums, codes, v)
    counts = np.bincount(codes, minlength=g).astype(float)
    rng = np.random.default_rng(seed)
    out = []
    for start in range(0, n_boot, _CHUNK):
        idx = rng.integers(0, g, size=(min(_CHUNK, n_boot - start), g))
        out.append(sums[idx].sum(axis=1) / counts[idx].sum(axis=1)[:, None])
    res = np.concatenate(out)
    return res if values.ndim == 2 else res[:, 0]


def cluster_bootstrap_ci(values, clusters, n_boot: int = 1000, seed: int = 0, confidence: float = 0.95) -> dict:
    """Mean and percentile CI of a per-case metric, resampling clusters (customers)."""
    v = np.asarray(values, dtype=float)
    lo, hi = _interval(_cluster_resampled_means(v, clusters, n_boot, seed), confidence)
    return {"mean": float(v.mean()), "ci_low": lo, "ci_high": hi}


def paired_cluster_bootstrap_diff(
    a: pd.Series, b: pd.Series, clusters: pd.Series, n_boot: int = 1000, seed: int = 0,
    confidence: float = 0.95,
) -> dict:
    """Paired CI for mean(a) - mean(b) when one customer contributes several cases.

    The same resampled customers are applied to both models.
    """
    if not (a.index.equals(b.index) and a.index.equals(clusters.index)):
        raise ValueError("paired bootstrap needs both models and clusters on the same cases in the same order")
    pair = np.column_stack([a.to_numpy(dtype=float), b.to_numpy(dtype=float)])
    means = _cluster_resampled_means(pair, clusters.to_numpy(), n_boot, seed)
    diffs = means[:, 0] - means[:, 1]
    lo, hi = _interval(diffs, confidence)
    return {
        "mean_diff": float(pair[:, 0].mean() - pair[:, 1].mean()),
        "ci_low": lo,
        "ci_high": hi,
        "share_not_better": float((diffs <= 0).mean()),
    }
