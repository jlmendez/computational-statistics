"""Nonparametric bootstrap estimates and percentile confidence intervals."""
from __future__ import annotations

import numpy as np


def bootstrap_statistic(data, statistic=np.mean, n_boot: int = 2000, seed: int = 42):
    x = np.asarray(data)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    estimates = np.apply_along_axis(statistic, 1, x[idx])
    return estimates


def percentile_interval(estimates, alpha: float = .05):
    values = np.asarray(estimates, dtype=float)
    return tuple(map(float, np.quantile(values, [alpha / 2, 1 - alpha / 2])))


def bootstrap_summary(data, statistic=np.mean, n_boot=2000, seed=42):
    estimates = bootstrap_statistic(data, statistic, n_boot, seed)
    low, high = percentile_interval(estimates)
    return {
        "estimate": float(statistic(np.asarray(data))),
        "bootstrap_mean": float(estimates.mean()),
        "bootstrap_se": float(estimates.std(ddof=1)),
        "ci95": (low, high),
    }
