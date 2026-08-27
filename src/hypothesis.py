"""Simulation-based hypothesis tests and power analysis."""
from __future__ import annotations

import numpy as np


def permutation_mean_difference(a, b, n_perm=5000, alternative='two-sided', seed=42):
    a = np.asarray(a, float); b = np.asarray(b, float)
    observed = a.mean() - b.mean()
    pooled = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    stats = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(pooled)
        stats[i] = perm[:len(a)].mean() - perm[len(a):].mean()
    if alternative == 'greater':
        p = (np.sum(stats >= observed) + 1) / (n_perm + 1)
    elif alternative == 'less':
        p = (np.sum(stats <= observed) + 1) / (n_perm + 1)
    else:
        p = (np.sum(np.abs(stats) >= abs(observed)) + 1) / (n_perm + 1)
    return {"observed_difference": float(observed), "p_value": float(p)}


def simulated_power(effect_size=.5, n_per_group=50, alpha=.05, simulations=2000, seed=42):
    from scipy.stats import ttest_ind
    rng = np.random.default_rng(seed)
    reject = 0
    for _ in range(simulations):
        a = rng.normal(0, 1, n_per_group)
        b = rng.normal(effect_size, 1, n_per_group)
        reject += ttest_ind(a, b, equal_var=False).pvalue < alpha
    return reject / simulations
