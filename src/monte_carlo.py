"""Monte Carlo simulation helpers for probability and uncertainty studies."""
from __future__ import annotations

import numpy as np


def estimate_pi(n: int = 100_000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    points = rng.uniform(-1, 1, size=(n, 2))
    inside = np.square(points).sum(axis=1) <= 1
    estimate = 4 * inside.mean()
    se = 4 * np.sqrt(inside.mean() * (1 - inside.mean()) / n)
    return {"estimate": float(estimate), "standard_error": float(se), "n": int(n)}


def monte_carlo_expectation(draw, transform=lambda x: x, simulations: int = 10000, seed: int = 42):
    rng = np.random.default_rng(seed)
    sample = draw(rng, simulations)
    values = np.asarray(transform(sample), dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "se": float(values.std(ddof=1) / np.sqrt(len(values))),
        "q025": float(np.quantile(values, .025)),
        "q975": float(np.quantile(values, .975)),
    }


def convergence_trace(draw, transform=lambda x: x, n=20000, checkpoints=(100, 500, 1000, 5000, 10000, 20000), seed=42):
    rng = np.random.default_rng(seed)
    values = np.asarray(transform(draw(rng, n)), dtype=float)
    cumulative = np.cumsum(values) / np.arange(1, n + 1)
    return {int(k): float(cumulative[k - 1]) for k in checkpoints if k <= n}
