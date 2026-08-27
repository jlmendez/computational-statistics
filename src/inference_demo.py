"""Monte Carlo and bootstrap examples for computational inference."""
from __future__ import annotations

import numpy as np


def monte_carlo_variance_bias(population_sigma: float = 4.0, sample_size: int = 12, simulations: int = 20_000, seed: int = 42) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    samples = rng.normal(0.0, population_sigma, size=(simulations, sample_size))
    biased = samples.var(axis=1, ddof=0)
    unbiased = samples.var(axis=1, ddof=1)
    truth = population_sigma**2
    return {
        "true_variance": truth,
        "mean_ddof0": float(biased.mean()),
        "bias_ddof0": float(biased.mean() - truth),
        "mean_ddof1": float(unbiased.mean()),
        "bias_ddof1": float(unbiased.mean() - truth),
    }


def bootstrap_mean_ci(values: np.ndarray, confidence: float = 0.95, resamples: int = 10_000, seed: int = 42) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def main() -> None:
    result = monte_carlo_variance_bias()
    print("Variance estimator Monte Carlo")
    for key, value in result.items():
        print(f"  {key}: {value:.4f}")

    rng = np.random.default_rng(7)
    response_times = rng.lognormal(mean=4.4, sigma=0.25, size=250)
    lo, hi = bootstrap_mean_ci(response_times)
    print(f"\nBootstrap 95% CI for mean response time: [{lo:.2f}, {hi:.2f}]")


if __name__ == "__main__":
    main()
