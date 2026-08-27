"""Time-series diagnostics, Bayesian updating and numerical MLE."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


def simulate_ar1(n: int = 500, phi: float = 0.78, sigma: float = 1.0, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, sigma, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def arima_summary(series: np.ndarray) -> dict[str, float]:
    adf_stat, adf_p, *_ = adfuller(series)
    fit = ARIMA(series, order=(1, 0, 0)).fit()
    return {
        "adf_statistic": float(adf_stat),
        "adf_pvalue": float(adf_p),
        "ar1_estimate": float(fit.arparams[0]),
        "aic": float(fit.aic),
    }


def beta_binomial_update(successes: int, trials: int, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> dict[str, float]:
    a = prior_alpha + successes
    b = prior_beta + trials - successes
    return {"posterior_alpha": a, "posterior_beta": b, "posterior_mean": a / (a + b)}


def normal_mle(values: np.ndarray) -> dict[str, float | bool]:
    values = np.asarray(values, dtype=float)

    def nll(params: np.ndarray) -> float:
        mu, log_sigma = params
        sigma = np.exp(log_sigma)
        return float(len(values) * np.log(sigma) + 0.5 * np.sum(((values - mu) / sigma) ** 2))

    init = np.array([values.mean(), np.log(values.std(ddof=0))])
    result = minimize(nll, init, method="BFGS")
    mu, log_sigma = result.x
    return {"mu": float(mu), "sigma": float(np.exp(log_sigma)), "success": bool(result.success)}


def main() -> None:
    series = simulate_ar1()
    print("ARIMA / stationarity")
    for key, value in arima_summary(series).items():
        print(f"  {key}: {value:.4f}")

    print("\nBayesian default-rate update")
    print(beta_binomial_update(successes=18, trials=250, prior_alpha=2, prior_beta=18))

    rng = np.random.default_rng(9)
    sample = rng.normal(12.0, 2.4, 400)
    print("\nNormal MLE")
    print(normal_mle(sample))


if __name__ == "__main__":
    main()
