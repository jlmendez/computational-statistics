"""Small Bayesian updating examples for conjugate models."""
from __future__ import annotations

import numpy as np


def beta_binomial_update(successes: int, trials: int, alpha_prior=1.0, beta_prior=1.0):
    failures = trials - successes
    alpha_post = alpha_prior + successes
    beta_post = beta_prior + failures
    mean = alpha_post / (alpha_post + beta_post)
    return {"alpha": float(alpha_post), "beta": float(beta_post), "posterior_mean": float(mean)}


def beta_credible_interval(alpha, beta, level=.95):
    from scipy.stats import beta as beta_dist
    tail = (1 - level) / 2
    return tuple(map(float, beta_dist.ppf([tail, 1-tail], alpha, beta)))


def normal_mean_update(sample, prior_mean=0.0, prior_sd=10.0, known_sd=1.0):
    x = np.asarray(sample, float)
    prior_precision = 1 / prior_sd**2
    data_precision = len(x) / known_sd**2
    posterior_var = 1 / (prior_precision + data_precision)
    posterior_mean = posterior_var * (prior_precision*prior_mean + data_precision*x.mean())
    return {"mean": float(posterior_mean), "sd": float(np.sqrt(posterior_var))}
