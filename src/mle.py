"""Closed-form and numerical maximum-likelihood estimation."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def normal_mle(data):
    x = np.asarray(data, float)
    return {"mu": float(x.mean()), "sigma": float(np.sqrt(np.mean((x-x.mean())**2)))}


def exponential_mle(data):
    x = np.asarray(data, float)
    if np.any(x < 0):
        raise ValueError("Exponential samples must be non-negative")
    return {"rate": float(1 / x.mean())}


def logistic_mle(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(x)), x])
    def nll(beta):
        z = X @ beta
        return float(np.sum(np.logaddexp(0, z) - y*z))
    result = minimize(nll, np.zeros(X.shape[1]), method='BFGS')
    return {"coef": result.x, "success": bool(result.success), "nll": float(result.fun)}
