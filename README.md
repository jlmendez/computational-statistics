# Computational Statistics

Reproducible statistical-computing examples spanning simulation-based inference, bootstrap uncertainty, time-series diagnostics, Bayesian updating and maximum-likelihood estimation.

## Highlights

- Monte Carlo experiments for estimator bias
- Bootstrap confidence intervals
- Hypothesis testing and sampling variability
- ARIMA time-series workflow and stationarity checks
- Beta-Binomial Bayesian updating
- Maximum-likelihood estimation with numerical optimization

## Tech stack

Python · NumPy · pandas · SciPy · statsmodels · scikit-learn

## Repository structure

- `src/inference_demo.py` — Monte Carlo and bootstrap inference
- `src/time_series_bayes_mle.py` — ARIMA, Bayesian updating and MLE
- `requirements.txt` — dependencies

## Run

```bash
pip install -r requirements.txt
python src/inference_demo.py
python src/time_series_bayes_mle.py
```

All public demonstrations generate their own data and are reproducible without external datasets.
