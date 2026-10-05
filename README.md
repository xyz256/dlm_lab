# DLM Lab

An interactive Streamlit workbench for learning **dynamic linear models** and the **Kalman filter**:
the model, filtering, smoothing, likelihood and MLE, diagnostics, forecasting, discount factors
and full Bayesian inference (FFBS + Gibbs).

All inference code is hand-written NumPy in `dlm/core.py` (~300 lines) so you can read it next to
the app. It's cross-checked against `statsmodels` in `tests/`.

## Run it on a Mac

```bash
cd dlm_lab
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. Python 3.10+ is recommended
(`python3 --version`; if it's older, `brew install python@3.12`).

Or just run `./run.sh`, which does the steps above.

Run the tests (optional):

```bash
pip install pytest
pytest -q
```

## Layout

```
app.py            navigation + shared sidebar (data source, model, variances)
ui.py             sidebar, cached filter runs, plotting helpers
dlm/models.py     model specs: local level, local linear trend, level+seasonal, dynamic regression
dlm/core.py       simulate, Kalman filter, RTS smoother, FFBS, forecasting, MLE, Gibbs sampler
views/            one file per page
tests/            engine tests (vs statsmodels) and page smoke tests (Streamlit AppTest)
```

## Pages

| Page | What you learn |
|---|---|
| Model & data | What F, G, V, W do; superposition; simulate and see the hidden state |
| Kalman filter, step by step | Predict → forecast → error → gain → update, with real numbers at any t; missing data; steady-state gain |
| Smoothing (RTS) | Backward pass, hindsight estimates, why smoothed variance ≤ filtered |
| Likelihood & MLE | Prediction-error decomposition, log-likelihood surface with optimiser path, SEs via Hessian |
| Diagnostics | Standardized errors, ACF/Ljung–Box, QQ, heteroskedasticity; scale W to see misspecification |
| Forecasting | k-step means and variances, fan charts, hold-out RMSE / coverage / log score |
| Discount factors | West & Harrison δ, implied time-varying W_t, choosing δ by likelihood |
| FFBS + Gibbs | Joint state paths, conjugate IG updates for V and W, traces, posteriors, ESS |

## Ideas to extend it

- EM algorithm page (uses the smoother's E-step) next to the MLE page
- Multivariate observations (y_t a vector, V a matrix)
- Unknown V learned sequentially (normal–gamma filter, Student-t forecasts)
- Outlier handling / intervention analysis (e.g. the 1899 Aswan dam break in the Nile data)
- Particle filter page for a nonlinear model, contrasted with the Kalman filter
