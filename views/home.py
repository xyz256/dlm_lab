import streamlit as st

st.title("📈 DLM Lab")
st.markdown(
    "An interactive workbench for **dynamic linear models** (DLMs) and the "
    "**Kalman filter**. Every number on these pages comes from ~300 lines of plain "
    "NumPy in `dlm/core.py`, written to be read alongside the app."
)

st.markdown("#### The one-line summary")
st.info("A **DLM** is a *model* (linear-Gaussian state space). The **Kalman filter** is the "
        "*algorithm* that computes the exact posterior of the hidden state in that model. "
        "Smoothing, likelihood estimation, forecasting and Gibbs sampling are all built on top "
        "of the same filter output.")

st.latex(r"""
\begin{aligned}
\text{observation:}\quad & y_t = F_t^\top \theta_t + v_t, & v_t &\sim N(0, V)\\
\text{evolution:}\quad & \theta_t = G\,\theta_{t-1} + w_t, & w_t &\sim N(0, W)\\
\text{prior:}\quad & \theta_0 \sim N(m_0, C_0)
\end{aligned}""")

st.markdown("#### Suggested learning path")
c1, c2 = st.columns(2)
with c1:
    st.markdown("""
1. **🧱 Model & data** — what F, G, V, W mean; simulate and see the hidden state.
2. **🔁 Kalman filter, step by step** — scrub through time and watch
   prior → forecast → error → gain → posterior, with the actual numbers.
3. **↩️ Smoothing** — revisit the past with the full sample (RTS).
4. **⛰️ Likelihood & MLE** — the prediction-error decomposition, the likelihood
   surface, and the optimiser's path to the estimate.
""")
with c2:
    st.markdown("""
5. **🩺 Diagnostics** — standardized one-step errors should be white noise. Break it on purpose.
6. **🔮 Forecasting** — k-step-ahead means and widening intervals; hold-out checks.
7. **🎚️ Discount factors** — West & Harrison's way to specify W without estimating it.
8. **🎲 FFBS + Gibbs** — full Bayesian inference for states *and* variances.
""")

st.markdown("#### How to use the sidebar")
st.markdown("""
- **Data source**: start with *Simulate* — you know the truth, so you can see when inference works.
  Then try the *Nile* data (the textbook example of Durbin & Koopman) or your own CSV.
- **Variances used for inference** are what the filter *believes*. Setting them away from the
  true values is the fastest way to build intuition (try W → 0 and W → huge).
""")

st.markdown("#### Experiments worth trying")
st.markdown("""
- Local level, true W/V = 0.01 vs 10: how does the steady-state Kalman gain change? (filter page)
- Delete a block of observations (filter page): watch the variance balloon and then collapse.
- Set inference W to 1/50 of the truth and check the diagnostics page — the ACF tells on you.
- Compare MLE standard errors to the Gibbs posterior spread for T = 50 vs T = 500.
""")
st.caption("References: West & Harrison (1997) *Bayesian Forecasting and Dynamic Models*; "
           "Petris, Petrone & Campagnoli (2009) *Dynamic Linear Models with R*; "
           "Durbin & Koopman (2012) *Time Series Analysis by State Space Methods*.")
