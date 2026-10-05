import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import ui
from dlm import core

ctx = ui.get_ctx()
spec, y = ctx.spec, ctx.y
st.title("🎲 Bayesian inference: FFBS + Gibbs")
st.caption(ctx.label)

ui.concept(r"""
MLE plugs in point estimates of $V, W$ and ignores their uncertainty. The Bayesian route samples
from the **joint** posterior $p(\theta_{1:T}, V, W \mid y)$ by alternating two easy conditionals:

**① States given variances — Forward-Filtering Backward-Sampling (FFBS).**
Run the Kalman filter, draw $\theta_T \sim N(m_T, C_T)$, then go backwards:
$$
\theta_t \mid \theta_{t+1}, y_{1:t} \sim N\big(m_t + B_t(\theta_{t+1}-a_{t+1}),\; C_t - B_t R_{t+1} B_t^\top\big)
$$
This gives one *whole path* drawn jointly — unlike the smoother, which gives marginal means/variances.

**② Variances given states — conjugate inverse-gamma updates.**
With priors $V \sim IG(\alpha_0, \beta_0)$ and $W_{ii} \sim IG(\alpha_0, \beta_0)$:
$$
V \mid \theta, y \sim IG\Big(\alpha_0 + \tfrac{T}{2},\ \beta_0 + \tfrac12\sum_t (y_t - F_t^\top\theta_t)^2\Big)
$$
$$
W_{ii} \mid \theta \sim IG\Big(\alpha_0 + \tfrac{T-1}{2},\ \beta_0 + \tfrac12\sum_t (\theta_{t,i} - (G\theta_{t-1})_i)^2\Big)
$$
Once the states are known, the variances are just variances of residuals.

Repeat thousands of times, discard burn-in, and the draws approximate the posterior.
""")

c1, c2, c3, c4 = st.columns(4)
n_iter = c1.select_slider("Iterations", [200, 500, 1000, 2000, 5000], 1000, key="gn")
burn = c2.slider("Burn-in (discard)", 0, n_iter // 2, min(200, n_iter // 4), key="gb")
a0 = c3.number_input("Prior shape α₀", 1.01, 100.0, 2.0, key="ga0",
                     help="Small α₀ = weak prior. β₀ is set so the prior mean ≈ var(Δy).")
seed = c4.number_input("Seed", 0, 9999, 0, key="gs")
st.caption(f"Cost ≈ {n_iter} Kalman filter + backward-sampling passes over T = {len(y)}.")

sig = (ctx.label, float(np.nansum(y)), len(y), n_iter, a0, seed)
if st.button("▶️ Run Gibbs sampler", type="primary"):
    bar = st.progress(0.0, text="Sampling…")
    res = core.gibbs(spec, y, n_iter=n_iter, a0=a0, seed=int(seed),
                     progress=lambda i, n: bar.progress(i / n, text=f"Sampling… {i}/{n}") if i % 25 == 0 else None)
    bar.empty()
    st.session_state["gibbs"] = (sig, res)
g = st.session_state.get("gibbs")
g = g[1] if g and g[0] == sig else None
if g is None:
    st.info("Press **Run** to sample. Start with 1000 iterations.")
    st.stop()

draws = np.column_stack([g["V"], g["W"]])
names = g["names"]
truth = None
if ctx.V_true is not None:
    truth = [ctx.V_true] + list(np.diag(ctx.W_true)[spec.w_free])
mle = st.session_state.get("mle")
mle = mle[1]["est"] if mle and mle[0][0] == ctx.label and mle[0][1] == len(y) else None

# ------------------------------------------------ traces and posteriors
k = draws.shape[1]
fig = make_subplots(rows=k, cols=2, column_widths=[0.62, 0.38], horizontal_spacing=0.07,
                    subplot_titles=sum([[f"trace: {n}", f"posterior: {n}"] for n in names], []))
for j in range(k):
    fig.add_trace(go.Scatter(y=draws[:, j], line=dict(color=ui.PALETTE["bayes"], width=1), showlegend=False),
                  row=j + 1, col=1)
    fig.add_vrect(x0=0, x1=burn, fillcolor="#999", opacity=0.12, line_width=0, row=j + 1, col=1)
    fig.add_trace(go.Histogram(x=draws[burn:, j], nbinsx=40, histnorm="probability density",
                               marker_color=ui.PALETTE["bayes"], showlegend=False), row=j + 1, col=2)
    if truth is not None:
        fig.add_vline(x=truth[j], line=dict(color="#ff006e", width=2), row=j + 1, col=2)
    if mle is not None:
        fig.add_vline(x=mle[j], line=dict(color="#ffb703", width=2, dash="dash"), row=j + 1, col=2)
ui.layout(fig, h=230 * k + 60)
fig.update_layout(hovermode="closest")
ui.show(fig)
st.caption("Grey = burn-in. Pink line = truth (simulated data). Orange dashed = MLE (if fitted on the MLE page).")

post = draws[burn:]
summ = pd.DataFrame({"posterior mean": post.mean(0), "posterior sd": post.std(0),
                     "2.5%": np.percentile(post, 2.5, 0), "97.5%": np.percentile(post, 97.5, 0)},
                    index=names)
if truth is not None:
    summ.insert(0, "true", truth)
if mle is not None:
    summ["MLE"] = mle
# effective sample size via initial positive sequence of autocorrelations
def ess(x):
    x = x - x.mean()
    n = len(x)
    if n < 10 or x.var() == 0:
        return np.nan
    ac = np.correlate(x, x, "full")[n - 1:] / (x.var() * n)
    s = 0.0
    for r in ac[1:]:
        if r < 0.05:
            break
        s += r
    return n / (1 + 2 * s)
summ["ESS"] = [ess(post[:, j]) for j in range(k)]
st.dataframe(summ.style.format("{:.4g}"), width="stretch")

# ------------------------------------------------ states
st.subheader("Posterior of the states")
comp = st.selectbox("State component", spec.state_names, key="bcomp")
i = spec.state_names.index(comp)
f2 = go.Figure()
if i == 0:
    for p_ in g["paths"]:
        f2.add_trace(go.Scatter(x=ctx.t, y=p_, line=dict(color=ui.PALETTE["bayes"], width=0.6),
                                opacity=0.35, showlegend=False, hoverinfo="skip"))
ui.band(f2, ctx.t, g["theta_mean"][:, i], g["theta_sd"][:, i], "posterior mean ±1.96 sd (Gibbs)",
        ui.PALETTE["bayes"], opacity=0.2)
sm = core.rts_smoother(spec, ui.filt(ctx))
f2.add_trace(go.Scatter(x=ctx.t, y=sm["s"][:, i], name="RTS smoother (sidebar V, W)",
                        line=dict(color=ui.PALETTE["smooth"], dash="dash")))
if ctx.theta_true is not None:
    f2.add_trace(go.Scatter(x=ctx.t, y=ctx.theta_true[:, i], name="true",
                            line=dict(color=ui.PALETTE["true"], width=1.2)))
if i in spec.obs_combo and len(spec.obs_combo) == 1:
    f2.add_trace(ui.obs_trace(ctx.t, y))
ui.show(ui.layout(f2, h=460))
st.caption("Thin lines (level only) are individual FFBS draws — each is a plausible full history. "
           "State moments are averaged over all iterations, including burn-in.")

ui.notice([
    "Gibbs bands are a bit wider than the smoother's: they also carry uncertainty about V and W.",
    "Traces for W often mix slowly (high autocorrelation, low ESS) — states and W are strongly dependent. "
    "That's the motivation for marginal samplers / HMC in tools like Stan.",
    "With T = 50 the posterior for W is wide and skewed; with T = 500 it tightens toward the MLE.",
])
