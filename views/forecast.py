import numpy as np
import plotly.graph_objects as go
import streamlit as st

import ui
from dlm import core

ctx = ui.get_ctx()
spec, y = ctx.spec, ctx.y
T = len(y)
st.title("🔮 Forecasting")
st.caption(ctx.label)

ui.concept(r"""
From the last filtered posterior $\theta_T \sim N(m_T, C_T)$, just keep predicting **without updating**:
$$
a_T(k) = G\,a_T(k-1),\qquad R_T(k) = G R_T(k-1) G^\top + W
$$
$$
f_T(k) = F^\top a_T(k),\qquad Q_T(k) = F^\top R_T(k) F + V
$$
starting from $a_T(0)=m_T,\ R_T(0)=C_T$. This is literally the filter with every future
observation missing.

Forecast shapes follow from $G$: local level → **flat** forecast with intervals that widen like
$\sqrt{k}$; local linear trend → **straight line**, intervals widen faster (slope uncertainty
compounds); seasonal → the seasonal pattern repeats.
""")

c1, c2 = st.columns(2)
h = c1.slider("Forecast horizon k", 1, max(2, min(60, T // 2)), min(12, T // 3), key="h")
holdout = c2.checkbox("Hold out the last k observations and score the forecast", value=True, key="hold")

n_fit = T - h if holdout else T
spec_fit = type(spec)(**{**spec.__dict__, "F": spec.F[:n_fit]})
out = core.kalman_filter(spec_fit, y[:n_fit], ctx.V, ctx.W)
F_future = spec.F[n_fit:n_fit + h] if holdout else None
if spec.name.startswith("Dynamic regression") and not holdout:
    st.warning("Dynamic regression needs future x values; holding the last x fixed for the forecast.")
fc = core.forecast(spec_fit, out, ctx.V, ctx.W, h, F_future=F_future)

xf = ctx.t[n_fit:n_fit + h] if holdout else np.arange(ctx.t[-1] + 1, ctx.t[-1] + 1 + h)
fig = go.Figure()
fig.add_trace(ui.obs_trace(ctx.t, y))
sig_m = np.einsum("tp,tp->t", spec_fit.F, out["m"])
fig.add_trace(go.Scatter(x=ctx.t[:n_fit], y=sig_m, name="filtered signal",
                         line=dict(color=ui.PALETTE["filter"])))
sd = np.sqrt(fc["Q"])
ui.band(fig, xf, fc["f"], sd, "95% interval", ui.PALETTE["fcst"], z=1.96, show_mean=False, opacity=0.15)
ui.band(fig, xf, fc["f"], sd, "forecast f_T(k)", ui.PALETTE["fcst"], z=0.674, opacity=0.25)
if holdout:
    fig.add_vrect(x0=xf[0], x1=xf[-1], fillcolor="#999", opacity=0.07, line_width=0)
ui.show(ui.layout(fig, h=460, title="Forecast fan (50% and 95% predictive intervals)"))

if holdout:
    yh = y[n_fit:n_fit + h]
    ok = ~np.isnan(yh)
    err = yh[ok] - fc["f"][ok]
    cover = np.mean(np.abs(err) <= 1.96 * sd[ok])
    lps = np.mean(-0.5 * (np.log(2 * np.pi * fc["Q"][ok]) + err ** 2 / fc["Q"][ok]))
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE (hold-out)", f"{np.sqrt(np.mean(err ** 2)):.4g}")
    c2.metric("95% interval coverage", f"{100 * cover:.0f}%")
    c3.metric("Mean log predictive score", f"{lps:.3f}", help="Higher is better; rewards calibrated intervals")

with st.expander("How the forecast variance builds up"):
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=np.arange(1, h + 1), y=fc["Q"], name="Q_T(k) total",
                              line=dict(color=ui.PALETTE["fcst"])))
    fig2.add_trace(go.Scatter(x=np.arange(1, h + 1), y=fc["Q"] - ctx.V, name="state part F'R_T(k)F",
                              line=dict(color=ui.PALETTE["filter"], dash="dot")))
    fig2.add_hline(y=ctx.V, line=dict(dash="dash", color="#999"), annotation_text="V (irreducible)")
    ui.show(ui.layout(fig2, h=300, xaxis_title="k (steps ahead)"))
    st.caption("Observation noise V is a floor; the state part grows with k at a rate set by W and G.")

ui.notice([
    "Switch between local level and local linear trend on the same data: same history, very different futures.",
    "Shrink W toward 0 in the sidebar: intervals stop widening — you've told the model the future looks like the past.",
    "Coverage far below 95% on the hold-out is a sign the variances (usually W) are too small.",
])
