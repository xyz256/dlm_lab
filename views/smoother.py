import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import ui
from dlm import core

ctx = ui.get_ctx()
spec = ctx.spec
st.title("↩️ Smoothing: using the future to revisit the past")
st.caption(ctx.label)

ui.concept(r"""
The filter gives $\theta_t \mid y_{1:t}$ — what you could know **in real time**.
The smoother gives $\theta_t \mid y_{1:T}$ — the best estimate **in hindsight**.

The Rauch–Tung–Striebel recursion runs **backwards**, starting from $s_T=m_T,\ S_T=C_T$:
$$
B_t = C_t G^\top R_{t+1}^{-1}
$$
$$
s_t = m_t + B_t\,(s_{t+1} - a_{t+1})
$$
$$
S_t = C_t + B_t\,(S_{t+1} - R_{t+1})\,B_t^\top
$$

Interpretation: $s_{t+1}-a_{t+1}$ is how much the future revised our view of $t+1$ relative to what
we predicted at time $t$. $B_t$ maps that revision back to time $t$. Because $S_{t+1}\le R_{t+1}$, the
smoothed variance is never larger than the filtered one.

It reuses the filter's stored $a, R, m, C$ — no new pass over the data. Smoothing is what you want
for **signal extraction** and for the **E-step** of EM; filtering is what you want for **forecasting**.
""")

out = ui.filt(ctx)
sm = core.rts_smoother(spec, out)
x = ctx.t

comp = st.selectbox("State component", spec.state_names, key="scomp")
i = spec.state_names.index(comp)

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.06,
                    subplot_titles=[f"{comp}: filtered vs smoothed (95% bands)", "Posterior standard deviation"])
ui.band(fig, x, out["m"][:, i], np.sqrt(out["C"][:, i, i]), "filtered m_t", ui.PALETTE["filter"],
        opacity=0.12, row=1, col=1)
ui.band(fig, x, sm["s"][:, i], np.sqrt(sm["S"][:, i, i]), "smoothed s_t", ui.PALETTE["smooth"],
        opacity=0.18, row=1, col=1)
if i in spec.obs_combo and len(spec.obs_combo) == 1:
    fig.add_trace(ui.obs_trace(x, ctx.y), row=1, col=1)
if ctx.theta_true is not None:
    fig.add_trace(go.Scatter(x=x, y=ctx.theta_true[:, i], name="true state",
                             line=dict(color=ui.PALETTE["true"], width=1.5)), row=1, col=1)
ui.focus_y(fig, out["m"][:, i], np.sqrt(out["C"][:, i, i]), spec.p, row=1)
fig.add_trace(go.Scatter(x=x, y=np.sqrt(out["C"][:, i, i]), name="sd filtered",
                         line=dict(color=ui.PALETTE["filter"])), row=2, col=1)
fig.add_trace(go.Scatter(x=x, y=np.sqrt(sm["S"][:, i, i]), name="sd smoothed",
                         line=dict(color=ui.PALETTE["smooth"])), row=2, col=1)
lo = np.nanpercentile(np.sqrt(out["C"][:, i, i]), 2)
hi = np.nanpercentile(np.sqrt(out["C"][:, i, i]), 98)
fig.update_yaxes(range=[0, hi * 1.3 if hi > 0 else 1], row=2, col=1)
ui.show(ui.layout(fig, h=620))

if ctx.theta_true is not None:
    rmse_f = np.sqrt(np.mean((out["m"][:, i] - ctx.theta_true[:, i]) ** 2))
    rmse_s = np.sqrt(np.mean((sm["s"][:, i] - ctx.theta_true[:, i]) ** 2))
    cov_s = np.mean(np.abs(sm["s"][:, i] - ctx.theta_true[:, i]) <= 1.96 * np.sqrt(sm["S"][:, i, i]))
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE filtered vs truth", f"{rmse_f:.4g}")
    c2.metric("RMSE smoothed vs truth", f"{rmse_s:.4g}", delta=f"{100 * (rmse_s / rmse_f - 1):.0f}%",
              delta_color="inverse")
    c3.metric("95% band coverage (smoothed)", f"{100 * cov_s:.0f}%")

with st.expander("Smoother gain B_t over time"):
    f2 = go.Figure(go.Scatter(x=x[:-1], y=sm["B"][:-1, i, i], line=dict(color=ui.PALETTE["smooth"])))
    ui.show(ui.layout(f2, h=260, title=f"B_t[{comp},{comp}]"))
    st.caption("For the local level model B_t → 1 − K (steady state): the less weight the filter put on "
               "the new observation, the more the future has to say about the past.")

ui.notice([
    "Filtered estimates lag turning points; smoothed estimates don't, because they've seen what came next.",
    "Smoothed sd is smallest in the middle and equals the filtered sd at t = T (no future to borrow from).",
    "Inside a missing-data gap the smoothed path interpolates linearly-ish, and its band is narrowest at the edges.",
])
