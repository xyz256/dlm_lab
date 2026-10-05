import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import ui
from dlm import core

ctx = ui.get_ctx()
spec, T = ctx.spec, len(ctx.y)
st.title("🔁 The Kalman filter, step by step")
st.caption(ctx.label)

ui.concept(r"""
At each time $t$ the filter does two things.

**Predict (time update)** — push yesterday's belief through the evolution equation:
$$
a_t = G m_{t-1},\qquad R_t = G C_{t-1} G^\top + W
$$
$$
f_t = F_t^\top a_t,\qquad Q_t = F_t^\top R_t F_t + V
$$

**Update (measurement update)** — correct by the surprise $e_t = y_t - f_t$:
$$
K_t = R_t F_t / Q_t \quad(\text{Kalman gain / adaptive coefficient})
$$
$$
m_t = a_t + K_t e_t,\qquad C_t = R_t - K_t Q_t K_t^\top
$$

Everything is Gaussian, so means and variances are the whole story. The gain is a ratio of
*how unsure we are about the state* ($R_t$) to *how unsure we are about the next observation* ($Q_t$).
If $y_t$ is missing, skip the update: $m_t=a_t$, $C_t=R_t$.
""", expanded=False)

with st.expander("✂️ Knock out observations (missing data)"):
    gap = st.slider("Treat these time steps as missing", 1, T, (T, T), key="gap")
    use_gap = st.checkbox("Apply gap", value=False, key="use_gap")
y = ctx.y.copy()
if use_gap:
    y[gap[0] - 1: gap[1]] = np.nan

out = ui.filt(ctx, y=y)
t_idx = st.slider("Time step t", 1, T, min(10, T), key="t_step") - 1
x = ctx.t

# ---------------------------------------------------- numbers at step t
st.subheader(f"What happens at t = {x[t_idx]}")
m_prev = out["m0"] if t_idx == 0 else out["m"][t_idx - 1]
C_prev = out["C0"] if t_idx == 0 else out["C"][t_idx - 1]
a, R, f, Q = out["a"][t_idx], out["R"][t_idx], out["f"][t_idx], out["Q"][t_idx]
m, C, e, K = out["m"][t_idx], out["C"][t_idx], out["e"][t_idx], out["K"][t_idx]
missing = np.isnan(y[t_idx])

if spec.p == 1:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**① Predict**")
        st.latex(rf"a_t = G m_{{t-1}} = {m_prev[0]:.4g}")
        st.latex(rf"R_t = {C_prev[0, 0]:.4g} + {ctx.W[0, 0]:.4g} = {R[0, 0]:.4g}")
    with c2:
        st.markdown("**② Forecast y**")
        st.latex(rf"f_t = {f:.4g}")
        st.latex(rf"Q_t = {R[0, 0]:.4g} + {ctx.V:.4g} = {Q:.4g}")
        y_txt = r"\text{missing}" if missing else f"{y[t_idx]:.4g}"
        st.latex(r"y_t = " + y_txt)
    with c3:
        st.markdown("**③ Update**")
        if missing:
            st.latex(r"m_t=a_t,\ C_t=R_t")
        else:
            st.latex(rf"e_t = {e:.4g},\quad K_t = \tfrac{{{R[0, 0]:.4g}}}{{{Q:.4g}}} = {K[0]:.4f}")
            st.latex(rf"m_t = a_t + K_t e_t")
            st.latex(rf"= {a[0]:.4g} + {K[0]:.4f}\times{e:.4g} = {m[0]:.4g}")
            st.latex(rf"C_t = (1-K_t)R_t = {C[0, 0]:.4g}")
    if not missing:
        st.caption(f"The posterior mean moves {100 * K[0]:.1f}% of the way from the prediction "
                   f"toward the observation — this is exponential smoothing with weight K_t.")
else:
    names = spec.state_names
    tbl = pd.DataFrame({"m_{t-1}": m_prev, "a_t (prior)": a, "√diag R_t": np.sqrt(np.diag(R)),
                        "K_t (gain)": K, "m_t (posterior)": m, "√diag C_t": np.sqrt(np.diag(C))},
                       index=names)
    c1, c2 = st.columns([2, 1])
    with c1:
        st.dataframe(tbl.style.format("{:.4g}"), width="stretch")
    with c2:
        st.metric("Forecast f_t", f"{f:.4g}", help="F_t' a_t")
        st.metric("√Q_t (forecast sd)", f"{np.sqrt(Q):.4g}")
        st.metric("Error e_t", "missing" if missing else f"{e:.4g}")
    with st.expander("Full matrices at this step"):
        a1, a2 = st.columns(2)
        a1.markdown("R_t (prior covariance)")
        a1.dataframe(ui.matrix_df(R, names), width="stretch")
        a2.markdown("C_t (posterior covariance)")
        a2.dataframe(ui.matrix_df(C, names), width="stretch")
    st.caption("Notice the gain is non-zero even for states that don't enter F_t directly (e.g. the "
               "slope): the observation informs them through their correlation with the level in R_t.")

# ---------------------------------------------------- plot over time
st.subheader("The whole filtered path")
comp = st.selectbox("State component", spec.state_names, key="fcomp")
i = spec.state_names.index(comp)
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.06,
                    subplot_titles=[f"{comp}: prior a_t and posterior m_t (95% bands)", "Kalman gain K_t"])
ui.band(fig, x, out["a"][:, i], np.sqrt(out["R"][:, i, i]), "prior a_t", ui.PALETTE["prior"],
        dash="dot", opacity=0.10, row=1, col=1)
ui.band(fig, x, out["m"][:, i], np.sqrt(out["C"][:, i, i]), "filtered m_t", ui.PALETTE["filter"],
        row=1, col=1)
if i in spec.obs_combo and len(spec.obs_combo) == 1:
    fig.add_trace(ui.obs_trace(x, y), row=1, col=1)
if ctx.theta_true is not None:
    fig.add_trace(go.Scatter(x=x, y=ctx.theta_true[:, i], name="true state",
                             line=dict(color=ui.PALETTE["true"], width=1.5)), row=1, col=1)
ui.focus_y(fig, out["m"][:, i], np.sqrt(out["C"][:, i, i]), spec.p, row=1)
fig.add_vline(x=x[t_idx], line=dict(color="#999", dash="dash"))
fig.add_trace(go.Scatter(x=x, y=out["K"][:, i], name="gain", line=dict(color=ui.PALETTE["fcst"]),
                         showlegend=False), row=2, col=1)
if spec.p == 1:
    Kinf, r = core.steady_state_gain_local_level(ctx.V, ctx.W[0, 0])
    fig.add_hline(y=Kinf, line=dict(dash="dot", color="#666"), row=2, col=1,
                  annotation_text=f"steady state {Kinf:.3f} (W/V={r:.3g})")
ui.show(ui.layout(fig, h=620))

ui.notice([
    "Early on, R_t is huge (vague prior) so K_t ≈ 1: the filter essentially copies the first data points.",
    "The gain settles to a constant quickly — after that the filter is a fixed linear smoother.",
    "Raise W (or lower V) in the sidebar → bigger gain, more jittery path. Lower W → smoother, laggier path.",
    "Apply a missing-data gap: no updates → the band widens linearly (variance grows by W each step), "
    "then snaps back when data return.",
])
