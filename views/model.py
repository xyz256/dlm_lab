import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import ui

ctx = ui.get_ctx()
spec = ctx.spec
st.title("🧱 Model & data")
st.caption(ctx.label)
st.markdown(f"**{spec.name}.** {spec.description}")

ui.concept(r"""
A DLM has two equations.

* **Observation** $y_t = F_t^\top\theta_t + v_t$: what you see is a linear combination of hidden
  states plus noise with variance $V$.
* **Evolution** $\theta_t = G\theta_{t-1} + w_t$: the hidden state moves by a known linear map $G$
  plus shocks with covariance $W$.

$V$ controls how noisy the measurements are; $W$ controls how fast the truth moves.
Only their **ratio** (signal-to-noise) matters for the shape of the filtered path; the scale
matters for interval widths and the likelihood.

Building blocks combine by *superposition*: stack the states, put $G$ blocks on the diagonal,
and concatenate the $F$ vectors. That's how "level + seasonal" is built here.
""", expanded=True)

ui.concept(r"""
| Symbol | Meaning |
|---|---|
| $t = 1,\dots,T$ | time index / number of observations |
| $y_t$ | observed data at time $t$ |
| $\theta_t$ | **hidden state vector** — the model's unobserved parameters at time $t$ |
| $F_t$ | **observation vector/matrix** — maps the state $\theta_t$ to the mean of $y_t$ |
| $G$ | **evolution matrix** — maps $\theta_{t-1}$ to $\theta_t$ |
| $v_t \sim N(0, V)$ | observation noise |
| $w_t \sim N(0, W)$ | state evolution noise (shocks to the state) |
| $V$ | observation noise **variance** |
| $W$ | state evolution noise **covariance matrix** |
| $a_t,\ R_t$ | **prior** (predicted) mean/covariance of $\theta_t$ given $y_{1:t-1}$ |
| $f_t,\ Q_t$ | **forecast** mean/variance of $y_t$ given $y_{1:t-1}$ |
| $e_t = y_t - f_t$ | one-step forecast error (the "surprise", aka innovation) |
| $K_t$ | **Kalman gain** — how much the surprise moves the state estimate |
| $m_t,\ C_t$ | **filtered** (posterior) mean/covariance of $\theta_t$ given $y_{1:t}$ |
| $s_t,\ S_t$ | **smoothed** mean/covariance of $\theta_t$ given the *whole* series $y_{1:T}$ |
| $B_t$ | smoother gain (RTS recursion) |
| $\delta$ | discount factor — an alternative, unit-free way to set $W$ |
| $\psi = (V, W)$ | the unknown variances, estimated by MLE or sampled in the Bayesian page |
| $\alpha_0,\ \beta_0$ | inverse-gamma prior hyperparameters used by the Gibbs sampler |

This notation is shared across every page in the app — refer back here if a symbol looks unfamiliar.
""", title="📖 Notation reference", expanded=False)

c1, c2, c3 = st.columns([1.2, 1.2, 1])
with c1:
    st.markdown("**G** (evolution)")
    st.dataframe(ui.matrix_df(spec.G, spec.state_names), width="stretch")
with c2:
    st.markdown("**W** (used for inference)")
    st.dataframe(ui.matrix_df(ctx.W, spec.state_names), width="stretch")
with c3:
    st.markdown("**F_t** (first 3 steps)")
    st.dataframe(ui.matrix_df(spec.F[:3], [f"t={i}" for i in (1, 2, 3)], spec.state_names),
                 width="stretch")
    st.metric("V (used for inference)", f"{ctx.V:.4g}")

free = np.diag(ctx.W)[spec.w_free]
st.markdown("**Signal-to-noise ratios** (W_ii / V): " +
            ", ".join(f"{n}: `{w / ctx.V:.3g}`" for n, w in zip(spec.free_names(), free)))

# ---- plot
n_states = spec.p if spec.p <= 3 else 2
fig = make_subplots(rows=1 + n_states, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                    subplot_titles=["Observations"] + spec.state_names[:n_states])
fig.add_trace(ui.obs_trace(ctx.t, ctx.y), row=1, col=1)
if ctx.theta_true is not None:
    sig = np.einsum("tp,tp->t", spec.F, ctx.theta_true)
    fig.add_trace(go.Scatter(x=ctx.t, y=sig, name="true signal F'θ",
                             line=dict(color=ui.PALETTE["true"])), row=1, col=1)
    for i in range(n_states):
        fig.add_trace(go.Scatter(x=ctx.t, y=ctx.theta_true[:, i], name=f"true {spec.state_names[i]}",
                                 line=dict(color=ui.PALETTE["true"]), showlegend=False),
                      row=2 + i, col=1)
else:
    for i in range(n_states):
        fig.add_annotation(text="hidden — no ground truth for real data; see the filter/smoother pages",
                           xref="x domain", yref="y domain", x=0.5, y=0.5, showarrow=False,
                           row=2 + i, col=1)
ui.show(ui.layout(fig, h=220 + 180 * n_states))

if ctx.theta_true is None:
    st.info("With real data the state is never observed. Everything we learn about it comes "
            "through the filter, smoother or sampler.")

ui.notice([
    "In simulation, bump the true W up or down in the sidebar and regenerate: the state path gets "
    "rougher or smoother while the observation noise stays the same.",
    "For the trend model, G's off-diagonal 1 is what makes the slope feed into the level.",
    "For dynamic regression, F_t changes every step (it contains x_t) — the model is still linear.",
])
