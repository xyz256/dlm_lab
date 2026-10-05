import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import ui
from dlm import core

ctx = ui.get_ctx()
spec, y = ctx.spec, ctx.y
st.title("🎚️ Discount factors")
st.caption(ctx.label)

ui.concept(r"""
Specifying $W$ directly is awkward: its scale depends on the units of the data. West & Harrison's
alternative is a **discount factor** $\delta\in(0,1]$: each step, the state's information is
discounted by $\delta$, i.e. its variance is inflated by $1/\delta$:
$$
R_t = \frac{G C_{t-1} G^\top}{\delta}\quad\Longleftrightarrow\quad W_t = \frac{1-\delta}{\delta}\,G C_{t-1} G^\top
$$

* $\delta = 1$: $W_t = 0$, a static model — all data get equal weight forever.
* $\delta \approx 0.95$–$0.99$: typical; roughly "forget ~$1/(1-\delta)$ observations back".
* $W_t$ is now **time-varying** and automatically in the right units.

In practice you choose $\delta$ by comparing the one-step predictive log-likelihood across a grid —
which this page does. (Separate discount factors per component, e.g. trend vs seasonal, are common.)
""")

delta = st.slider("Discount factor δ", 0.70, 1.0, 0.95, 0.005, key="delta")
od = ui.filt(ctx, discount=delta)
ow = ui.filt(ctx)

comp = st.selectbox("State component", spec.state_names, key="dcomp")
i = spec.state_names.index(comp)
x = ctx.t
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.07,
                    subplot_titles=[f"{comp}: discount filter vs fixed-W filter", "Implied W_t (discount) vs fixed W"])
ui.band(fig, x, ow["m"][:, i], np.sqrt(ow["C"][:, i, i]), "fixed W (sidebar)", ui.PALETTE["filter"],
        opacity=0.10, row=1, col=1)
ui.band(fig, x, od["m"][:, i], np.sqrt(od["C"][:, i, i]), f"discount δ={delta}", ui.PALETTE["bayes"],
        opacity=0.18, row=1, col=1)
if i in spec.obs_combo and len(spec.obs_combo) == 1:
    fig.add_trace(ui.obs_trace(x, y), row=1, col=1)
if ctx.theta_true is not None:
    fig.add_trace(go.Scatter(x=x, y=ctx.theta_true[:, i], name="true", line=dict(color=ui.PALETTE["true"])),
                  row=1, col=1)
ui.focus_y(fig, ow["m"][:, i], np.sqrt(ow["C"][:, i, i]), spec.p, row=1)
fig.add_trace(go.Scatter(x=x[spec.p:], y=od["Wt"][spec.p:, i, i], name="W_t (discount)",
                         line=dict(color=ui.PALETTE["bayes"])), row=2, col=1)
fig.add_hline(y=ctx.W[i, i], line=dict(dash="dash", color=ui.PALETTE["filter"]), row=2, col=1)
wt = od["Wt"][spec.p + 3:, i, i]
if len(wt):
    fig.update_yaxes(range=[0, 1.3 * max(np.percentile(wt, 98), ctx.W[i, i]) + 1e-12], row=2, col=1)
ui.show(ui.layout(fig, h=600))

st.subheader("Choosing δ by predictive log-likelihood")


@st.cache_data(show_spinner=False)
def grid(_spec, key, y, V):
    ds = np.round(np.arange(0.70, 1.0001, 0.01), 3)
    return ds, np.array([core.kalman_filter(_spec, y, V, None, discount=d, burn=_spec.p)["loglik"] for d in ds])


ds, lls = grid(spec, ctx.label + str(spec.F.shape), y, ctx.V)
best = ds[np.argmax(lls)]
f2 = go.Figure(go.Scatter(x=ds, y=lls, mode="lines+markers", line=dict(color=ui.PALETTE["bayes"])))
f2.add_vline(x=best, line=dict(dash="dot"), annotation_text=f"best δ = {best:.2f}")
f2.add_hline(y=ow["loglik"] if spec.p else 0, line=dict(dash="dash", color=ui.PALETTE["filter"]),
             annotation_text="fixed-W log-lik", annotation_position="bottom right")
ui.show(ui.layout(f2, h=320, xaxis_title="δ", yaxis_title="log-likelihood"))
st.caption("V is held at the sidebar value. In the full W&H treatment V is also learned sequentially "
           "with a conjugate gamma prior.")

ui.notice([
    "δ = 1 gives a model that stops learning: the band collapses and the filter can't follow changes.",
    "The implied W_t settles to a constant once C_t reaches steady state — discounting and a fixed W are "
    "close cousins for a stable model.",
    "With the Nile data, the best δ is well below 1 — the level does move.",
])
