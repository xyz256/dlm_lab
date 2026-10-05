import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from dlm import core

ctx = ui.get_ctx()
spec, y = ctx.spec, ctx.y
st.title("⛰️ Likelihood & maximum likelihood")
st.caption(ctx.label)

ui.concept(r"""
The variances $\psi = (V, W)$ are unknown. The filter gives the likelihood almost for free through the
**prediction-error decomposition**:
$$
\log p(y_{1:T}\mid\psi) = \sum_{t=1}^T \log p(y_t\mid y_{1:t-1},\psi)
= -\tfrac12\sum_t \Big[\log 2\pi + \log Q_t + \frac{e_t^2}{Q_t}\Big]
$$

Each term is just the Gaussian density of the one-step forecast error. So: **run the filter, add up
these terms, hand the result to an optimiser.**

Practical details implemented here:
* Variances are optimised on the **log scale** (keeps them positive, makes the surface rounder).
* The first $p$ terms are dropped (*burn-in*): with a vague prior they mostly reflect the prior, an
  approximation to the exact diffuse likelihood.
* Standard errors come from the inverse **numerical Hessian** at the optimum (on the log scale), then
  the delta method: $\mathrm{se}(\hat\sigma^2) \approx \hat\sigma^2\,\mathrm{se}(\log\hat\sigma^2)$.
* Alternatives you'll meet in practice: EM (uses the smoother), and Bayesian estimation (see the Gibbs page).
""")

# ------------------------------------------- prediction error decomposition at current params
out = ui.filt(ctx)
burn = spec.p
st.subheader("1 · Prediction-error decomposition at the current sidebar variances")
c1, c2 = st.columns([1.3, 1])
with c1:
    ll = out["ll"].copy()
    ll[:burn] = np.nan
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ctx.t, y=ll, name="log p(y_t | y_1:t-1)", marker_color=ui.PALETTE["filter"]))
    fig.add_trace(go.Scatter(x=ctx.t, y=np.nancumsum(ll), name="cumulative", yaxis="y2",
                             line=dict(color=ui.PALETTE["smooth"])))
    ui.layout(fig, h=330, yaxis2=dict(overlaying="y", side="right", showgrid=False))
    ui.show(fig)
with c2:
    df = pd.DataFrame({"t": ctx.t, "y_t": y, "f_t": out["f"], "Q_t": out["Q"], "e_t": out["e"],
                       "log-lik term": out["ll"]})
    st.dataframe(df.style.format({c: "{:.4g}" for c in df.columns if c != "t"}), height=300,
                 width="stretch", hide_index=True)
st.metric("Total log-likelihood (burn-in dropped)", f"{out['loglik']:.3f}")
st.caption("Big negative bars = observations the model found surprising (large e_t relative to √Q_t).")

# ------------------------------------------------------------------ MLE
st.subheader("2 · Maximise it")
sig = (ctx.label, len(y), float(np.nansum(y)))
if st.button("🚀 Fit by maximum likelihood", type="primary"):
    with st.spinner("Optimising…"):
        st.session_state["mle"] = (sig, core.fit_mle(spec, y))
fit = st.session_state.get("mle")
fit = fit[1] if fit and fit[0] == sig else None

if fit:
    tbl = pd.DataFrame({"estimate": fit["est"], "approx. s.e.": fit["se"]}, index=fit["names"])
    if ctx.V_true is not None:
        tbl.insert(0, "true", [ctx.V_true] + list(np.diag(ctx.W_true)[spec.w_free]))
    st.dataframe(tbl.style.format("{:.4g}"), width="stretch")
    st.caption(f"log-lik = {fit['loglik']:.3f} · {fit['nit']} iterations · converged: {fit['success']} "
               f"· signal-to-noise: " + ", ".join(f"{e / fit['est'][0]:.3g}" for e in fit["est"][1:]))
    st.button("📌 Use these estimates in the sidebar (all pages)", on_click=ui.set_inference,
              args=(fit["est"][0], list(fit["est"][1:])))

# ------------------------------------------------------------------ surface
st.subheader("3 · Look at the likelihood surface")
names = ["V (obs.)"] + [f"W: {n}" for n in spec.free_names()]
cur = np.log10([ctx.V] + list(np.maximum(np.diag(ctx.W)[spec.w_free], 1e-12)))
c1, c2, c3 = st.columns(3)
px_ = c1.selectbox("x-axis parameter", names, index=0, key="sx")
py_ = c2.selectbox("y-axis parameter", names, index=1, key="sy")
n = c3.slider("Grid resolution", 11, 41, 21, step=2, key="sn",
              help="Each grid point is one full Kalman filter run.")
span = st.slider("± range around centre (log10 units)", 0.5, 4.0, 2.0, 0.5, key="span")
ix, iy = names.index(px_), names.index(py_)

if ix == iy:
    st.warning("Pick two different parameters.")
else:
    centre = np.log10(fit["est"]) if fit else cur

    @st.cache_data(show_spinner=False)
    def surface(_spec, key, y, centre, ix, iy, n, span):
        gx = np.linspace(centre[ix] - span, centre[ix] + span, n)
        gy = np.linspace(centre[iy] - span, centre[iy] + span, n)
        Z = np.zeros((n, n))
        for a, vy in enumerate(gy):
            for b, vx in enumerate(gx):
                psi = centre.copy()
                psi[ix], psi[iy] = vx, vy
                Z[a, b] = -core.neg_loglik(psi * np.log(10), _spec, y, _spec.p)
        return gx, gy, Z

    with st.spinner(f"Running {n * n} Kalman filters…"):
        gx, gy, Z = surface(spec, ctx.label + str(spec.F.shape), y, centre, ix, iy, n, span)
    Zc = np.clip(Z, Z.max() - 50, None)
    fig = go.Figure(go.Contour(x=gx, y=gy, z=Zc, colorscale="Viridis", ncontours=30,
                               colorbar=dict(title="log-lik"),
                               hovertemplate=f"log10 {px_}=%{{x:.2f}}<br>log10 {py_}=%{{y:.2f}}<br>ll=%{{z:.2f}}<extra></extra>"))
    if fit:
        path = fit["path"] / np.log(10)
        fig.add_trace(go.Scatter(x=path[:, ix], y=path[:, iy], mode="lines+markers", name="optimiser path",
                                 line=dict(color="white"), marker=dict(size=5, color="white")))
        fig.add_trace(go.Scatter(x=[path[-1, ix]], y=[path[-1, iy]], mode="markers", name="MLE",
                                 marker=dict(symbol="star", size=16, color="#ffd60a")))
    if ctx.V_true is not None:
        tv = np.log10([ctx.V_true] + list(np.diag(ctx.W_true)[spec.w_free]))
        fig.add_trace(go.Scatter(x=[tv[ix]], y=[tv[iy]], mode="markers", name="truth",
                                 marker=dict(symbol="x", size=14, color="#ff006e")))
    ui.layout(fig, h=520, xaxis_title=f"log10 {px_}", yaxis_title=f"log10 {py_}")
    fig.update_layout(hovermode="closest")
    ui.show(fig)
    if len(names) > 2:
        st.caption("Other parameters are held at the MLE (or the sidebar values if you haven't fitted). "
                   "This is a slice, not a profile likelihood.")

with st.expander("Cross-check against statsmodels (exact diffuse likelihood)"):
    if spec.name.startswith(("Local level", "Local linear")) and st.button("Run statsmodels"):
        import statsmodels.api as sm
        lvl = "llevel" if spec.p == 1 else "lltrend"
        r = sm.tsa.UnobservedComponents(y, level=lvl).fit(disp=False)
        st.dataframe(pd.DataFrame({"statsmodels": r.params, "s.e.": r.bse}, index=names)
                     .style.format("{:.4g}"), width="stretch")
    elif not spec.name.startswith(("Local level", "Local linear")):
        st.caption("Available for the local level and local linear trend models.")

ui.notice([
    "The surface is often a long, curved ridge: V and W trade off against each other, which is why "
    "SEs can be large and why the optimiser zig-zags.",
    "With short series, a W estimate of ~0 (on the boundary) is common — the 'pile-up' problem.",
    "Fit, then click 'Use these estimates' and revisit the filter and diagnostics pages.",
])
