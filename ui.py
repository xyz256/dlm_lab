"""Shared sidebar, data context and plotting helpers for every page."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dlm import core, models

PALETTE = {
    "obs": "#8a8f98", "true": "#2b2d42", "prior": "#e09f3e", "filter": "#2a6fdb",
    "smooth": "#c0392b", "fcst": "#7b4fd6", "bayes": "#1b9e77",
}
DEFAULT_SIM_W = {"Local level": [0.3], "Local linear trend": [0.3, 0.01],
                 "Level + seasonal": [0.1, 0.05], "Dynamic regression": [0.05, 0.02]}


@dataclass
class Ctx:
    spec: models.DLMSpec
    y: np.ndarray
    t: np.ndarray                 # x-axis labels (time index)
    theta_true: np.ndarray | None
    V_true: float | None
    W_true: np.ndarray | None
    V: float                      # variances currently used for inference
    W: np.ndarray
    label: str


# ------------------------------------------------------------------ data
@st.cache_data
def load_nile():
    try:
        import statsmodels.api as sm
        df = sm.datasets.nile.load_pandas().data
        return df["year"].astype(int).to_numpy(), df["volume"].to_numpy(float)
    except Exception:  # pragma: no cover
        return None, None


@st.cache_data
def simulate_cached(key, T, period, V, W_free, seed):
    spec = models.build(key, T, period)
    W = spec.W_from_vec(W_free)
    theta0 = np.zeros(spec.p)
    theta0[0] = 10.0
    y, th = core.simulate(spec, V, W, theta0=theta0, seed=seed)
    return y, th, spec.F


def _reset_inference(defaults_V, defaults_W):
    st.session_state["V_inf"] = float(defaults_V)
    for i, w in enumerate(defaults_W):
        st.session_state[f"W_inf_{i}"] = float(w)


def set_inference(V, W_vec):
    """Callback used by other pages (e.g. 'use MLE estimates')."""
    _reset_inference(V, W_vec)


def sidebar() -> Ctx:
    sb = st.sidebar
    sb.markdown("### 📦 Data & model")
    source = sb.radio("Data source", ["Simulate", "Nile river (classic)", "Upload CSV"],
                      key="source", horizontal=False)
    model_key = sb.selectbox("Model", models.MODEL_KEYS, key="model_key")
    period = 4
    if model_key == "Level + seasonal":
        period = sb.slider("Season period s", 2, 12, 4, key="period")

    theta_true = V_true = W_true = None
    x_reg = None
    if source == "Simulate":
        T = sb.slider("Length T", 30, 500, 120, step=10, key="T")
        seed = sb.number_input("Random seed", 0, 10_000, 1, key="seed")
        with sb.expander("True variances (data-generating)", expanded=False):
            V_true = st.number_input("True V (observation)", 1e-6, 1e6, 1.0,
                                     format="%.4g", key=f"Vtrue_{model_key}")
            spec_tmp = models.build(model_key, T, period)
            wt = []
            for i, n in enumerate(spec_tmp.free_names()):
                wt.append(st.number_input(f"True W: {n}", 1e-8, 1e6, DEFAULT_SIM_W[model_key][i],
                                          format="%.4g", key=f"Wtrue_{model_key}_{i}"))
        y, theta_true, F = simulate_cached(model_key, T, period, V_true, tuple(wt), int(seed))
        spec = models.build(model_key, T, period)
        if model_key == "Dynamic regression":
            spec = models.dynamic_regression(F[:, 1])
        W_true = spec.W_from_vec(wt)
        t = np.arange(1, T + 1)
        label = f"Simulated · {spec.name}"
        def_V, def_W = V_true, wt
    elif source == "Nile river (classic)":
        years, y = load_nile()
        if y is None:
            st.error("Couldn't load the Nile data (needs statsmodels). Switch source.")
            st.stop()
        T, t = len(y), years
        if model_key == "Dynamic regression":
            x_reg = np.arange(T) / T
        spec = models.build(model_key, T, period, x=x_reg)
        label = f"Nile flow 1871–1970 · {spec.name}"
        def_V = 15099.0
        def_W = [1469.0] + [10.0] * (int(spec.w_free.sum()) - 1)
    else:
        up = sb.file_uploader("CSV file", type="csv", key="csv")
        if up is None:
            st.info("Upload a CSV in the sidebar (one numeric column for y; "
                    "optionally a regressor column).")
            st.stop()
        df = pd.read_csv(up)
        num = df.select_dtypes("number").columns.tolist()
        ycol = sb.selectbox("y column", num, key="ycol")
        y = df[ycol].to_numpy(float)
        T, t = len(y), np.arange(1, len(y) + 1)
        if model_key == "Dynamic regression":
            xcol = sb.selectbox("Regressor column x", [c for c in num if c != ycol], key="xcol")
            x_reg = df[xcol].to_numpy(float)
        spec = models.build(model_key, T, period, x=x_reg)
        label = f"{up.name}: {ycol} · {spec.name}"
        v = float(np.nanvar(np.diff(y[~np.isnan(y)])))
        def_V, def_W = v / 2, [v / 10] * int(spec.w_free.sum())

    # reset the inference variances whenever data/model changes
    sig = (source, model_key, period, T, label, tuple(np.round(np.atleast_1d(def_W), 10)), def_V)
    if st.session_state.get("_sig") != sig or "V_inf" not in st.session_state:
        st.session_state["_sig"] = sig
        _reset_inference(def_V, def_W)

    sb.markdown("### 🎛️ Variances used for inference")
    sb.caption("The filter only knows these. Mis-set them on purpose and watch what breaks.")
    V = sb.number_input("V (observation noise)", 1e-8, 1e9, format="%.4g", key="V_inf")
    wv = [sb.number_input(f"W: {n}", 0.0, 1e9, format="%.4g", key=f"W_inf_{i}")
          for i, n in enumerate(spec.free_names())]
    if theta_true is not None:
        sb.button("↺ Reset to true values", on_click=_reset_inference, args=(def_V, def_W))
    else:
        sb.button("↺ Reset to defaults", on_click=_reset_inference, args=(def_V, def_W))
    sb.caption("Tip: the **Likelihood & MLE** page can fill these with estimates.")
    return Ctx(spec, y, t, theta_true, V_true, W_true, float(V), spec.W_from_vec(wv), label)


def get_ctx() -> Ctx:
    return st.session_state["ctx"]


# ---------------------------------------------------------------- cached runs
@st.cache_data(show_spinner=False)
def run_filter(_spec, spec_key, y, V, W, discount=None):
    # burn = p: drop the first p log-lik terms (approximate diffuse start), as in fit_mle
    return core.kalman_filter(_spec, y, V, W, discount=discount, burn=_spec.p)


def filt(ctx: Ctx, y=None, V=None, W=None, discount=None):
    y = ctx.y if y is None else y
    V = ctx.V if V is None else V
    W = ctx.W if W is None else W
    key = (ctx.label, ctx.spec.F.tobytes().__hash__())
    return run_filter(ctx.spec, key, y, V, W, discount)


# ---------------------------------------------------------------- plotting
def band(fig, x, mean, sd, name, color, z=1.96, show_mean=True, dash=None, opacity=0.18,
         legendgroup=None, row=None, col=None):
    lg = legendgroup or name
    kw = dict(row=row, col=col) if row else {}
    fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]),
                             y=np.concatenate([mean + z * sd, (mean - z * sd)[::-1]]),
                             fill="toself", fillcolor=_rgba(color, opacity), line=dict(width=0),
                             hoverinfo="skip", showlegend=False, legendgroup=lg), **kw)
    if show_mean:
        fig.add_trace(go.Scatter(x=x, y=mean, name=name, legendgroup=lg,
                                 line=dict(color=color, width=2, dash=dash)), **kw)


def _rgba(hex_, a):
    h = hex_.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


def obs_trace(x, y, name="observed y"):
    return go.Scatter(x=x, y=y, mode="markers", name=name,
                      marker=dict(color=PALETTE["obs"], size=5))


def layout(fig, title=None, h=420, **kw):
    fig.update_layout(title=title, height=h, margin=dict(l=10, r=10, t=90 if title else 70, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0),
                      hovermode="x unified", **kw)
    return fig


def focus_y(fig, mean, sd, skip, row=1, col=1):
    """Ignore the vague-prior start when choosing the y-range (otherwise ±1000s squash the plot)."""
    m, s = mean[skip + 2:], sd[skip + 2:]
    if len(m) < 3:
        return
    lo, hi = np.nanmin(m - 2.5 * s), np.nanmax(m + 2.5 * s)
    pad = 0.08 * (hi - lo + 1e-12)
    fig.update_yaxes(range=[lo - pad, hi + pad], row=row, col=col)


def show(fig):
    st.plotly_chart(fig, width="stretch")


def concept(md: str, title="📘 Concept", expanded=False):
    with st.expander(title, expanded=expanded):
        st.markdown(md)


def notice(items):
    st.markdown("**👀 What to notice**\n" + "\n".join(f"- {i}" for i in items))


def matrix_df(M, rows, cols=None, fmt="{:.4g}"):
    cols = rows if cols is None else cols
    return pd.DataFrame(np.atleast_2d(M), index=rows, columns=cols).style.format(fmt)
