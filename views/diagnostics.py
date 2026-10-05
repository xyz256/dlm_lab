import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

import ui

ctx = ui.get_ctx()
st.title("🩺 Diagnostics: is the model right?")
st.caption(ctx.label)

ui.concept(r"""
If the model (including $V$ and $W$) is correct, the **standardized one-step forecast errors**
$$
\tilde e_t = \frac{y_t - f_t}{\sqrt{Q_t}}
$$
are i.i.d. $N(0,1)$. That gives three checks:

1. **Independence** — the ACF should sit inside $\pm 1.96/\sqrt{T}$; the Ljung–Box test summarises it.
2. **Normality** — QQ plot close to the line; Jarque–Bera / Shapiro tests.
3. **Constant variance** — no fanning out over time; compare the first and last thirds.

Mis-specification leaves fingerprints: **W too small** → the filter adapts too slowly → errors are
positively autocorrelated. **W too large** → over-reacts → negative lag-1 autocorrelation.
""")

st.markdown("##### Stress-test: scale the inference W by a factor")
fac = st.select_slider("W × factor", options=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
                       value=1.0, key="wfac")
out = ui.filt(ctx, W=ctx.W * fac)
r = out["std_resid"].copy()
r[: ctx.spec.p] = np.nan
ok = ~np.isnan(r)
rr = r[ok]
T = len(rr)

maxlag = min(24, T // 3)
rc = rr - rr.mean()
acf = np.array([np.sum(rc[k:] * rc[:T - k]) / np.sum(rc * rc) for k in range(1, maxlag + 1)])
Qlb = T * (T + 2) * np.cumsum(acf ** 2 / (T - np.arange(1, maxlag + 1)))
lb_p = 1 - stats.chi2.cdf(Qlb, np.arange(1, maxlag + 1))
jb = stats.jarque_bera(rr)
third = T // 3
h_ratio = np.sum(rr[-third:] ** 2) / np.sum(rr[:third] ** 2)
h_p = 2 * min(stats.f.cdf(h_ratio, third, third), 1 - stats.f.cdf(h_ratio, third, third))

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Ljung–Box p (lag {min(10, maxlag)})", f"{lb_p[min(10, maxlag) - 1]:.3f}")
c2.metric("Jarque–Bera p (normality)", f"{jb.pvalue:.3f}")
c3.metric("Heteroskedasticity H p", f"{h_p:.3f}", help="Var(last third) / Var(first third) ~ F")
c4.metric("Log-likelihood", f"{out['loglik']:.2f}")
st.caption("Small p-values (< 0.05) flag a problem.")

fig = make_subplots(rows=2, cols=2, subplot_titles=["Standardized errors over time", "ACF",
                                                     "Normal QQ plot", "Histogram vs N(0,1)"],
                    vertical_spacing=0.14)
fig.add_trace(go.Scatter(x=ctx.t[ok], y=rr, mode="lines+markers", marker=dict(size=3),
                         line=dict(color=ui.PALETTE["filter"], width=1), name="ẽ_t"), row=1, col=1)
for v in (-1.96, 1.96):
    fig.add_hline(y=v, line=dict(dash="dot", color="#999"), row=1, col=1)
fig.add_trace(go.Bar(x=np.arange(1, maxlag + 1), y=acf, marker_color=ui.PALETTE["fcst"], name="ACF"),
              row=1, col=2)
for v in (-1.96 / np.sqrt(T), 1.96 / np.sqrt(T)):
    fig.add_hline(y=v, line=dict(dash="dot", color="#999"), row=1, col=2)
(osm, osr), (sl, ic, _) = stats.probplot(rr)
fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", marker=dict(size=4, color=ui.PALETTE["filter"]),
                         name="QQ"), row=2, col=1)
fig.add_trace(go.Scatter(x=osm, y=sl * osm + ic, line=dict(color=ui.PALETTE["smooth"]), name="line"),
              row=2, col=1)
fig.add_trace(go.Histogram(x=rr, histnorm="probability density", nbinsx=25,
                           marker_color=ui.PALETTE["obs"], name="hist"), row=2, col=2)
g = np.linspace(-4, 4, 200)
fig.add_trace(go.Scatter(x=g, y=stats.norm.pdf(g), line=dict(color=ui.PALETTE["smooth"]), name="N(0,1)"),
              row=2, col=2)
ui.layout(fig, h=620, showlegend=False)
fig.update_layout(hovermode="closest")
ui.show(fig)

ui.notice([
    "At factor 1 with simulated data, everything should look like white noise (p-values mostly > 0.05).",
    "Factor 0.01: the ACF shows a slow positive decay — the filter is chasing a moving level too lazily.",
    "Factor 100: the lag-1 bar goes negative — the filter over-reacts to every observation.",
    "On the Nile data, the outlier-ish years and the 1899 dam break show up as large standardized errors.",
])
