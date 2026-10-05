"""Estimation and inference for univariate-observation DLMs, written for reading.

Notation follows West & Harrison / Petris et al.:
    prior        theta_t | y_{1:t-1} ~ N(a_t, R_t)
    one-step     y_t     | y_{1:t-1} ~ N(f_t, Q_t)
    posterior    theta_t | y_{1:t}   ~ N(m_t, C_t)
    smoothed     theta_t | y_{1:T}   ~ N(s_t, S_t)
"""
from __future__ import annotations

import numpy as np
from scipy import optimize

from .models import DLMSpec

LOG2PI = np.log(2 * np.pi)


# --------------------------------------------------------------------------- utils
def _sym(A):
    return 0.5 * (A + A.T)


def mvn_draw(mean, cov, rng):
    """Draw from N(mean, cov) for a possibly singular PSD cov (eigen-decomposition)."""
    vals, vecs = np.linalg.eigh(_sym(cov))
    vals = np.clip(vals, 0.0, None)
    return mean + vecs @ (np.sqrt(vals) * rng.standard_normal(len(mean)))


def default_prior(spec: DLMSpec, y, scale=1e6):
    """Vague prior: centre the first state at the first observation, huge variance."""
    m0 = np.zeros(spec.p)
    first = np.asarray(y)[~np.isnan(y)]
    if len(first):
        m0[0] = first[0]
    return m0, np.eye(spec.p) * scale


# ---------------------------------------------------------------------- simulate
def simulate(spec: DLMSpec, V: float, W: np.ndarray, theta0=None, seed: int = 0):
    rng = np.random.default_rng(seed)
    T, p = spec.T, spec.p
    theta = np.zeros((T, p))
    prev = np.zeros(p) if theta0 is None else np.asarray(theta0, float)
    for t in range(T):
        prev = spec.G @ prev + mvn_draw(np.zeros(p), W, rng)
        theta[t] = prev
    y = np.einsum("tp,tp->t", spec.F, theta) + np.sqrt(V) * rng.standard_normal(T)
    return y, theta


# ------------------------------------------------------------------ Kalman filter
def kalman_filter(spec: DLMSpec, y, V, W, m0=None, C0=None, discount=None, burn=0):
    """Forward filtering. Missing observations (NaN) skip the update step.

    If ``discount`` (0 < δ ≤ 1) is given, W is ignored and set per step as
    W_t = (1-δ)/δ · G C_{t-1} G'  (West & Harrison discounting).
    ``burn`` drops the first steps from the log-likelihood (approximate diffuse start).
    """
    y = np.asarray(y, float)
    T, p = len(y), spec.p
    if m0 is None or C0 is None:
        m0_, C0_ = default_prior(spec, y)
        m0 = m0_ if m0 is None else m0
        C0 = C0_ if C0 is None else C0
    G = spec.G
    out = {k: np.full((T, p), np.nan) for k in ("a", "m", "K")}
    out.update({k: np.full((T, p, p), np.nan) for k in ("R", "C", "Wt")})
    out.update({k: np.full(T, np.nan) for k in ("f", "Q", "e", "ll")})

    m, C = np.asarray(m0, float), np.asarray(C0, float)
    for t in range(T):
        # ---- predict (time update)
        a = G @ m
        P = G @ C @ G.T
        Wt = (1 - discount) / discount * P if discount else W
        R = _sym(P + Wt)
        Ft = spec.F[t]
        f = Ft @ a
        Q = Ft @ R @ Ft + V
        out["a"][t], out["R"][t], out["f"][t], out["Q"][t], out["Wt"][t] = a, R, f, Q, Wt

        if np.isnan(y[t]):          # no data: posterior = prior
            m, C = a, R
        else:                        # ---- update (measurement update)
            e = y[t] - f
            K = R @ Ft / Q           # Kalman gain (adaptive coefficient A_t)
            m = a + K * e
            IKF = np.eye(p) - np.outer(K, Ft)
            C = _sym(IKF @ R @ IKF.T + V * np.outer(K, K))   # Joseph form
            out["e"][t], out["K"][t] = e, K
            out["ll"][t] = -0.5 * (LOG2PI + np.log(Q) + e * e / Q)
        out["m"][t], out["C"][t] = m, C

    ll = out["ll"].copy()
    ll[:burn] = np.nan
    out["loglik"] = float(np.nansum(ll))
    out["m0"], out["C0"] = np.asarray(m0, float), np.asarray(C0, float)
    out["std_resid"] = out["e"] / np.sqrt(out["Q"])
    return out


# ----------------------------------------------------------------- RTS smoother
def rts_smoother(spec: DLMSpec, filt):
    """Rauch–Tung–Striebel backward pass. Also returns smoother gains B_t."""
    m, C, a, R = filt["m"], filt["C"], filt["a"], filt["R"]
    T, p = m.shape
    s, S, B = np.zeros_like(m), np.zeros_like(C), np.zeros_like(C)
    s[-1], S[-1] = m[-1], C[-1]
    for t in range(T - 2, -1, -1):
        Bt = C[t] @ spec.G.T @ np.linalg.pinv(R[t + 1])
        s[t] = m[t] + Bt @ (s[t + 1] - a[t + 1])
        S[t] = _sym(C[t] + Bt @ (S[t + 1] - R[t + 1]) @ Bt.T)
        B[t] = Bt
    return {"s": s, "S": S, "B": B}


# ------------------------------------------------------------------------ FFBS
def ffbs(spec: DLMSpec, filt, rng):
    """Forward-filtering backward-sampling: one joint draw of theta_{1:T} | y."""
    m, C, a, R = filt["m"], filt["C"], filt["a"], filt["R"]
    T, p = m.shape
    th = np.zeros((T, p))
    th[-1] = mvn_draw(m[-1], C[-1], rng)
    for t in range(T - 2, -1, -1):
        Bt = C[t] @ spec.G.T @ np.linalg.pinv(R[t + 1])
        h = m[t] + Bt @ (th[t + 1] - a[t + 1])
        H = C[t] - Bt @ R[t + 1] @ Bt.T
        th[t] = mvn_draw(h, H, rng)
    return th


# --------------------------------------------------------------------- forecast
def forecast(spec: DLMSpec, filt, V, W, h: int, F_future=None):
    """k-step-ahead forecasts from the end of the filter: a_t(k), R_t(k), f_t(k), Q_t(k)."""
    m, C = filt["m"][-1], filt["C"][-1]
    if F_future is None:
        F_future = np.tile(spec.F[-1], (h, 1))
    f, Q, a_out = np.zeros(h), np.zeros(h), np.zeros((h, spec.p))
    a, R = m, C
    for k in range(h):
        a = spec.G @ a
        R = _sym(spec.G @ R @ spec.G.T + W)
        f[k] = F_future[k] @ a
        Q[k] = F_future[k] @ R @ F_future[k] + V
        a_out[k] = a
    return {"f": f, "Q": Q, "a": a_out}


# --------------------------------------------------------------------------- MLE
def unpack(spec, psi):
    psi = np.asarray(psi, float)
    return float(np.exp(psi[0])), spec.W_from_vec(np.exp(psi[1:]))


def neg_loglik(psi, spec, y, burn):
    V, W = unpack(spec, psi)
    return -kalman_filter(spec, y, V, W, burn=burn)["loglik"]


def fit_mle(spec: DLMSpec, y, psi0=None, burn=None, track=True):
    """Maximise the prediction-error-decomposition likelihood over log-variances.

    Returns estimates, approximate SEs (inverse numerical Hessian, on the log scale
    then delta-method to variances) and the optimiser path for plotting.
    """
    y = np.asarray(y, float)
    burn = spec.p if burn is None else burn
    k = 1 + int(spec.w_free.sum())
    if psi0 is None:
        v = np.nanvar(np.diff(y[~np.isnan(y)]))
        psi0 = np.full(k, np.log(max(v, 1e-8) / 2))
    path = [np.array(psi0, float)]
    cb = (lambda xk: path.append(np.array(xk))) if track else None
    res = optimize.minimize(neg_loglik, psi0, args=(spec, y, burn),
                            method="L-BFGS-B", bounds=[(-25, 25)] * k, callback=cb)
    H = numerical_hessian(lambda z: neg_loglik(z, spec, y, burn), res.x)
    try:
        cov_psi = np.linalg.inv(H)
        se_psi = np.sqrt(np.clip(np.diag(cov_psi), 0, None))
    except np.linalg.LinAlgError:
        se_psi = np.full(k, np.nan)
    est = np.exp(res.x)
    return {
        "psi": res.x, "est": est, "se": est * se_psi, "se_psi": se_psi,
        "loglik": -res.fun, "success": bool(res.success), "message": str(res.message),
        "nit": int(res.nit), "path": np.array(path), "burn": burn,
        "names": ["V (obs.)"] + [f"W: {n}" for n in spec.free_names()],
    }


def numerical_hessian(fun, x, eps=1e-4):
    x = np.asarray(x, float)
    k = len(x)
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            ei, ej = np.eye(k)[i] * eps, np.eye(k)[j] * eps
            H[i, j] = H[j, i] = (fun(x + ei + ej) - fun(x + ei - ej)
                                 - fun(x - ei + ej) + fun(x - ei - ej)) / (4 * eps * eps)
    return H


# ------------------------------------------------------------------------- Gibbs
def gibbs(spec: DLMSpec, y, n_iter=1000, a0=2.0, b0=None, seed=0, init=None, progress=None):
    """Gibbs sampler: theta | V,W via FFBS; V, W_ii | theta via inverse-gamma conjugacy.

    Priors: V ~ IG(a0, b0), W_ii ~ IG(a0, b0) for each free component.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    obs = ~np.isnan(y)
    T, p = spec.T, spec.p
    if b0 is None:
        b0 = float(np.nanvar(np.diff(y[obs]))) * (a0 - 1)   # prior mean ≈ var of diffs
    free = np.where(spec.w_free)[0]
    V, wv = (init if init is not None else (np.nanvar(y) / 2, np.full(len(free), np.nanvar(y) / 10)))
    wv = np.asarray(wv, float)
    draws_V, draws_W = np.zeros(n_iter), np.zeros((n_iter, len(free)))
    sum_th, sum_th2 = np.zeros((T, p)), np.zeros((T, p))
    keep_paths = []
    for it in range(n_iter):
        W = spec.W_from_vec(wv)
        filt = kalman_filter(spec, y, V, W)
        th = ffbs(spec, filt, rng)
        # V | theta
        resid = y[obs] - np.einsum("tp,tp->t", spec.F[obs], th[obs])
        V = 1.0 / rng.gamma(a0 + obs.sum() / 2, 1.0 / (b0 + 0.5 * resid @ resid))
        # W_ii | theta  (first state uses the prior mean as theta_0 ~ m0)
        prev = np.vstack([filt["m0"], th[:-1]])
        innov = th - prev @ spec.G.T
        for j, i in enumerate(free):
            ss = innov[1:, i] @ innov[1:, i]
            wv[j] = 1.0 / rng.gamma(a0 + (T - 1) / 2, 1.0 / (b0 + 0.5 * ss))
        draws_V[it], draws_W[it] = V, wv
        sum_th += th
        sum_th2 += th ** 2
        if it % max(1, n_iter // 20) == 0:
            keep_paths.append(th[:, 0].copy())
        if progress is not None:
            progress(it + 1, n_iter)
    mean = sum_th / n_iter
    sd = np.sqrt(np.clip(sum_th2 / n_iter - mean ** 2, 0, None))
    return {"V": draws_V, "W": draws_W, "theta_mean": mean, "theta_sd": sd,
            "paths": np.array(keep_paths), "b0": b0, "a0": a0,
            "names": ["V (obs.)"] + [f"W: {n}" for n in spec.free_names()]}


# ---------------------------------------------------------------- steady state
def steady_state_gain_local_level(V, W):
    """Closed-form limiting gain for the local level model (signal-to-noise r = W/V)."""
    r = W / V
    R = 0.5 * (W + np.sqrt(W * W + 4 * W * V))     # limiting prior variance
    return R / (R + V), r
