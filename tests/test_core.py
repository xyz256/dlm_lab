"""Cross-check the hand-written DLM code against statsmodels and known results."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dlm import core, models  # noqa: E402

sm = pytest.importorskip("statsmodels.api")


def _known_init_uc(y, level, trend, V, W, m0, C0):
    kw = dict(level="llevel" if not trend else "lltrend")
    mod = sm.tsa.UnobservedComponents(y, **kw)
    mod.ssm.initialize_known(m0, C0)
    params = [V] + list(np.diag(W)[: (2 if trend else 1)])
    return mod.smooth(params)


@pytest.mark.parametrize("trend", [False, True])
def test_filter_and_smoother_match_statsmodels(trend):
    T = 80
    spec = models.local_linear_trend(T) if trend else models.local_level(T)
    W = np.diag([0.5, 0.05]) if trend else np.array([[0.8]])
    y, _ = core.simulate(spec, V=2.0, W=W, seed=3)
    m0, C0 = np.zeros(spec.p), np.eye(spec.p) * 10.0
    ours = core.kalman_filter(spec, y, 2.0, W, m0=m0, C0=C0)
    # statsmodels' initial state is theta_1 itself; ours is theta_0 with theta_1 = G theta_0 + w.
    # Make them comparable: our prior for theta_1 is (G m0, G C0 G' + W).
    a1, R1 = spec.G @ m0, spec.G @ C0 @ spec.G.T + W
    sm_res = _known_init_uc(y, True, trend, 2.0, W, a1, R1)
    np.testing.assert_allclose(ours["m"], sm_res.filtered_state.T, rtol=1e-6, atol=1e-8)
    # statsmodels' .llf drops burn-in steps; compare the full per-step sum
    np.testing.assert_allclose(ours["loglik"], sm_res.llf_obs.sum(), rtol=1e-8)
    sm_ = core.rts_smoother(spec, ours)
    np.testing.assert_allclose(sm_["s"], sm_res.smoothed_state.T, rtol=1e-6, atol=1e-8)


def test_missing_data_handled():
    spec = models.local_level(50)
    y, _ = core.simulate(spec, 1.0, np.array([[0.3]]), seed=1)
    y[10:15] = np.nan
    out = core.kalman_filter(spec, y, 1.0, np.array([[0.3]]))
    assert np.all(np.isfinite(out["m"]))
    # variance grows through the gap
    assert out["C"][14, 0, 0] > out["C"][9, 0, 0]


def test_steady_state_gain():
    spec = models.local_level(400)
    y, _ = core.simulate(spec, 4.0, np.array([[1.0]]), seed=2)
    out = core.kalman_filter(spec, y, 4.0, np.array([[1.0]]))
    K_inf, _ = core.steady_state_gain_local_level(4.0, 1.0)
    assert abs(out["K"][-1, 0] - K_inf) < 1e-8


def test_mle_recovers_parameters():
    spec = models.local_level(600)
    y, _ = core.simulate(spec, 2.0, np.array([[0.5]]), seed=7)
    fit = core.fit_mle(spec, y)
    assert fit["success"]
    np.testing.assert_allclose(fit["est"], [2.0, 0.5], rtol=0.35)
    # compare to statsmodels' exact-diffuse MLE
    res = sm.tsa.UnobservedComponents(y, level="llevel").fit(disp=False)
    np.testing.assert_allclose(fit["est"], res.params, rtol=0.05)


def test_gibbs_runs_and_is_reasonable():
    spec = models.local_level(200)
    y, _ = core.simulate(spec, 1.0, np.array([[0.2]]), seed=5)
    g = core.gibbs(spec, y, n_iter=400, seed=0)
    assert 0.5 < np.median(g["V"][100:]) < 1.8
    assert 0.05 < np.median(g["W"][100:, 0]) < 0.6


def test_seasonal_and_regression_models_filter():
    for spec in [models.level_plus_seasonal(60, 4), models.dynamic_regression(np.linspace(-1, 1, 60))]:
        W = spec.W_from_vec([0.1] * spec.w_free.sum())
        y, _ = core.simulate(spec, 0.5, W, seed=0)
        out = core.kalman_filter(spec, y, 0.5, W)
        sm_ = core.rts_smoother(spec, out)
        assert np.isfinite(out["loglik"]) and np.all(np.isfinite(sm_["s"]))
        th = core.ffbs(spec, out, np.random.default_rng(0))
        assert th.shape == (60, spec.p)


def test_discount_filter():
    spec = models.local_level(100)
    y, _ = core.simulate(spec, 1.0, np.array([[0.2]]), seed=0)
    out = core.kalman_filter(spec, y, 1.0, None, discount=0.9)
    # with discounting W_t/C_{t-1} is (1-δ)/δ
    ratio = out["Wt"][50, 0, 0] / out["C"][49, 0, 0]
    assert abs(ratio - (0.1 / 0.9)) < 1e-10
