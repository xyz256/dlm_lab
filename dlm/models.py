"""Model specifications for univariate-observation DLMs.

    y_t     = F_t' theta_t + v_t,      v_t ~ N(0, V)
    theta_t = G theta_{t-1} + w_t,     w_t ~ N(0, W),  W = diag(w)

A spec stores F (T x p), G (p x p), which diagonal entries of W are free
parameters, and human-readable names for the state components.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DLMSpec:
    name: str
    F: np.ndarray            # (T, p) design vectors, one row per time step
    G: np.ndarray            # (p, p) evolution matrix
    w_free: np.ndarray       # (p,) bool: which W diagonal entries are estimated
    state_names: list[str]
    description: str = ""
    obs_combo: list[int] = field(default_factory=list)  # states summed for the "signal" plot

    @property
    def p(self) -> int:
        return self.G.shape[0]

    @property
    def T(self) -> int:
        return self.F.shape[0]

    def W_from_vec(self, w_vec) -> np.ndarray:
        """Expand the free W variances into a full diagonal W matrix."""
        w = np.zeros(self.p)
        w[self.w_free] = np.asarray(w_vec, dtype=float)
        return np.diag(w)

    def free_names(self) -> list[str]:
        return [n for n, f in zip(self.state_names, self.w_free) if f]


def local_level(T: int) -> DLMSpec:
    return DLMSpec(
        name="Local level (random walk + noise)",
        F=np.ones((T, 1)),
        G=np.array([[1.0]]),
        w_free=np.array([True]),
        state_names=["level μ"],
        description="μ_t = μ_{t-1} + w_t ;  y_t = μ_t + v_t. The simplest DLM; "
                    "the Kalman filter here is exponential smoothing with a data-driven weight.",
        obs_combo=[0],
    )


def local_linear_trend(T: int) -> DLMSpec:
    return DLMSpec(
        name="Local linear trend",
        F=np.tile([1.0, 0.0], (T, 1)),
        G=np.array([[1.0, 1.0], [0.0, 1.0]]),
        w_free=np.array([True, True]),
        state_names=["level μ", "slope β"],
        description="μ_t = μ_{t-1} + β_{t-1} + w₁ ;  β_t = β_{t-1} + w₂. "
                    "Level and slope both drift; G couples them.",
        obs_combo=[0],
    )


def level_plus_seasonal(T: int, period: int = 4) -> DLMSpec:
    s = period
    p = 1 + (s - 1)
    G = np.zeros((p, p))
    G[0, 0] = 1.0
    # dummy-variable seasonal: γ_t = -(γ_{t-1} + ... + γ_{t-s+1}) + w
    G[1, 1:] = -1.0
    for i in range(2, p):
        G[i, i - 1] = 1.0
    F = np.zeros((T, p))
    F[:, 0] = 1.0
    F[:, 1] = 1.0
    w_free = np.zeros(p, dtype=bool)
    w_free[[0, 1]] = True
    names = ["level μ", "season γ"] + [f"γ lag {i}" for i in range(1, s - 1)]
    return DLMSpec(
        name=f"Local level + seasonal (period {s})",
        F=F, G=G, w_free=w_free, state_names=names,
        description="Seasonal effects sum to (approximately) zero over a cycle; the "
                    "lagged seasonal states are bookkeeping that G shifts forward each step.",
        obs_combo=[0, 1],
    )


def dynamic_regression(x: np.ndarray) -> DLMSpec:
    x = np.asarray(x, dtype=float)
    T = len(x)
    F = np.column_stack([np.ones(T), x])
    return DLMSpec(
        name="Dynamic regression (time-varying α, β)",
        F=F, G=np.eye(2), w_free=np.array([True, True]),
        state_names=["intercept α", "coefficient β"],
        description="y_t = α_t + β_t x_t + v_t with α, β random walks. "
                    "F_t changes every step because it contains the regressor x_t.",
        obs_combo=[],
    )


MODEL_KEYS = ["Local level", "Local linear trend", "Level + seasonal", "Dynamic regression"]


def build(key: str, T: int, period: int = 4, x: np.ndarray | None = None) -> DLMSpec:
    if key == "Local level":
        return local_level(T)
    if key == "Local linear trend":
        return local_linear_trend(T)
    if key == "Level + seasonal":
        return level_plus_seasonal(T, period)
    if key == "Dynamic regression":
        if x is None:
            x = np.sin(np.arange(T) / 6.0) + 0.3 * np.random.default_rng(0).standard_normal(T)
        return dynamic_regression(x)
    raise ValueError(key)
