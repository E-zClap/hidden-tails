from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import t as student_t


@dataclass(frozen=True)
class Certificate:
    lower: float
    upper: float
    lower_eta: float
    upper_eta: float
    lower_se: float
    upper_se: float
    enlarged_lower: float
    enlarged_upper: float

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def enlarged_width(self) -> float:
        return self.enlarged_upper - self.enlarged_lower


def sensitivity_bounds(p1: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p1, dtype=float)
    gamma = float(gamma)
    lo = p / (p + gamma * (1.0 - p))
    hi = gamma * p / (1.0 - p + gamma * p)
    return lo, hi


def sensitivity_derivatives(p1: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p1, dtype=float)
    gamma = float(gamma)
    dlo = gamma / np.square(gamma + (1.0 - gamma) * p)
    dhi = gamma / np.square(1.0 + (gamma - 1.0) * p)
    return dlo, dhi


def brier_losses(prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = np.clip(np.asarray(prediction, dtype=float), 0.0, 1.0)
    return np.square(f), np.square(1.0 - f)


def empirical_cvar(loss: np.ndarray, alpha: float) -> float:
    values = np.sort(np.asarray(loss, dtype=float))
    if len(values) == 0:
        return float("nan")
    beta = 1.0 - float(alpha)
    candidates = np.unique(np.concatenate(([0.0, 1.0], values)))
    prefix = np.concatenate(([0.0], np.cumsum(values)))
    index = np.searchsorted(values, candidates, side="right")
    excess = (prefix[-1] - prefix[index]) - candidates * (len(values) - index)
    return float(np.min(candidates + excess / (beta * len(values))))


def _endpoint_coefficients(
    prediction: np.ndarray,
    p1: np.ndarray,
    propensity: np.ndarray,
    selected: np.ndarray,
    outcome: np.ndarray,
    gamma: float,
    upper: bool,
    audited: np.ndarray | None,
    orthogonal: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    l0, l1 = brier_losses(prediction)
    p1 = np.asarray(p1, dtype=float)
    pi = np.asarray(propensity, dtype=float)
    s = np.asarray(selected, dtype=bool)
    y = np.nan_to_num(np.asarray(outcome, dtype=float), nan=0.0)
    audit = np.zeros_like(s) if audited is None else np.asarray(audited, dtype=bool)
    if np.any(audit & s):
        audit = audit & ~s
    known = s | audit
    hidden = ~known
    lo, hi = sensitivity_bounds(p1, gamma)
    dlo, dhi = sensitivity_derivatives(p1, gamma)
    label_one_worse = l1 >= l0
    if upper:
        q = np.where(label_one_worse, hi, lo)
        qprime = np.where(label_one_worse, dhi, dlo)
    else:
        q = np.where(label_one_worse, lo, hi)
        qprime = np.where(label_one_worse, dlo, dhi)
    correction = np.zeros_like(p1)
    if orthogonal:
        correction[s] = ((1.0 - pi[s]) / pi[s]) * qprime[s] * (y[s] - p1[s])
    w0 = known * (1.0 - y) + hidden * (1.0 - q) - s * correction
    w1 = known * y + hidden * q + s * correction
    return l0, l1, w0.astype(float), w1.astype(float)


def _minimize_weighted_hinges(
    loss0: np.ndarray,
    loss1: np.ndarray,
    weight0: np.ndarray,
    weight1: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    losses = np.concatenate([loss0, loss1]).astype(float)
    weights = np.concatenate([weight0, weight1]).astype(float)
    order = np.argsort(losses, kind="mergesort")
    losses = losses[order]
    weights = weights[order]
    weighted_losses = weights * losses
    suffix_w = np.concatenate((np.cumsum(weights[::-1])[::-1], [0.0]))
    suffix_wl = np.concatenate((np.cumsum(weighted_losses[::-1])[::-1], [0.0]))
    candidates = np.unique(np.concatenate(([0.0], losses, [1.0])))
    index = np.searchsorted(losses, candidates, side="right")
    excess = suffix_wl[index] - candidates * suffix_w[index]
    objectives = candidates + excess / ((1.0 - float(alpha)) * len(loss0))
    best = int(np.nanargmin(objectives))
    return float(objectives[best]), float(candidates[best])


def _fold_standard_error(
    eta: float,
    loss0: np.ndarray,
    loss1: np.ndarray,
    weight0: np.ndarray,
    weight1: np.ndarray,
    alpha: float,
    folds: np.ndarray,
) -> float:
    score = weight0 * np.maximum(loss0 - eta, 0.0) + weight1 * np.maximum(loss1 - eta, 0.0)
    unique = np.unique(folds)
    values = np.array([eta + np.mean(score[folds == fold]) / (1.0 - alpha) for fold in unique])
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def estimate_certificate(
    prediction: np.ndarray,
    p1: np.ndarray,
    propensity: np.ndarray,
    selected: np.ndarray,
    outcome: np.ndarray,
    alpha: float,
    gamma: float,
    folds: np.ndarray | None = None,
    audited: np.ndarray | None = None,
    confidence_level: float = 0.95,
    orthogonal: bool = True,
) -> Certificate:
    endpoint_data = {}
    for upper in (False, True):
        arrays = _endpoint_coefficients(
            prediction, p1, propensity, selected, outcome, gamma, upper, audited, orthogonal
        )
        value, eta = _minimize_weighted_hinges(*arrays, alpha)
        se = 0.0 if folds is None else _fold_standard_error(eta, *arrays, alpha, np.asarray(folds))
        endpoint_data[upper] = (value, eta, se)
    lower, lower_eta, lower_se = endpoint_data[False]
    upper, upper_eta, upper_se = endpoint_data[True]
    if folds is None or len(np.unique(folds)) < 2:
        multiplier = 0.0
    else:
        multiplier = float(student_t.ppf(0.5 + confidence_level / 2.0, len(np.unique(folds)) - 1))
    enlarged_lower = float(np.clip(lower - multiplier * lower_se, 0.0, 1.0))
    enlarged_upper = float(np.clip(upper + multiplier * upper_se, 0.0, 1.0))
    lower = float(np.clip(lower, 0.0, 1.0))
    upper = float(np.clip(upper, 0.0, 1.0))
    if lower > upper:
        midpoint = 0.5 * (lower + upper)
        lower = upper = midpoint
    return Certificate(
        lower=lower,
        upper=upper,
        lower_eta=lower_eta,
        upper_eta=upper_eta,
        lower_se=lower_se,
        upper_se=upper_se,
        enlarged_lower=min(enlarged_lower, enlarged_upper),
        enlarged_upper=max(enlarged_lower, enlarged_upper),
    )


def tail_identification_value(
    prediction: np.ndarray,
    p1: np.ndarray,
    propensity: np.ndarray,
    eta: float,
    alpha: float,
    gamma: float,
    conditional_on_hidden: bool = False,
) -> np.ndarray:
    l0, l1 = brier_losses(prediction)
    a0 = np.maximum(l0 - eta, 0.0)
    a1 = np.maximum(l1 - eta, 0.0)
    lo, hi = sensitivity_bounds(p1, gamma)
    mass = 1.0 if conditional_on_hidden else (1.0 - np.asarray(propensity, dtype=float))
    return mass * (hi - lo) * np.abs(a1 - a0) / (1.0 - alpha)

