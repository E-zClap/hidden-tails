from __future__ import annotations

from typing import Any
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.model_selection import train_test_split

from .certificates import brier_losses, empirical_cvar, estimate_certificate, sensitivity_bounds, tail_identification_value
from .models import fit_probability_model

def _zscore(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    median = float(np.nanmedian(values))
    values = np.nan_to_num(values, nan=median)
    scale = float(np.std(values))
    return np.zeros_like(values) if scale <= 1e-12 else (values - float(np.mean(values))) / scale


def masking_probabilities(data: pd.DataFrame, gamma0: float, target_fraction: float) -> tuple[np.ndarray, float, np.ndarray]:
    """Known logit P(S=1|X,Y)=g(X)-log(Gamma0)Y mechanism."""
    y = data["Y"].to_numpy(dtype=float)
    base = 0.65 * _zscore(data["risk_score"]) - 0.35 * _zscore(data["dti"]) + 0.15 * _zscore(
        np.log1p(data["requested_amount"])
    )
    base = np.clip(base, -3.0, 3.0)
    delta = -float(np.log(gamma0))

    def mean_difference(intercept: float) -> float:
        probability = 1.0 / (1.0 + np.exp(-(base + intercept + delta * y)))
        return float(np.mean(probability) - target_fraction)

    intercept = float(brentq(mean_difference, -20.0, 20.0))
    probability = 1.0 / (1.0 + np.exp(-(base + intercept + delta * y)))
    return probability, intercept, base


def impose_mask(data: pd.DataFrame, cfg: dict[str, Any], repetition: int) -> tuple[pd.DataFrame, dict[str, float]]:
    gamma0 = float(cfg["part_b"]["gamma0"])
    target = float(cfg["part_b"]["target_selected_fraction"])
    probability, intercept, _ = masking_probabilities(data, gamma0, target)
    rng = np.random.default_rng(int(cfg["seed"]) + 7919 * (repetition + 1))
    selected = rng.binomial(1, probability).astype(np.int8)
    if selected.sum() < 10 or (1 - selected).sum() < 10:
        raise RuntimeError("Controlled mask produced a degenerate selected/hidden split.")
    masked = data.copy().reset_index(drop=True)
    masked["S"] = selected
    metadata = {
        "gamma0": gamma0,
        "delta": -float(np.log(gamma0)),
        "intercept": intercept,
        "target_selected_fraction": target,
        "realized_selected_fraction": float(selected.mean()),
    }
    return masked, metadata


def _audit_cost(data: pd.DataFrame, cfg: dict[str, Any]) -> np.ndarray:
    weight = float(cfg["part_b"].get("audit_cost_amount_weight", 0.0))
    if weight == 0.0:
        return np.ones(len(data), dtype=float)
    amount = _zscore(np.log1p(data["requested_amount"]))
    return np.maximum(0.1, 1.0 + weight * amount)


def _adaptive_policy(
    train: pd.DataFrame,
    test: pd.DataFrame,
    initial_selected: np.ndarray,
    historical_p1: np.ndarray,
    historical_pi: np.ndarray,
    policy: str,
    cfg: dict[str, Any],
    repetition: int,
) -> list[dict[str, Any]]:
    y_train = train["Y"].to_numpy(dtype=float)
    y_test = test["Y"].to_numpy(dtype=float)
    audited = np.zeros(len(train), dtype=bool)
    original_hidden = ~initial_selected
    initial_hidden_count = int(original_hidden.sum())
    batch_size = max(1, int(np.ceil(float(cfg["part_b"]["batch_fraction_hidden"]) * initial_hidden_count)))
    costs = _audit_cost(train, cfg)
    rows: list[dict[str, Any]] = []
    alpha = max(map(float, cfg["evaluation"]["alphas"]))
    gamma = float(cfg["part_b"]["gamma0"])
    pi_lo, pi_hi = map(float, cfg["evaluation"]["propensity_clip"])
    for round_id in range(int(cfg["part_b"]["adaptive_rounds"]) + 1):
        known = initial_selected | audited
        model_kind = str(cfg["part_b"]["adaptive_model"])
        prediction_train, fitted = fit_probability_model(
            model_kind,
            cfg,
            train.loc[known],
            y_train[known].astype(np.int8),
            train,
            int(cfg["seed"]) + repetition * 100003 + round_id * 101,
        )
        prediction_test = fitted.predict_proba(test)[:, 1]
        known_pi_raw, _ = fit_probability_model(
            str(cfg["evaluation"]["nuisance_propensity_model"]),
            cfg,
            train,
            known.astype(np.int8),
            train,
            int(cfg["seed"]) + repetition * 100003 + round_id * 101 + 1,
        )
        known_outcome_raw, _ = fit_probability_model(
            str(cfg["evaluation"]["nuisance_outcome_model"]),
            cfg,
            train.loc[known],
            y_train[known].astype(np.int8),
            train,
            int(cfg["seed"]) + repetition * 100003 + round_id * 101 + 2,
        )
        known_pi = np.clip(known_pi_raw, pi_lo, pi_hi)
        cert = estimate_certificate(
            prediction_train,
            historical_p1,
            historical_pi,
            initial_selected,
            y_train,
            alpha,
            gamma,
            audited=audited,
            folds=None,
            orthogonal=False,
        )
        test_loss = np.square(prediction_test - y_test)
        test_cvar = empirical_cvar(test_loss, alpha)
        rows.append(
            {
                "repetition": repetition,
                "policy": policy,
                "round": round_id,
                "alpha": alpha,
                "gamma": gamma,
                "audited_labels": int(audited.sum()),
                "audited_fraction_hidden": float(audited.sum() / initial_hidden_count),
                "cumulative_cost": float(costs[audited].sum()),
                "estimated_retained_lower": cert.lower,
                "estimated_retained_upper": cert.upper,
                "estimated_retained_width": cert.width,
                "true_test_cvar": test_cvar,
                "mean_test_brier": float(np.mean(test_loss)),
                "estimator": "retained_plugin",
                "n_train": len(train),
                "n_test": len(test),
                "upper_eta": cert.upper_eta,
            }
        )
        hidden_now = original_hidden & ~audited
        if round_id == int(cfg["part_b"]["adaptive_rounds"]) or not hidden_now.any():
            continue
        rng = np.random.default_rng(
            int(cfg["seed"]) + repetition * 10007 + round_id * 503 + list(cfg["part_b"]["policies"]).index(policy)
        )
        if policy == "tiv_per_cost":
            score = tail_identification_value(
                prediction_train,
                historical_p1,
                historical_pi,
                cert.upper_eta,
                alpha,
                gamma,
                conditional_on_hidden=True,
            ) / costs
        elif policy == "random":
            score = rng.random(len(train))
        elif policy == "uncertainty":
            score = 1.0 - 2.0 * np.abs(prediction_train - 0.5)
        elif policy == "low_propensity":
            score = -known_pi
        elif policy == "predicted_loss":
            l0, l1 = brier_losses(prediction_train)
            score = (1.0 - known_outcome_raw) * l0 + known_outcome_raw * l1
        elif policy == "sensitivity_width":
            lo, hi = sensitivity_bounds(historical_p1, gamma)
            score = hi - lo
        else:
            raise ValueError(f"Unknown audit policy: {policy}")
        eligible = np.flatnonzero(hidden_now)
        take = min(batch_size, len(eligible))
        chosen = eligible[np.argpartition(score[eligible], -take)[-take:]]
        audited[chosen] = True
    return rows


def _adaptive_repetition(masked: pd.DataFrame, cfg: dict[str, Any], repetition: int) -> list[dict[str, Any]]:
    index = np.arange(len(masked))
    train_idx, test_idx = train_test_split(
        index,
        test_size=float(cfg["part_b"]["test_fraction"]),
        stratify=2 * masked["S"].to_numpy(dtype=np.int8) + masked["Y"].to_numpy(dtype=np.int8),
        random_state=int(cfg["seed"]) + repetition * 1297,
    )
    train = masked.iloc[train_idx].reset_index(drop=True)
    test = masked.iloc[test_idx].reset_index(drop=True)
    initial_selected = train["S"].to_numpy(dtype=bool)
    y_train = train["Y"].to_numpy(dtype=np.int8)
    historical_p1_raw, _ = fit_probability_model(
        str(cfg["evaluation"]["nuisance_outcome_model"]),
        cfg,
        train.loc[initial_selected],
        y_train[initial_selected],
        train,
        int(cfg["seed"]) + repetition * 100003 + 7001,
    )
    historical_pi_raw, _ = fit_probability_model(
        str(cfg["evaluation"]["nuisance_propensity_model"]),
        cfg,
        train,
        initial_selected.astype(np.int8),
        train,
        int(cfg["seed"]) + repetition * 100003 + 7002,
    )
    p1_lo, p1_hi = map(float, cfg["evaluation"]["p1_clip"])
    pi_lo, pi_hi = map(float, cfg["evaluation"]["propensity_clip"])
    historical_p1 = np.clip(historical_p1_raw, p1_lo, p1_hi)
    historical_pi = np.clip(historical_pi_raw, pi_lo, pi_hi)
    rows: list[dict[str, Any]] = []
    for policy in cfg["part_b"]["policies"]:
        started = time.perf_counter()
        print(f"Mask {repetition + 1}: {policy} started", flush=True)
        rows.extend(
            _adaptive_policy(
                train,
                test,
                initial_selected,
                historical_p1,
                historical_pi,
                str(policy),
                cfg,
                repetition,
            )
        )
        print(f"Mask {repetition + 1}: {policy} complete in {time.perf_counter() - started:.1f} seconds", flush=True)
    return rows

