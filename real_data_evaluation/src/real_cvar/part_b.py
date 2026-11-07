from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.model_selection import StratifiedKFold, train_test_split

from .certificates import (
    brier_losses,
    empirical_cvar,
    estimate_certificate,
    sensitivity_bounds,
    tail_identification_value,
)
from .models import cross_fitted_predictions, fit_probability_model


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


def _folds(masked: pd.DataFrame, cfg: dict[str, Any], repetition: int) -> np.ndarray:
    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=int(cfg["seed"]) + 3571 * (repetition + 1),
    )
    fold_ids = np.empty(len(masked), dtype=np.int8)
    for fold, (_, test) in enumerate(splitter.split(np.zeros(len(masked)), masked["S"])):
        fold_ids[test] = fold
    return fold_ids


def _fixed_audit_curves(
    masked: pd.DataFrame,
    predictions: pd.DataFrame,
    cfg: dict[str, Any],
    repetition: int,
) -> list[dict[str, Any]]:
    model = str(cfg["part_b"]["adaptive_model"])
    prediction = predictions[f"pred_{model}"].to_numpy(dtype=float)
    p1 = predictions["p1"].to_numpy(dtype=float)
    pi = predictions["pi"].to_numpy(dtype=float)
    folds = predictions["fold"].to_numpy(dtype=np.int8)
    selected = masked["S"].to_numpy(dtype=bool)
    hidden = ~selected
    outcome = masked["Y"].to_numpy(dtype=float)
    alpha = max(map(float, cfg["evaluation"]["alphas"]))
    gamma = float(cfg["part_b"]["gamma0"])
    costs = _audit_cost(masked, cfg)
    initial = estimate_certificate(
        prediction, p1, pi, selected, outcome, alpha, gamma, folds=folds, orthogonal=True
    )
    l0, l1 = brier_losses(prediction)
    lo, hi = sensitivity_bounds(p1, gamma)
    rng = np.random.default_rng(int(cfg["seed"]) + repetition * 65537 + 19)
    scores = {
        "tiv_per_cost": tail_identification_value(
            prediction, p1, pi, initial.upper_eta, alpha, gamma, conditional_on_hidden=True
        ) / costs,
        "random": rng.random(len(masked)),
        "uncertainty": 1.0 - 2.0 * np.abs(prediction - 0.5),
        "low_propensity": -pi,
        "predicted_loss": (1.0 - p1) * l0 + p1 * l1,
        "sensitivity_width": hi - lo,
    }
    hidden_index = np.flatnonzero(hidden)
    fixed_local_width = tail_identification_value(
        prediction, p1, pi, initial.upper_eta, alpha, gamma, conditional_on_hidden=True
    )
    batch_size = max(
        1,
        int(np.ceil(float(cfg["part_b"]["batch_fraction_hidden"]) * len(hidden_index))),
    )
    true_cvar = empirical_cvar(np.square(prediction - outcome), alpha)
    rows: list[dict[str, Any]] = []
    for policy in cfg["part_b"]["policies"]:
        policy = str(policy)
        order = hidden_index[np.argsort(-scores[policy][hidden_index], kind="mergesort")]
        for round_id in range(int(cfg["part_b"]["adaptive_rounds"]) + 1):
            count = min(round_id * batch_size, len(order))
            audited = np.zeros(len(masked), dtype=bool)
            audited[order[:count]] = True
            cert = estimate_certificate(
                prediction,
                p1,
                pi,
                selected,
                outcome,
                alpha,
                gamma,
                folds=folds,
                audited=audited,
                orthogonal=True,
            )
            rows.append(
                {
                    "repetition": repetition,
                    "policy": policy,
                    "round": round_id,
                    "model": model,
                    "alpha": alpha,
                    "gamma": gamma,
                    "audited_labels": count,
                    "audited_fraction_hidden": float(count / len(hidden_index)),
                    "cumulative_cost": float(costs[audited].sum()),
                    "certificate_lower": cert.lower,
                    "certificate_upper": cert.upper,
                    "certificate_width": cert.width,
                    "fixed_threshold_width": float(np.mean((hidden & ~audited) * fixed_local_width)),
                    "true_cvar": true_cvar,
                    "fixed_eta": initial.upper_eta,
                }
            )
    return rows


def _static_repetition(
    masked: pd.DataFrame, cfg: dict[str, Any], repetition: int
) -> tuple[list[dict], list[dict], list[dict]]:
    fold_ids = _folds(masked, cfg, repetition)
    predictions = cross_fitted_predictions(masked, cfg, fold_ids)
    y = masked["Y"].to_numpy(dtype=float)
    s = masked["S"].to_numpy(dtype=bool)
    rows: list[dict] = []
    dominance_rows: list[dict] = []
    certificate_lookup: dict[tuple[float, float, str], Any] = {}
    true_lookup: dict[tuple[float, str], float] = {}
    for model in cfg["evaluation"]["models"]:
        prediction = predictions[f"pred_{model}"].to_numpy()
        loss = np.square(prediction - y)
        for alpha in map(float, cfg["evaluation"]["alphas"]):
            true_cvar = empirical_cvar(loss, alpha)
            true_lookup[(alpha, model)] = true_cvar
            selected_cvar = empirical_cvar(loss[s], alpha)
            for gamma in map(float, cfg["evaluation"]["gamma_grid"]):
                cert = estimate_certificate(
                    prediction,
                    predictions["p1"].to_numpy(),
                    predictions["pi"].to_numpy(),
                    s,
                    y,
                    alpha,
                    gamma,
                    folds=fold_ids,
                    confidence_level=float(cfg["evaluation"]["confidence_level"]),
                )
                certificate_lookup[(alpha, gamma, model)] = cert
                rows.append(
                    {
                        "repetition": repetition,
                        "model": model,
                        "alpha": alpha,
                        "gamma": gamma,
                        "true_cvar": true_cvar,
                        "selected_only_cvar": selected_cvar,
                        "lower": cert.lower,
                        "upper": cert.upper,
                        "width": cert.width,
                        "covered": cert.lower <= true_cvar <= cert.upper,
                        "enlarged_lower": cert.enlarged_lower,
                        "enlarged_upper": cert.enlarged_upper,
                        "enlarged_width": cert.enlarged_width,
                        "enlarged_covered": cert.enlarged_lower <= true_cvar <= cert.enlarged_upper,
                        "enlargement_method": "crossfit_fold_t",
                        "is_true_gamma": gamma == float(cfg["part_b"]["gamma0"]),
                    }
                )
    for alpha in map(float, cfg["evaluation"]["alphas"]):
        for gamma in map(float, cfg["evaluation"]["gamma_grid"]):
            for left, right in combinations(cfg["evaluation"]["models"], 2):
                left_cert = certificate_lookup[(alpha, gamma, left)]
                right_cert = certificate_lookup[(alpha, gamma, right)]
                if left_cert.upper < right_cert.lower:
                    claimed, comparator = left, right
                elif right_cert.upper < left_cert.lower:
                    claimed, comparator = right, left
                else:
                    claimed = comparator = None
                dominance_rows.append(
                    {
                        "repetition": repetition,
                        "alpha": alpha,
                        "gamma": gamma,
                        "model_a": left,
                        "model_b": right,
                        "certified": claimed is not None,
                        "certified_better": claimed,
                        "correct": (
                            np.nan
                            if claimed is None
                            else true_lookup[(alpha, claimed)] < true_lookup[(alpha, comparator)]
                        ),
                    }
                )
    fixed_audit_rows = _fixed_audit_curves(masked, predictions, cfg, repetition)
    return rows, dominance_rows, fixed_audit_rows


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
            orthogonal=True,
        )
        test_cvar = empirical_cvar(np.square(prediction_test - y_test), alpha)
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
                "certificate_lower": cert.lower,
                "certificate_upper": cert.upper,
                "certificate_width": cert.width,
                "true_test_cvar": test_cvar,
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
    return rows


def run_part_b(data: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    output = Path(cfg["output"]["directory"])
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    pseudo = data.loc[data["S"].eq(1) & data["Y"].notna()].reset_index(drop=True)
    static_rows: list[dict] = []
    dominance_rows: list[dict] = []
    fixed_audit_rows: list[dict] = []
    adaptive_rows: list[dict] = []
    mask_metadata: list[dict] = []
    for repetition in range(int(cfg["part_b"]["repetitions"])):
        masked, metadata = impose_mask(pseudo, cfg, repetition)
        metadata["repetition"] = repetition
        mask_metadata.append(metadata)
        current_static, current_dominance, current_fixed_audits = _static_repetition(masked, cfg, repetition)
        static_rows.extend(current_static)
        dominance_rows.extend(current_dominance)
        fixed_audit_rows.extend(current_fixed_audits)
        adaptive_rows.extend(_adaptive_repetition(masked, cfg, repetition))
    static = pd.DataFrame(static_rows)
    dominance = pd.DataFrame(dominance_rows)
    fixed_audits = pd.DataFrame(fixed_audit_rows)
    adaptive = pd.DataFrame(adaptive_rows)
    cost_rows: list[dict[str, Any]] = []
    target = float(cfg["part_b"]["target_width"])
    for experiment, frame in (("fixed", fixed_audits), ("adaptive", adaptive)):
        for (repetition, policy), cur in frame.groupby(["repetition", "policy"]):
            reached = cur[cur["certificate_width"] <= target].sort_values("round")
            first = None if reached.empty else reached.iloc[0]
            cost_rows.append(
                {
                    "experiment": experiment,
                    "repetition": repetition,
                    "policy": policy,
                    "target_width": target,
                    "reached": first is not None,
                    "round_to_target": np.nan if first is None else int(first["round"]),
                    "labels_to_target": np.nan if first is None else int(first["audited_labels"]),
                    "cost_to_target": np.nan if first is None else float(first["cumulative_cost"]),
                }
            )
    costs = pd.DataFrame(cost_rows)
    static.to_csv(tables / "part_b_masking_certificates.csv", index=False)
    dominance.to_csv(tables / "part_b_dominance.csv", index=False)
    fixed_audits.to_csv(tables / "part_b_fixed_audits.csv", index=False)
    adaptive.to_csv(tables / "part_b_adaptive_audits.csv", index=False)
    costs.to_csv(tables / "part_b_cost_to_width.csv", index=False)
    with (output / "part_b_mask_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(mask_metadata, handle, indent=2)
    return {
        "static": static,
        "dominance": dominance,
        "fixed_audits": fixed_audits,
        "adaptive": adaptive,
        "costs": costs,
    }
