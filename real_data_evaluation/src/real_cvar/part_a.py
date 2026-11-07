from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .certificates import empirical_cvar, estimate_certificate
from .models import cross_fitted_predictions


def _make_folds(data: pd.DataFrame, cfg: dict[str, Any], seed_offset: int = 0) -> np.ndarray:
    splitter = StratifiedKFold(
        n_splits=int(cfg["evaluation"]["n_folds"]),
        shuffle=True,
        random_state=int(cfg["seed"]) + seed_offset,
    )
    fold_ids = np.empty(len(data), dtype=np.int8)
    stratify = data["S"].to_numpy(dtype=np.int8)
    for fold, (_, test) in enumerate(splitter.split(np.zeros(len(data)), stratify)):
        fold_ids[test] = fold
    return fold_ids


def _certificate_rows(
    data: pd.DataFrame,
    predictions: pd.DataFrame,
    cfg: dict[str, Any],
    subset_name: str = "all",
    subset_value: str = "all",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected = data["S"].to_numpy(dtype=bool)
    outcome = data["Y"].to_numpy(dtype=float)
    for model in cfg["evaluation"]["models"]:
        prediction = predictions[f"pred_{model}"].to_numpy(dtype=float)
        for alpha in map(float, cfg["evaluation"]["alphas"]):
            selected_cvar = empirical_cvar(np.square(prediction[selected] - outcome[selected]), alpha)
            for gamma in map(float, cfg["evaluation"]["gamma_grid"]):
                cert = estimate_certificate(
                    prediction=prediction,
                    p1=predictions["p1"].to_numpy(),
                    propensity=predictions["pi"].to_numpy(),
                    selected=selected,
                    outcome=outcome,
                    alpha=alpha,
                    gamma=gamma,
                    folds=predictions["fold"].to_numpy(),
                    confidence_level=float(cfg["evaluation"]["confidence_level"]),
                )
                rows.append(
                    {
                        "subset_type": subset_name,
                        "subset": subset_value,
                        "n": len(data),
                        "n_selected": int(selected.sum()),
                        "model": model,
                        "alpha": alpha,
                        "gamma": gamma,
                        "accepted_only_cvar": selected_cvar,
                        "lower": cert.lower,
                        "upper": cert.upper,
                        "width": cert.width,
                        "lower_eta": cert.lower_eta,
                        "upper_eta": cert.upper_eta,
                        "enlarged_lower": cert.enlarged_lower,
                        "enlarged_upper": cert.enlarged_upper,
                        "enlarged_width": cert.enlarged_width,
                    }
                )
    return rows


def _ranking_diagnostics(certificates: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows = certificates.query("subset_type == 'all'").copy()
    rank_rows: list[dict[str, Any]] = []
    break_rows: list[dict[str, Any]] = []
    models = list(cfg["evaluation"]["models"])
    gammas = list(map(float, cfg["evaluation"]["gamma_grid"]))
    for alpha in map(float, cfg["evaluation"]["alphas"]):
        cur_alpha = all_rows[all_rows["alpha"] == alpha]
        accepted_scores = cur_alpha.groupby("model")["accepted_only_cvar"].first()
        accepted_rank = accepted_scores.rank(method="min").astype(int)
        for gamma in gammas:
            cur = cur_alpha[cur_alpha["gamma"] == gamma].set_index("model")
            cert_rank = cur["upper"].rank(method="min").astype(int)
            tau = kendalltau(
                [accepted_rank[m] for m in models], [cert_rank[m] for m in models]
            ).statistic
            reversals = 0
            for a, b in combinations(models, 2):
                if np.sign(accepted_scores[a] - accepted_scores[b]) != np.sign(cur.loc[a, "upper"] - cur.loc[b, "upper"]):
                    reversals += 1
            for model in models:
                rank_rows.append(
                    {
                        "alpha": alpha,
                        "gamma": gamma,
                        "model": model,
                        "accepted_only_rank": int(accepted_rank[model]),
                        "upper_certificate_rank": int(cert_rank[model]),
                        "rank_change": int(cert_rank[model] - accepted_rank[model]),
                        "kendall_tau": float(tau) if np.isfinite(tau) else np.nan,
                        "pairwise_rank_reversals": reversals,
                    }
                )
        for first, second in combinations(models, 2):
            if accepted_scores[first] <= accepted_scores[second]:
                preferred, comparator = first, second
            else:
                preferred, comparator = second, first
            break_gamma = np.nan
            break_gamma_grid = np.nan
            right_censored = True
            holds_at_one = False
            previous_gamma = gammas[0]
            previous_gap = np.nan
            for gamma in gammas:
                cur = cur_alpha[cur_alpha["gamma"] == gamma].set_index("model")
                gap = float(cur.loc[comparator, "lower"] - cur.loc[preferred, "upper"])
                dominance = gap > 0.0
                if gamma == gammas[0]:
                    holds_at_one = dominance
                if not dominance:
                    break_gamma_grid = gamma
                    right_censored = False
                    if gamma == gammas[0] or not np.isfinite(previous_gap) or previous_gap <= 0.0:
                        break_gamma = gamma
                    else:
                        fraction = previous_gap / (previous_gap - gap)
                        break_gamma = float(np.exp(np.log(previous_gamma) + fraction * (np.log(gamma) - np.log(previous_gamma))))
                    break
                previous_gamma = gamma
                previous_gap = gap
            break_rows.append(
                {
                    "alpha": alpha,
                    "accepted_preferred": preferred,
                    "comparator": comparator,
                    "accepted_preferred_cvar": accepted_scores[preferred],
                    "comparator_cvar": accepted_scores[comparator],
                    "dominance_holds_at_gamma_1": holds_at_one,
                    "gamma_break": break_gamma,
                    "gamma_break_first_failed_grid": break_gamma_grid,
                    "right_censored_above_grid": right_censored,
                }
            )
    return pd.DataFrame(rank_rows), pd.DataFrame(break_rows)


def run_part_a(data: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    output = Path(cfg["output"]["directory"])
    tables = output / "tables"
    cache = output / "cache"
    tables.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    data = data.reset_index(drop=True)
    fold_ids = _make_folds(data, cfg)
    predictions = cross_fitted_predictions(data, cfg, fold_ids)
    prediction_cache = pd.concat(
        [data[["source_row_id", "S", "Y"]], predictions], axis=1
    )
    prediction_cache.to_parquet(cache / "part_a_cross_fitted_predictions.parquet", index=False)

    cert_rows = _certificate_rows(data, predictions, cfg)
    subgroup_columns = ["subgroup_fico", "subgroup_dti", "subgroup_amount"]
    minimum_subgroup = max(250, int(0.002 * len(data)))
    for column in subgroup_columns:
        for value, indices in data.groupby(column, observed=True).groups.items():
            index = np.asarray(list(indices), dtype=int)
            if len(index) < minimum_subgroup or int(data.loc[index, "S"].sum()) < 50:
                continue
            cert_rows.extend(
                _certificate_rows(
                    data.loc[index].reset_index(drop=True),
                    predictions.loc[index].reset_index(drop=True),
                    cfg,
                    subset_name=column,
                    subset_value=str(value),
                )
            )
    certificates = pd.DataFrame(cert_rows)
    rankings, breakdown = _ranking_diagnostics(certificates, cfg)
    pi_raw = predictions["pi_raw"].to_numpy()
    pi = predictions["pi"].to_numpy()
    s = data["S"].to_numpy(dtype=np.int8)
    pi_lo, pi_hi = map(float, cfg["evaluation"]["propensity_clip"])
    diagnostics = pd.DataFrame(
        [
            {
                "n": len(data),
                "n_selected": int(s.sum()),
                "n_unselected": int((1 - s).sum()),
                "selected_fraction": float(s.mean()),
                "propensity_auc": roc_auc_score(s, pi_raw),
                "propensity_brier": brier_score_loss(s, pi_raw),
                "pi_raw_min": float(np.min(pi_raw)),
                "pi_raw_p01": float(np.quantile(pi_raw, 0.01)),
                "pi_raw_p05": float(np.quantile(pi_raw, 0.05)),
                "pi_raw_median": float(np.quantile(pi_raw, 0.50)),
                "pi_raw_p95": float(np.quantile(pi_raw, 0.95)),
                "pi_raw_p99": float(np.quantile(pi_raw, 0.99)),
                "pi_raw_max": float(np.max(pi_raw)),
                "fraction_clipped_low": float(np.mean(pi_raw < pi_lo)),
                "fraction_clipped_high": float(np.mean(pi_raw > pi_hi)),
                "effective_ipw_max": float(np.max(1.0 / pi)),
            }
        ]
    )
    certificates.to_csv(tables / "part_a_certificates.csv", index=False)
    certificates.query("subset_type != 'all'").to_csv(tables / "part_a_subgroups.csv", index=False)
    rankings.to_csv(tables / "part_a_rankings.csv", index=False)
    breakdown.to_csv(tables / "part_a_breakdown.csv", index=False)
    diagnostics.to_csv(tables / "part_a_overlap_diagnostics.csv", index=False)
    metadata = {
        "cohort_year": int(cfg["cohort_year"]),
        "n": len(data),
        "accepted": int(data["S"].sum()),
        "rejected": int((1 - data["S"]).sum()),
        "outcome_definition": "Y=1 for Charged Off/Default; Y=0 for Fully Paid; non-mature statuses excluded",
        "loss": "Brier loss (f(X)-Y)^2",
        "cross_fitting_folds": int(cfg["evaluation"]["n_folds"]),
    }
    with (output / "part_a_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return {
        "certificates": certificates,
        "rankings": rankings,
        "breakdown": breakdown,
        "diagnostics": diagnostics,
    }
