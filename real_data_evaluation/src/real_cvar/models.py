from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import feature_columns


def _preprocessor(cfg: dict[str, Any], scale_numeric: bool) -> ColumnTransformer:
    numeric, categorical = feature_columns(cfg)
    numeric_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=25)),
                    ]
                ),
                categorical,
            ),
        ],
        sparse_threshold=0.3,
    )


def build_classifier(kind: str, cfg: dict[str, Any], seed: int) -> BaseEstimator:
    if kind in {"logistic", "conservative_logistic"}:
        class_weight = None
        if kind == "conservative_logistic":
            class_weight = {0: 1.0, 1: float(cfg["evaluation"]["conservative_default_weight"])}
        estimator = LogisticRegression(
            C=1.0,
            class_weight=class_weight,
            max_iter=1000,
            solver="lbfgs",
            random_state=seed,
        )
        return Pipeline([("preprocess", _preprocessor(cfg, True)), ("model", estimator)])
    if kind == "gradient_boosting":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "gradient_boosting requires XGBoost; install real_data_evaluation/requirements.txt"
            ) from exc
        xcfg = cfg["xgboost"]
        estimator = XGBClassifier(
            n_estimators=int(xcfg["n_estimators"]),
            max_depth=int(xcfg["max_depth"]),
            learning_rate=float(xcfg["learning_rate"]),
            min_child_weight=float(xcfg["min_child_weight"]),
            subsample=float(xcfg["subsample"]),
            colsample_bytree=float(xcfg["colsample_bytree"]),
            reg_lambda=float(xcfg["reg_lambda"]),
            tree_method="hist",
            device=str(xcfg["device"]),
            n_jobs=int(xcfg["n_jobs"]),
            eval_metric="logloss",
            random_state=seed,
        )
        return Pipeline([("preprocess", _preprocessor(cfg, False)), ("model", estimator)])
    raise ValueError(f"Unknown model kind: {kind}")


def fit_probability_model(
    kind: str,
    cfg: dict[str, Any],
    train: pd.DataFrame,
    target: np.ndarray,
    predict: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, BaseEstimator]:
    target = np.asarray(target, dtype=np.int8)
    if len(np.unique(target)) < 2:
        prevalence = float(np.mean(target))
        return np.full(len(predict), prevalence, dtype=float), _ConstantProbability(prevalence)
    model = build_classifier(kind, cfg, seed)
    model.fit(train, target)
    probability = np.asarray(model.predict_proba(predict)[:, 1], dtype=float)
    return probability, model


class _ConstantProbability:
    def __init__(self, probability: float):
        self.probability = float(probability)

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        p = np.full(len(data), self.probability, dtype=float)
        return np.column_stack([1.0 - p, p])


def cross_fitted_predictions(data: pd.DataFrame, cfg: dict[str, Any], fold_ids: np.ndarray) -> pd.DataFrame:
    """Five-fold predictions for candidates, p1, and selection propensity."""
    n = len(data)
    models = list(cfg["evaluation"]["models"])
    result = pd.DataFrame(index=data.index)
    for name in models:
        result[f"pred_{name}"] = np.nan
    result["p1_raw"] = np.nan
    result["pi_raw"] = np.nan
    base_seed = int(cfg["seed"])
    n_folds = int(cfg["evaluation"]["n_folds"])
    for fold in range(n_folds):
        train_mask = fold_ids != fold
        test_mask = ~train_mask
        train = data.loc[train_mask]
        test = data.loc[test_mask]
        selected_train = train["S"].to_numpy(dtype=np.int8) == 1
        if selected_train.sum() == 0:
            raise ValueError(f"Fold {fold} has no selected training outcomes.")
        y_train = train.loc[selected_train, "Y"].to_numpy(dtype=np.int8)
        for offset, name in enumerate(models):
            probability, _ = fit_probability_model(
                name, cfg, train.loc[selected_train], y_train, test, base_seed + 101 * fold + offset
            )
            result.loc[test_mask, f"pred_{name}"] = probability
        p1, _ = fit_probability_model(
            str(cfg["evaluation"]["nuisance_outcome_model"]),
            cfg,
            train.loc[selected_train],
            y_train,
            test,
            base_seed + 1009 + fold,
        )
        pi, _ = fit_probability_model(
            str(cfg["evaluation"]["nuisance_propensity_model"]),
            cfg,
            train,
            train["S"].to_numpy(dtype=np.int8),
            test,
            base_seed + 2003 + fold,
        )
        result.loc[test_mask, "p1_raw"] = p1
        result.loc[test_mask, "pi_raw"] = pi
    if result.isna().any().any() or len(result) != n:
        raise RuntimeError("Cross-fitting failed to assign every held-out prediction.")
    p_lo, p_hi = map(float, cfg["evaluation"]["p1_clip"])
    pi_lo, pi_hi = map(float, cfg["evaluation"]["propensity_clip"])
    result["p1"] = result["p1_raw"].clip(p_lo, p_hi)
    result["pi"] = result["pi_raw"].clip(pi_lo, pi_hi)
    result["fold"] = fold_ids
    return result
