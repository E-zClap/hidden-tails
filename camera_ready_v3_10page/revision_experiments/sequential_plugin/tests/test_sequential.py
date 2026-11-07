"""Behavioral regression tests; set SEQUENTIAL_TEST_ORIGINAL=1 for the RED check."""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if os.environ.get("SEQUENTIAL_TEST_ORIGINAL") == "1":
    sys.path.insert(0, str(ROOT.parents[2] / "real_data_evaluation" / "src"))
    PACKAGE = "real_cvar"
else:
    PACKAGE = "real_cvar_plugin"
part_b = importlib.import_module(f"{PACKAGE}.part_b")
certificates = importlib.import_module(f"{PACKAGE}.certificates")
models = importlib.import_module(f"{PACKAGE}.models")


def fixture():
    # The complete-label learner has nonconstant predictions. Historical p1 is
    # deliberately imperfect, so the stale residual cannot vanish accidentally.
    x = np.linspace(-2.0, 2.0, 80)
    y = ((np.arange(80) % 5 == 0) | (x > 0.8)).astype(float)
    train = pd.DataFrame({"x": x, "Y": y})
    test = train.iloc[::4].copy()
    selected = np.arange(80) % 3 == 0
    cfg = {
        "seed": 19,
        "data": {"features": {"numeric": ["x"], "categorical": []}},
        "evaluation": {
            "alphas": [0.70], "propensity_clip": [0.02, 0.98],
            "nuisance_propensity_model": "logistic", "nuisance_outcome_model": "logistic",
        },
        "part_b": {
            "gamma0": 3.0, "batch_fraction_hidden": 1.0,
            "adaptive_rounds": 1, "adaptive_model": "logistic",
            "policies": ["tiv_per_cost", "random"], "audit_cost_amount_weight": 0.0,
        },
    }
    return train, test, selected, np.full(80, 0.15), np.full(80, 0.4), cfg


class SequentialRegressionTests(unittest.TestCase):
    def test_six_policies_avoid_fits_for_unused_baseline_nuisances(self):
        train, test, selected, p1, pi, cfg = fixture()
        cfg["part_b"]["policies"] = ["tiv_per_cost", "random", "uncertainty", "low_propensity", "predicted_loss", "sensitivity_width"]
        for policy in cfg["part_b"]["policies"]:
            with self.subTest(policy=policy), patch.object(part_b, "fit_probability_model", wraps=models.fit_probability_model) as fitted:
                part_b._adaptive_policy(train, test, selected, p1, pi, policy, cfg, 0)
                per_round = 2 if policy in {"low_propensity", "predicted_loss"} else 1
                self.assertEqual(fitted.call_count, 2 * per_round)

    @unittest.skipIf(PACKAGE == "real_cvar", "Equivalence applies to the repaired implementation")
    def test_six_optimized_trajectories_equal_unoptimized_real_model_outputs(self):
        from real_cvar_plugin import part_b_unoptimized
        train, test, selected, p1, pi, cfg = fixture()
        cfg["part_b"]["policies"] = ["tiv_per_cost", "random", "uncertainty", "low_propensity", "predicted_loss", "sensitivity_width"]
        # Partial revelation makes every policy select a nontrivial subset.
        cfg["part_b"]["batch_fraction_hidden"] = 0.2
        cfg["part_b"]["adaptive_rounds"] = 2
        for policy in cfg["part_b"]["policies"]:
            with self.subTest(policy=policy):
                before = part_b_unoptimized._adaptive_policy(train, test, selected, p1, pi, policy, cfg, 0)
                after = part_b._adaptive_policy(train, test, selected, p1, pi, policy, cfg, 0)
                self.assertEqual(before, after)

    def test_initial_objective_uses_historical_bounds_and_actual_revealed_losses(self):
        train, test, selected, p1, pi, cfg = fixture()
        rows = part_b._adaptive_policy(train, test, selected, p1, pi, "random", cfg, 0)
        prediction, _ = models.fit_probability_model(
            "logistic", cfg, train.loc[selected], train.Y.to_numpy()[selected].astype(np.int8), train, 19,
        )
        l0, l1 = prediction ** 2, (1 - prediction) ** 2
        y = train.Y.to_numpy()
        candidates = np.unique(np.r_[0.0, l0, l1, 1.0])
        # For historical p1=.15 and Gamma=3, exact bounds are 1/18 and 9/26.
        for upper, key in [(False, "estimated_retained_lower"), (True, "estimated_retained_upper")]:
            q = np.where(l1 >= l0, 9 / 26 if upper else 1 / 18, 1 / 18 if upper else 9 / 26)
            objectives = []
            for eta in candidates:
                known = np.maximum((prediction - y) ** 2 - eta, 0)
                hidden = (1 - q) * np.maximum(l0 - eta, 0) + q * np.maximum(l1 - eta, 0)
                objectives.append(eta + np.mean(np.where(selected, known, hidden)) / 0.3)
            if PACKAGE == "real_cvar":
                key = key.replace("estimated_retained_", "certificate_")
            self.assertAlmostEqual(rows[0][key], min(objectives), places=11)

    def test_each_round_records_held_out_mean_brier_and_cvar(self):
        train, test, selected, p1, pi, cfg = fixture()
        rows = part_b._adaptive_policy(train, test, selected, p1, pi, "random", cfg, 0)
        for round_id, row in enumerate(rows):
            known = selected if round_id == 0 else np.ones(len(train), dtype=bool)
            _, fitted = models.fit_probability_model(
                "logistic", cfg, train.loc[known], train.Y.to_numpy()[known].astype(np.int8),
                train, 19 + 101 * round_id,
            )
            losses = np.sort(np.square(fitted.predict_proba(test)[:, 1] - test.Y.to_numpy()))
            self.assertIn("mean_test_brier", row)
            self.assertAlmostEqual(row["mean_test_brier"], float(np.mean(losses)), places=12)
            # alpha=.70 and n_test=20: mean of the largest six losses.
            self.assertAlmostEqual(row["true_test_cvar"], float(np.mean(losses[-6:])), places=12)

    def test_full_revelation_trajectory_equals_full_data_empirical_cvar(self):
        train, test, selected, p1, pi, cfg = fixture()
        rows = part_b._adaptive_policy(train, test, selected, p1, pi, "tiv_per_cost", cfg, 0)
        predictions, _ = models.fit_probability_model(
            "logistic", cfg, train, train.Y.to_numpy().astype(np.int8), train, 120,
        )
        loss = np.sort(np.square(predictions - train.Y.to_numpy()))
        # alpha=.70 and n=80: exact empirical CVaR is the largest 24 losses' mean.
        truth = float(np.mean(loss[-24:]))
        terminal = rows[-1]
        self.assertEqual(terminal["audited_labels"], int((~selected).sum()))
        lower_key = "certificate_lower" if PACKAGE == "real_cvar" else "estimated_retained_lower"
        upper_key = "certificate_upper" if PACKAGE == "real_cvar" else "estimated_retained_upper"
        self.assertAlmostEqual(terminal[lower_key], truth, places=11)
        self.assertAlmostEqual(terminal[upper_key], truth, places=11)

    def test_plugin_endpoints_fully_revealed_with_historical_selection(self):
        prediction = np.array([0.1, 0.2, 0.7, 0.9, 0.4, 0.6])
        outcome = np.array([0, 0, 1, 1, 1, 0], dtype=float)
        selected = np.array([True, False, True, False, False, True])
        cert = certificates.estimate_certificate(
            prediction, np.full(6, 0.2), np.full(6, 0.4), selected, outcome,
            alpha=0.5, gamma=12.0, audited=~selected, orthogonal=False,
        )
        # Largest three losses are .36, .36, .09.
        self.assertAlmostEqual(cert.lower, 0.27, places=12)
        self.assertAlmostEqual(cert.upper, 0.27, places=12)

    def test_constant_half_forecast_has_quarter_risk_zero_width_and_tiv(self):
        prediction = np.full(6, 0.5)
        outcome = np.array([0, 1, 0, 1, 0, 1], dtype=float)
        selected = np.array([True, False, True, False, False, True])
        p1 = np.linspace(0.1, 0.9, 6)
        pi = np.linspace(0.05, 0.95, 6)
        self.assertAlmostEqual(float(np.mean(np.square(prediction - outcome))), 0.25)
        for gamma in [1.0, 3.0, 12.0]:
            for alpha in [0.5, 0.90, 0.95]:
                for audited in [np.zeros(6, dtype=bool), ~selected]:
                    cert = certificates.estimate_certificate(
                        prediction, p1, pi, selected, outcome, alpha, gamma,
                        audited=audited, orthogonal=False,
                    )
                    self.assertAlmostEqual(cert.lower, 0.25, places=12)
                    self.assertAlmostEqual(cert.upper, 0.25, places=12)
                    self.assertAlmostEqual(cert.width, 0.0, places=12)
                    self.assertAlmostEqual(certificates.empirical_cvar(np.full(6, 0.25), alpha), 0.25)
                    for eta in [0.0, 0.10, 0.25, 0.8]:
                        np.testing.assert_array_equal(
                            certificates.tail_identification_value(prediction, p1, pi, eta, alpha, gamma),
                            np.zeros(6),
                        )


if __name__ == "__main__":
    unittest.main()
