from __future__ import annotations

import unittest

import numpy as np

from real_cvar.certificates import (
    _minimize_weighted_hinges,
    empirical_cvar,
    estimate_certificate,
    sensitivity_bounds,
    tail_identification_value,
)


class CertificateTests(unittest.TestCase):
    def test_sensitivity_endpoints(self) -> None:
        p = np.array([0.1, 0.5, 0.9])
        lo, hi = sensitivity_bounds(p, 3.0)
        self.assertTrue(np.all(lo <= p))
        self.assertTrue(np.all(p <= hi))
        same_lo, same_hi = sensitivity_bounds(p, 1.0)
        np.testing.assert_allclose(same_lo, p)
        np.testing.assert_allclose(same_hi, p)

    def test_fast_hinge_optimizer_matches_brute_force(self) -> None:
        rng = np.random.default_rng(7)
        loss0 = rng.uniform(size=40)
        loss1 = rng.uniform(size=40)
        weight0 = rng.normal(0.5, 0.2, size=40)
        weight1 = 1.0 - weight0
        alpha = 0.9
        value, eta = _minimize_weighted_hinges(loss0, loss1, weight0, weight1, alpha)
        candidates = np.unique(np.concatenate(([0.0, 1.0], loss0, loss1)))
        brute = np.array(
            [
                x
                + np.mean(weight0 * np.maximum(loss0 - x, 0.0) + weight1 * np.maximum(loss1 - x, 0.0))
                / (1.0 - alpha)
                for x in candidates
            ]
        )
        self.assertAlmostEqual(value, float(brute.min()), places=12)
        self.assertAlmostEqual(value, float(brute[np.where(candidates == eta)[0][0]]), places=12)

    def test_fully_observed_certificate_collapses(self) -> None:
        prediction = np.array([0.1, 0.2, 0.7, 0.9, 0.4, 0.6])
        outcome = np.array([0, 0, 1, 1, 1, 0], dtype=float)
        selected = np.ones(len(outcome), dtype=bool)
        p1 = np.full(len(outcome), 0.5)
        propensity = np.ones(len(outcome))
        alpha = 0.8
        cert = estimate_certificate(
            prediction, p1, propensity, selected, outcome, alpha, gamma=12.0, orthogonal=True
        )
        truth = empirical_cvar(np.square(prediction - outcome), alpha)
        self.assertAlmostEqual(cert.lower, truth, places=12)
        self.assertAlmostEqual(cert.upper, truth, places=12)

    def test_tiv_hidden_pool_version_removes_hidden_mass_factor(self) -> None:
        prediction = np.array([0.2, 0.8])
        p1 = np.array([0.3, 0.3])
        pi = np.array([0.1, 0.9])
        population = tail_identification_value(prediction, p1, pi, 0.2, 0.9, 3.0)
        hidden_pool = tail_identification_value(
            prediction, p1, pi, 0.2, 0.9, 3.0, conditional_on_hidden=True
        )
        np.testing.assert_allclose(population, hidden_pool * (1.0 - pi))


if __name__ == "__main__":
    unittest.main()

