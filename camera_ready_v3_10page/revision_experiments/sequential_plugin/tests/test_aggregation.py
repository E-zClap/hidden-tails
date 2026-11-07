from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class AggregationTests(unittest.TestCase):
    def load_aggregation(self):
        self.assertIsNotNone(importlib.util.find_spec("aggregation"), "paired aggregation is not implemented")
        import aggregation
        return aggregation

    def fixture(self):
        rows = []
        for repetition, (tiv, random) in enumerate(zip([0.1, 0.1, 0.8], [0.05, 0.2, 0.3])):
            for policy, value in [("tiv_per_cost", tiv), ("random", random)]:
                rows.append({
                    "repetition": repetition, "policy": policy, "round": 8,
                    "alpha": 0.95, "gamma": 3.0, "audited_labels": 16,
                    "audited_fraction_hidden": 0.16, "cumulative_cost": 16.0,
                    "estimated_retained_width": value, "true_test_cvar": value,
                    "mean_test_brier": value,
                })
        return pd.DataFrame(rows)

    def test_terminal_comparison_pairs_same_mask_before_aggregation(self):
        aggregation = self.load_aggregation()
        outputs = aggregation.summarize(self.fixture(), terminal_round=8, expected_repetitions=3)
        result = outputs["terminal_comparison"].set_index("metric").loc["true_test_cvar"]
        self.assertEqual(result["n_pairs"], 3)
        self.assertAlmostEqual(result["paired_difference_median"], 0.05)
        self.assertAlmostEqual(result["paired_difference_mean"], 0.15)
        self.assertEqual(result["tiv_lower_count"], 1)
        self.assertEqual(result["tiv_higher_count"], 2)
        self.assertEqual(result["constant_reference"], 0.25)
        self.assertEqual(len(outputs["terminal_pairs"]), 9)

    def test_missing_terminal_pair_is_rejected(self):
        aggregation = self.load_aggregation()
        with self.assertRaises(ValueError):
            aggregation.summarize(self.fixture().iloc[:-1], terminal_round=8, expected_repetitions=3)

    def test_mismatched_label_budget_is_rejected(self):
        aggregation = self.load_aggregation()
        frame = self.fixture()
        frame.loc[0, "audited_labels"] = 17
        with self.assertRaises(ValueError):
            aggregation.summarize(frame, terminal_round=8, expected_repetitions=3)


if __name__ == "__main__":
    unittest.main()
