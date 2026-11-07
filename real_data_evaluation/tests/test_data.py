from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from real_cvar.data import _employment_years, _standardize_chunk


class DataPreparationTests(unittest.TestCase):
    def test_employment_years_accepts_fractional_less_than_one(self) -> None:
        source = pd.Series(["< 1 year", "10+ years", "3 years", "n/a", None])
        actual = _employment_years(source).to_numpy(dtype=float, na_value=np.nan)
        np.testing.assert_allclose(actual[:3], np.array([0.5, 10.0, 3.0]))
        self.assertTrue(np.isnan(actual[3:]).all())

    def test_minimal_accepted_chunk_standardizes_without_extension_dtype_error(self) -> None:
        chunk = pd.DataFrame(
            {
                "issue_d": ["Jan-2012", "Feb-2012"],
                "loan_amnt": [5000, 8000],
                "dti": [12.5, 8.0],
                "fico_range_low": [660, 700],
                "fico_range_high": [664, 704],
                "emp_length": ["< 1 year", "10+ years"],
                "addr_state": ["CA", "NY"],
                "zip_code": ["941xx", "100xx"],
                "purpose": ["debt_consolidation", "credit_card"],
                "policy_code": [1, 1],
                "loan_status": ["Fully Paid", "Charged Off"],
            }
        )
        mapping = {
            "date": "issue_d",
            "requested_amount": "loan_amnt",
            "dti": "dti",
            "risk_score_low": "fico_range_low",
            "risk_score_high": "fico_range_high",
            "employment_length": "emp_length",
            "state": "addr_state",
            "zip3": "zip_code",
            "purpose": "purpose",
            "policy_code": "policy_code",
            "loan_status": "loan_status",
        }
        result = _standardize_chunk(chunk, mapping, accepted=True, year=2012)
        np.testing.assert_allclose(result["employment_years"], np.array([0.5, 10.0]))
        np.testing.assert_array_equal(result["Y"], np.array([0.0, 1.0]))


if __name__ == "__main__":
    unittest.main()

