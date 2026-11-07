"""Descriptive summaries of the repaired sequential trajectories.

Differences are TIV minus random, paired by the same controlled mask and split.
The across-mask t interval summarizes a mean difference under this simulation
design; it is not a simultaneous certificate or a population coverage claim.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

METRICS = ("estimated_retained_width", "mean_test_brier", "true_test_cvar")
REFERENCE = {"estimated_retained_width": 0.0, "mean_test_brier": 0.25, "true_test_cvar": 0.25}


def summarize(frame: pd.DataFrame, terminal_round: int, expected_repetitions: int) -> dict[str, pd.DataFrame]:
    """Require complete terminal pairing, then summarize observed rounds."""
    required = {"repetition", "policy", "round", "alpha", "gamma", "audited_labels",
                "audited_fraction_hidden", "cumulative_cost", *METRICS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing trajectory columns: {sorted(missing)}")
    if frame.duplicated(["repetition", "policy", "round"]).any():
        raise ValueError("Duplicate repetition/policy/round rows.")
    if not np.isfinite(frame[list(METRICS)].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite sequential metrics.")
    terminal = frame.loc[frame["round"].eq(terminal_round)]
    tiv = terminal.loc[terminal.policy.eq("tiv_per_cost")].set_index("repetition").sort_index()
    random = terminal.loc[terminal.policy.eq("random")].set_index("repetition").sort_index()
    expected = list(range(expected_repetitions))
    if tiv.index.tolist() != expected or random.index.tolist() != expected:
        raise ValueError("Terminal TIV/random observations must include every identical repetition.")
    for name in ["alpha", "gamma", "audited_labels", "audited_fraction_hidden"]:
        if not np.allclose(tiv[name], random[name], rtol=0.0, atol=1e-12):
            raise ValueError(f"Paired terminal observations differ in {name}.")

    trajectory_rows = []
    for (policy, round_id), current in frame.groupby(["policy", "round"], sort=True):
        for metric in METRICS:
            values = current[metric]
            trajectory_rows.append({
                "policy": policy, "round": round_id, "metric": metric, "n": len(current),
                "audited_fraction_hidden_median": float(current.audited_fraction_hidden.median()),
                "mean": float(values.mean()), "median": float(values.median()),
                "q25": float(values.quantile(0.25)), "q75": float(values.quantile(0.75)),
                "constant_reference": REFERENCE[metric],
            })
    pair_rows, comparison_rows = [], []
    for metric in METRICS:
        difference = tiv[metric] - random[metric]
        n = len(difference)
        mean = float(difference.mean())
        halfwidth = (float(student_t.ppf(0.975, n - 1) * difference.std(ddof=1) / np.sqrt(n))
                     if n > 1 else float("nan"))
        for repetition in expected:
            pair_rows.append({
                "repetition": repetition, "round": terminal_round, "metric": metric,
                "tiv_value": float(tiv.loc[repetition, metric]),
                "random_value": float(random.loc[repetition, metric]),
                "tiv_minus_random": float(difference.loc[repetition]),
                "audited_labels": int(tiv.loc[repetition, "audited_labels"]),
                "audited_fraction_hidden": float(tiv.loc[repetition, "audited_fraction_hidden"]),
                "tiv_cumulative_cost": float(tiv.loc[repetition, "cumulative_cost"]),
                "random_cumulative_cost": float(random.loc[repetition, "cumulative_cost"]),
            })
        comparison_rows.append({
            "metric": metric, "round": terminal_round, "n_pairs": n,
            "tiv_median": float(tiv[metric].median()), "random_median": float(random[metric].median()),
            "tiv_mean": float(tiv[metric].mean()), "random_mean": float(random[metric].mean()),
            "paired_difference_mean": mean, "paired_difference_median": float(difference.median()),
            "paired_difference_q25": float(difference.quantile(0.25)),
            "paired_difference_q75": float(difference.quantile(0.75)),
            "paired_mean_descriptive_t95_lower": mean - halfwidth,
            "paired_mean_descriptive_t95_upper": mean + halfwidth,
            "tiv_lower_count": int((difference < -1e-12).sum()),
            "tie_count": int((difference.abs() <= 1e-12).sum()),
            "tiv_higher_count": int((difference > 1e-12).sum()),
            "constant_reference": REFERENCE[metric],
        })
    return {
        "trajectory_summary": pd.DataFrame(trajectory_rows),
        "terminal_pairs": pd.DataFrame(pair_rows),
        "terminal_comparison": pd.DataFrame(comparison_rows),
        "constant_reference": pd.DataFrame([{
            "predictor": "constant_half", "alpha": alpha, "gamma": gamma,
            "mean_test_brier": 0.25, "true_test_cvar": 0.25,
            "estimated_retained_lower": 0.25, "estimated_retained_upper": 0.25,
            "estimated_retained_width": 0.0, "tiv": 0.0,
        } for alpha, gamma in frame[["alpha", "gamma"]].drop_duplicates().itertuples(index=False, name=None)]),
    }
