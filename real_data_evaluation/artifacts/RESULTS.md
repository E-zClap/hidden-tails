# Real-data evaluation results

This file is generated from the run artifacts; it contains no hand-entered numerical claims.

## Part A: actual selective labels

The prepared cohort contains 390,644 applications: 53,367 accepted mature loans and 337,277 rejected applications.
The cross-fitted propensity AUC is 0.994. Fractions clipped below/above the predeclared overlap limits are 75.258%/1.803%.

Pairwise breakdown values are in `tables/part_a_breakdown.csv`; accepted-only versus certificate rank changes are in `tables/part_a_rankings.csv`.

## Part B: controlled masking

At the known masking sensitivity Gamma0=3, the empirical correctness rate among non-abstaining certified dominance claims is 100.000%.
Coverage and sharpness summaries are in `tables/part_b_coverage_summary.csv`.
Fixed-model TIV/cost trajectories are in `tables/part_b_fixed_audit_summary.csv`. Adaptive retraining trajectories are in `tables/part_b_adaptive_summary.csv`, with both cost-to-width comparisons in `tables/part_b_cost_to_width.csv`.

## Interpretation guardrail

The cross-fold t enlargement is evaluated empirically; it is not presented as an exact simultaneous confidence theorem. The iterative retraining study is an adaptive empirical extension, not a claim of globally optimal active learning.
