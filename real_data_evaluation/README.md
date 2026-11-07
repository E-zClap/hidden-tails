# Historical two-part real-data evaluation

This folder preserves the July 2026 selective-label CVaR evaluation and its recorded outputs. The current sequential experiment is in [`../camera_ready_v3_10page/revision_experiments/sequential_plugin/`](../camera_ready_v3_10page/revision_experiments/sequential_plugin/). The adaptive tables here belong to the earlier procedure and are retained for provenance and regression comparisons. See [the experiment guide](../docs/experiments.md) for separate reproduction commands.

The protocol has two deliberately different jobs:

1. **Part A — actual selective labels.** A 2012 LendingClub cohort combines accepted mature loans and rejected applications. Rejected outcomes remain genuinely unobserved, so this part reports certificates, ranking breakdowns, overlap, clipping, and subgroup width—but never ground-truth coverage.
2. **Part B — controlled masking.** The fully observed mature accepted-loan cohort becomes a pseudo-population. A known outcome-dependent selection law masks labels, making ground-truth coverage, sharpness, dominance correctness, audit cost, and adaptive retraining measurable.

The loss is the bounded Brier loss `(f(X)-Y)^2`. All predictors, the selected-outcome nuisance, and the propensity nuisance are evaluated by five-fold cross-fitting in Parts A and B's static masking study. The candidate set is intentionally small: logistic regression, GPU gradient boosting, and a predeclared default-upweighted conservative logistic model.

## Data contract

Obtain the LendingClub accepted and rejected application files yourself, subject to their source terms. The loader supports the common filenames `accepted_2007_to_2018Q4.csv.gz` and `rejected_2007_to_2018Q4.csv.gz`, plus equivalent CSV column spellings.

The common pre-decision covariates are:

- requested amount;
- debt-to-income ratio;
- FICO/risk score;
- employment length;
- state and three-digit ZIP prefix;
- normalized loan purpose;

The code deliberately excludes accepted-only underwriting and repayment fields such as grade, interest rate, installment, recoveries, and payment history. It also excludes `Policy Code` from modeling because it can encode a policy/eligibility determination rather than a pre-decision applicant attribute. `S=1` means accepted and issued. `Y=1` means `Charged Off` or `Default`; `Y=0` means `Fully Paid`. Other loan statuses are excluded. For rejected applications, the target is necessarily the counterfactual default outcome under loan issuance; the interpretation therefore inherits the paper's fixed-target-action convention. The recorded run contains 53,367 mature accepted loans and 337,277 rejected applications (390,644 total), including 8,644 accepted defaults. The frozen configuration still contains older illustrative `expected_counts` values of 27,157/206,161; the loader warns on that mismatch and records the observed counts. Those entries are retained to preserve the historical configuration hash. Exact source-file hashes and access instructions are in [the data guide](../docs/data.md).

The public accepted file supplies issue date, whereas the rejected file supplies application date. The 2012 filter therefore uses `issue_d` for accepted loans and `Application Date` for rejections. This timing mismatch is disclosed in the metadata, and calendar month is not used as a predictor.

## Run on Linux with an NVIDIA GPU

The historical wrapper requires an NVIDIA GPU and the pinned dependencies. It writes to the committed `artifacts/` paths, so use it in a separate clone. For a reproduction in this checkout with separate outputs, follow [the experiment guide](../docs/experiments.md). In a separate Linux clone, run:

```bash
bash run_gpu.sh /data/accepted_2007_to_2018Q4.csv.gz /data/rejected_2007_to_2018Q4.csv.gz
```

The runner creates `.venv`, installs pinned packages, runs lightweight preprocessing and certificate regression tests, checks that an NVIDIA GPU and CUDA-enabled XGBoost are available, prepares the cohort, runs Parts A and B, and creates the report. It stops before reading the full dataset or fitting models if any preflight test fails. Useful controls are:

Use Python 3.12 for the release environment. The original runner accepts Python 3.9 or newer and prints the interpreter before installing dependencies.

```bash
SKIP_INSTALL=1 RUN_STAGE=part-a bash run_gpu.sh /data/accepted.csv.gz /data/rejected.csv.gz
SKIP_INSTALL=1 RUN_STAGE=part-b bash run_gpu.sh /data/accepted.csv.gz /data/rejected.csv.gz
```

After a successful `prepare` stage, `RUN_STAGE=evaluate` reuses the prepared Parquet cohort and runs Part A, Part B, and reporting without rereading the raw CSV files.

For a short pipeline check on the Linux machine, lower `part_b.repetitions` temporarily or invoke the CLI directly with `--repetitions 1`. Do not use a one-repetition run for reported coverage.

## Outputs

All generated files go under `artifacts/`:

- `tables/part_a_certificates.csv`: lower/upper certificates, widths, thresholds, and accepted-only CVaR;
- `tables/part_a_breakdown.csv`: pairwise `Gamma_break` values;
- `tables/part_a_rankings.csv`: accepted-only versus upper-certificate rank changes;
- `tables/part_a_overlap_diagnostics.csv`: propensity overlap and clipping;
- `tables/part_a_subgroups.csv`: certificate widths for predeclared FICO, DTI, and amount strata;
- `tables/part_b_masking_certificates.csv`: truth, coverage, enlargement, and sharpness under known masking;
- `tables/part_b_dominance.csv`: correctness and abstention of certified pairwise dominance;
- `tables/part_b_fixed_audits.csv`: fixed-model, fixed-initial-threshold audit trajectories;
- `tables/part_b_adaptive_audits.csv`: historical round-by-round adaptive retraining results, superseded by the current sequential plug-in analysis;
- `tables/part_b_cost_to_width.csv`: historical fixed/adaptive labels and cost needed to hit the target width;
- `figures/`: publication-ready PNG and PDF figures;
- `RESULTS.md`: a numerical summary generated only from actual run artifacts.

The cross-fold t enlargement is intentionally identified as an empirical finite-sample enlargement, not an exact simultaneous confidence theorem. Likewise, adaptive retraining is an empirical extension: each round retrains the predictor, updates the known-label outcome and current label-observation nuisances, and reoptimizes the CVaR threshold, but no global active-learning optimality is claimed. The historical selected-law and selection nuisances remain fixed to the original `S`, because audited labels must not be relabeled as historical acceptances.

For acquisition from a pool already known to have `S=0`, the TIV score uses the conditional hidden-pool value `(p_hi-p_lo)|a1-a0|/(1-alpha)` and divides by audit cost. It does not multiply by `(1-pi)` a second time; that hidden-mass factor is needed when ranking population regions before conditioning on membership in the hidden pool.

## Historical recorded run

The original LendingClub tables here come from one run on 27 July 2026 on one GPU node of
Tsubame4, from a checkout at commit `50faeac` whose only local change was to `.gitignore`, with `config.yaml` as
committed (seed 20260727). The source files are `accepted_2007_to_2018Q4.csv.gz` and `rejected_2007_to_2018Q4.csv.gz`
from the Kaggle dataset "All Lending Club loan data" (Nathan George, CC0 Public Domain). The raw files, the prepared
cohort and the cached predictions are not tracked. `real_data_results_only.tar.gz` (SHA-256
`6565a0021443d164909d2d72bc47a9527cc76a6a7e33f2da3dca352cdc13285f`) holds `config.yaml` and every generated table, figure
and metadata file of that run. The following script recreates the historical three-panel figure, including the old adaptive panel; it does not generate the current paper's sequential figure:

```bash
python make_paper_figure.py artifacts artifacts/reproduction/historical_lendingclub_certificates
```

## Lightweight checks

These checks do not download data or fit models:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src tests
```

The current sequential figures and comparisons use `sequential_plugin/results/optimized/`; use [the figure guide](../docs/figures.md) to rebuild the current manuscript.
