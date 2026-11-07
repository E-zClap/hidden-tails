# Current sequential retained plug-in experiment

This directory contains a self-contained sequential-only rerun. Production estimation neither imports from nor writes to the original `real_data_evaluation/`, `experiments/`, or `tail-risk-certification-experiments/` directories. Regression checks can read the original implementation and tables. The complete rerun writes to `results/optimized/`; the unoptimized repaired-method benchmark remains separately in `results/checkpoints/`. Full source-data access instructions and exact commands are in [the data guide](../../../docs/data.md) and [experiment guide](../../../docs/experiments.md). Raw records and prepared cohorts are excluded from this public repository.

## Scientific change

For the current predictor and threshold, let `a_y = max((f-y)^2-eta, 0)`, let `K=S or audited`, and let `H=not K`. The retained plug-in objective is

```
eta + mean(K * a_Y + H * ((1-q) * a_0 + q * a_1)) / (1-alpha).
```

For the upper endpoint, `q` is the historical sensitivity upper bound where `a_1 >= a_0` and the lower bound otherwise; the lower endpoint reverses these choices. Each endpoint is minimized over the exact empirical hinge breakpoints. Because binary Brier losses preserve their ordering after taking a positive part, the existing endpoint optimizer applies directly.

The sequential call now sets `orthogonal=False`. Historical `p1` and `pi` remain fixed to the original selection mask. Revealed labels enter the empirical known-loss term; the remaining hidden rows retain the historical sensitivity bounds. Full revelation therefore makes both endpoints equal the full-training-data empirical CVaR. This descriptive objective has no added adaptive coverage claim.

The original `orthogonal=True` call retained the original selected residual correction after revelation. Before this repair, the real-model regression fixture returned a lower endpoint of `0.04550021964824364` instead of full-data empirical CVaR `0.409947514176745`. The same trajectory test now passes.

## Preserved protocol

`config.yaml` and `requirements.txt` are byte-for-byte copies of the original files. `certificates.py`, `models.py`, `config.py`, and `data.py` are unmodified local snapshots. `part_b.py` contains only the original sequential dependencies, with the estimator call, metric export, endpoint column names, and unused-fit optimization changed. Original source hashes are in `original_source_sha256.json`.

The runner retains seed 20260727, all 20 masks, the original 80/20 stratified train/test splits, six acquisition policies in their original order, nine observations per trajectory (rounds 0–8), batches of 2% of the original hidden training pool, 450-tree gradient boosting, alpha .95, Gamma 3, and unit label costs. It also preserves the historical-nuisance and per-round fitting seeds, current-nuisance fits used by the baselines, and acquisition tie handling. The refreshed propensity fit is computed only for `low_propensity`; the refreshed outcome fit only for `predicted_loss`. Other policies never consume those fitted values. This removes 1,800 unused fits without changing any scientific setting. `part_b_unoptimized.py` preserves the unoptimized repaired source. All six policies' full-size first-repetition outputs are byte-identical before and after this optimization; `optimization_equivalence.json` records the check. Changing the TIV threshold can change subsequent acquired labels, so its complete trajectory must be rerun.

The original preparation metadata records 53,367 accepted mature loans and 8,644 defaults. Those observed counts, rather than the older illustrative `expected_counts` entries in the original config, are enforced. Each generated mask is checked against the original run's stored numerical metadata. The rerun additionally records the input hash, pseudo-population hash, selection and train/test index hashes, source hashes, environment, and output hashes.

## Source reconstruction and rerun

The public accepted source was recovered from the Kaggle dataset used by the original experiment: 392,582,231 bytes, SHA-256 `55c16f75120f897683f02e7aabcf080d0e4a20c4832feb1d592cfa941bd62a2d`. `prepare_accepted.py` reconstructs the accepted mature 2012 rows with the byte-identical original loader and original row order. It verifies 53,367 accepted loans, 8,644 defaults, and all 20 original mask metadata records at absolute tolerance 1e-12 before saving. `data/source_manifest.json` records the URL, source/prepared hashes and checks. Combined-cohort descriptive subgroup columns require rejected rows and are unused by the sequential analysis; they are explicitly excluded. The original results-only archive lacks the input cohort and fitted/predicted values, but the original prepared cohort has now been restored separately from the cluster.

The separately restored historical combined cohort had 390,644 rows, 15 columns, 5,215,764 bytes and SHA-256 `6a9a44ee9338068d50bf3e0e773a0debab9cd1816369c8dbed639a0f15817f86`. Its 53,367 accepted rows equal the reconstructed input exactly on all 12 shared columns, including dtypes and row order. Their shared-column row hash is `9b252b44866f94f03fa74423595302e925495a24428f35011567c8991a2d6021`, matching `results/optimized/run_manifest.json`. The three additional descriptive subgroup columns are unused by the models. Evidence is in [`../../review/cohort_identity_20261107/cohort_identity_evidence.json`](../../review/cohort_identity_20261107/cohort_identity_evidence.json). That check compared local files with the supplied historical GPU-cluster hashes; it did not perform a new remote read or model rerun.

The completed rerun used the original accepted records; the later cohort restoration did not require another run. The combined and accepted-only files are excluded from this public release. Follow the reconstruction command below after obtaining the accepted source separately. The preparation helper checks source SHA-256, cohort counts and mask metadata, then writes a new sibling `.manifest.json` instead of changing the published `data/source_manifest.json`.

From this directory, after activating the Python 3.12 environment with the pinned requirements:

```bash
python prepare_accepted.py --source /data/accepted_2007_to_2018Q4.csv.gz --check-only
python prepare_accepted.py --source /data/accepted_2007_to_2018Q4.csv.gz --output data/accepted_2012_reconstructed.parquet
python -m unittest discover -s tests -v
python run.py --prepared data/accepted_2012_reconstructed.parquet --check-only
python run.py --prepared data/accepted_2012_reconstructed.parquet --output results/reproduction
```

Replace `/data/...` with the actual external source path. Preparation is CPU-only; the sequential runner enforces the reported CUDA/pinned-package prerequisites. `--check-only` performs no model fitting. The full run performs 1,480 fits after omitting unused baseline nuisances. If interrupted, repeat its command with `--resume`; checkpoints are reused only when data, code, settings and environment match. `--max-new-repetitions 1` creates a reusable runtime benchmark. Output is restricted to this directory's `results/` subtree. Use a new reproduction subdirectory; published manifests have host paths redacted and cannot be resumed from a different checkout/environment. A future run using the full 15-column cohort likewise has a different input identity despite matching scientific model inputs.

The actual rerun uses a dedicated `.venv` with every pinned package version, Python 3.12.4, CUDA-enabled XGBoost 2.1.4, and an RTX 3060 with 12 GiB on Windows. The unoptimized first repetition took 257.06 seconds; the optimized repetition took 112.86 seconds and produced a byte-identical checkpoint. All 20 optimized repetitions completed in 2,137.68 seconds of fitting. The expected XGBoost warning about CPU NumPy prediction inputs does not change the CUDA training setting.

All policies' initial held-out CVaRs, and every random, uncertainty, and predicted-loss CVaR, reproduce the original recorded values exactly. Sensitivity-width CVaRs differ by at most 1.20e-8. One low-propensity trajectory (repetition 15, rounds 3–8) differs from the original host, with maximum CVaR difference .001958. An independent replay using the preserved unoptimized repaired implementation reproduces the new trajectory within 8.4e-17, so the unused-fit optimization is not responsible. Exact equality with the restored original cohort also rules out input-data differences. The original archive lacks audited row identities and fitted models needed to locate the remaining original-host divergence; a particular hardware or software cause has not been established. See `platform_variation_check.json` and the later cohort-identity check above. All mask metadata and revealed-label counts match the original run.

## Completed terminal comparison

All 20 paired terminal comparisons favor TIV in estimated width and held-out CVaR, and favor random acquisition in mean Brier loss. The risk metrics therefore show an accuracy–tail-risk tradeoff. Differences below are TIV minus random at round 8; the interval is a descriptive across-mask t interval for the paired mean.

| Metric | TIV median | Random median | Paired mean difference | Descriptive 95% interval |
|---|---:|---:|---:|---:|
| Estimated retained width | .034778 | .067069 | -.032887 | [-.033873, -.031901] |
| Mean held-out Brier | .136159 | .134706 | +.001424 | [.001195, .001653] |
| Held-out CVaR | .854064 | .868167 | -.014763 | [-.016293, -.013233] |

`completed_run_verification.json` verifies all 1,080 rows, 20 masks, six policies, nine rounds, endpoint/width identities, held-out metric ordering, terminal pairing, output hashes and original-source preservation. The full trajectories and summaries are in `results/optimized/`.

## Outputs and interpretation

- `sequential_plugin_runs.csv`: 1,080 rows with `estimated_retained_lower`, `estimated_retained_upper`, `estimated_retained_width`, `mean_test_brier`, and `true_test_cvar`, plus budgets and thresholds.
- `trajectory_summary.csv`: mean, median, and linear IQR for each recorded round/policy/metric.
- `terminal_pairs.csv`: TIV-minus-random differences within each mask at the predeclared terminal round 8.
- `terminal_comparison.csv`: paired mean/median/IQR, a descriptive across-mask t interval for the mean difference, and lower/equal/higher counts. Smaller differences favor TIV. These summaries quantify empirical precision and predictive performance under the masking design; they do not establish incremental detection power.
- `constant_reference.csv`: analytic mean Brier and CVaR .25, endpoints [.25,.25], width 0 and TIV 0. This is a predictor reference, not an acquisition policy.
- `mask_metadata.json`, `run_manifest.json`, and `checkpoints/`: reproducibility and restart records.

The aggregation checks matching terminal masks, sensitivity levels, tail levels, and revealed-label budgets before calculating paired differences. Mean Brier and CVaR are evaluated only on the reserved test split. Widths come from the training-pool retained objectives and are explicitly labeled estimates.

## Verification

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest discover -s camera_ready_v3_10page/revision_experiments/sequential_plugin/tests -v
```

The 15-test regression suite verifies complete revelation, the explicit historical-bound objective, the constant reference, each round's held-out Brier/CVaR, paired aggregation, invalid pairing/budgets, missing-input preflight, output-path isolation, unused-fit counts, and exact six-policy equivalence using real model fits. The original six lightweight tests also pass unchanged. `verification.json` retains the historical failing regression and earlier check records. The release preparation ran the current 15 tests, the original six tests, source/GPU preflight, and both saved-output verifiers without rerunning full experiments. The additional tests check portable help and missing-source errors. The original failure can be reproduced from this checkout with `SEQUENTIAL_TEST_ORIGINAL=1`; this switch is only for demonstrating the original regression, not for a repaired run.

The following checks use only included tables and manifests, print JSON, and do not refit or overwrite the frozen reports:

```bash
python verify_completed_run.py
python validate_optimization.py
```

Pass `--output FILE` to save a new verification report. `investigate_platform_variation.py --prepared data/accepted_2012_reconstructed.parquet --check-only` checks prerequisites for the optional single-trajectory replay; omitting `--check-only` performs GPU fits and writes new artifacts under `data/platform_replay/`.
