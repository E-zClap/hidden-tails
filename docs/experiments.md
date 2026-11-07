# Experiment reproduction

Run the commands below from the repository root unless a block explicitly changes directory. Use Python 3.12 with the root `requirements.txt` in a virtual environment, as described in the root README. The historical scientific settings, numerical implementations, and seeds are retained. Raw LendingClub records are acquired separately; see [data.md](data.md).

## Checks without LendingClub data or a GPU

On Linux/macOS:

```bash
PYTHONPATH=real_data_evaluation/src python -m unittest discover -s real_data_evaluation/tests -v
python -m unittest discover -s camera_ready_v3_10page/revision_experiments/sequential_plugin/tests -v
python camera_ready_v3_10page/revision_experiments/sequential_plugin/verify_completed_run.py
python camera_ready_v3_10page/revision_experiments/sequential_plugin/validate_optimization.py
```

In PowerShell, set `$env:PYTHONPATH='real_data_evaluation/src'` before the first command and omit its `PYTHONPATH=...` prefix. The other commands are identical. The first suite has six tests; the second has 15, including small CPU logistic-regression fixtures and checks that help/preflight do not launch an experiment. The completed-run verifier checks 1,080 rows, endpoint identities, paired terminal results, recorded output hashes, and the preserved historical source hashes. The optimization verifier compares the two included first-repetition checkpoints (54 rows), without fitting models. Both print JSON and only save a new report when passed `--output FILE`.

These suites and verifiers passed in the release preparation environment (Python 3.12.4 and the pinned requirements). All CLI help commands listed below were checked. Data and GPU preflight also passed against the original local input files in read-only mode. No full scientific experiment was rerun during release preparation.

```bash
python experiments/run_tail_observability_experiments.py --help
python experiments/finite_sample_certificates.py --help
python camera_ready_v3_10page/revision_experiments/sequential_plugin/prepare_accepted.py --help
python camera_ready_v3_10page/revision_experiments/sequential_plugin/run.py --help
```

The standard-library release checks and the current paper/figure builds are described in the root README and [figures.md](figures.md).

## Synthetic and semi-synthetic experiments

| Command | Experiment | Seed and scale |
|---|---|---|
| `python experiments/run_tail_observability_experiments.py` | All seven population/learning/revelation studies | Master seed 20260503; constants in the source |
| `python experiments/run_tail_observability_experiments.py --main-figure` | Conceptual figure, 120-world certification frontier, and diabetes case | Seed 20260503; 120 cells per frontier world |
| `python experiments/run_tail_observability_experiments.py --semisynthetic` | Diabetes selective-label case only | Seed 20260503; built-in scikit-learn dataset |
| `python experiments/finite_sample_certificates.py --out experiments/reproduction` | Finite-cell certificate Monte Carlo | Seed 20261103; 60 worlds, 80 replications per size, 30 cells, five sample sizes |

The first script writes to `experiments/results/` and `experiments/figures/`; rerun it in a separate clone to preserve the published outputs. Its modes consume the random generator differently. Use the relevant mode when comparing its corresponding saved `run_summary*.json`; outputs from different modes need not be byte-identical.

The finite-sample command writes a separate output directory, preserving the published CSVs. Its committed settings are alpha .75, Gamma 2, minimum propensity .2, delta .05, and sample sizes 1,000/3,000/10,000/30,000/100,000. `--worlds 2 --reps 2 --n-grid 100 300 --out experiments/reproduction-smoke` is an optional plumbing check, not the reported experiment. `--plot-only --out experiments` redraws the archived summary without simulation and overwrites only that script's figure outputs. Current manuscript figure generation is documented separately in [figures.md](figures.md).

`experiments/results/finite_sample_config.json` records the historical run's Python 3.9.25 and NumPy 2.0.2, whereas the common release environment pins NumPy 1.26.4 for LendingClub. The recorded environment is part of the provenance; the common installation is a runnable environment, not a promise of byte-identical simulation CSVs across NumPy versions. Exact archived numerical values are included.

To target the recorded finite-sample numerical environment separately, install Python **3.9.25**, verify that version, and run:

```bash
python3.9 -m venv .venv-finite
.venv-finite/bin/python -m pip install -r experiments/requirements-finite-sample.txt
.venv-finite/bin/python experiments/finite_sample_certificates.py --out experiments/reproduction
```

On Windows use `py -3.9 -m venv .venv-finite` and `.venv-finite/Scripts/python.exe` in the last two commands, after verifying that the selected interpreter is 3.9.25. This optional requirements file pins the recorded NumPy 2.0.2 plus Matplotlib 3.9.4 for plotting. The historical Matplotlib and other transitive dependency versions were not recorded; the plotting pin is a release convenience, not a claim about the original environment. The original numerical environment was not recreated during release preparation.

## Historical LendingClub assessment and controlled masking

`real_data_evaluation/` preserves the July 2026 implementation and outputs. Part A evaluates actual selective labels; rejected outcomes are unobserved. Part B imposes known masking on the 53,367 mature accepted loans and includes static masking, fixed-predictor audits, and the older adaptive retraining procedure. Its `part_b_adaptive_*.csv` and adaptive entries in `part_b_cost_to_width.csv` are historical results; they are not the repaired sequential experiment used by the current paper.

For a full historical rerun, first create a separate configuration with only output paths changed:

```bash
python -c "from pathlib import Path; import yaml; p=Path('real_data_evaluation/config.yaml'); c=yaml.safe_load(p.read_text()); c['output']['directory']='artifacts/reproduction'; c['data']['prepared_path']='artifacts/reproduction/prepared/lendingclub_2012.parquet'; p.with_name('config.reproduction.yaml').write_text(yaml.safe_dump(c, sort_keys=False))"
```

Then, on Linux/macOS, supply the separately obtained files:

```bash
PYTHONPATH=real_data_evaluation/src python -m real_cvar.cli prepare --config real_data_evaluation/config.reproduction.yaml --accepted /data/accepted_2007_to_2018Q4.csv.gz --rejected /data/rejected_2007_to_2018Q4.csv.gz
PYTHONPATH=real_data_evaluation/src python -m real_cvar.cli part-a --config real_data_evaluation/config.reproduction.yaml --device cuda
PYTHONPATH=real_data_evaluation/src python -m real_cvar.cli part-b --config real_data_evaluation/config.reproduction.yaml --device cuda
PYTHONPATH=real_data_evaluation/src python -m real_cvar.cli report --config real_data_evaluation/config.reproduction.yaml
```

On PowerShell set `$env:PYTHONPATH='real_data_evaluation/src'`, remove those command prefixes, and replace `/data/...` with your actual input paths. Paths in a configuration resolve relative to its location. `evaluate` combines Part A, Part B, and reporting using a prepared cohort; `all` additionally prepares the input. `--device cpu` and `--repetitions 1` are available for exploratory checks but change the reported hardware/protocol. The full configuration uses five folds, 20 masks, 450-tree gradient boosting, and seed 20260727; it is substantially more expensive than the tests above.

`real_data_evaluation/run_gpu.sh` is the historical Linux wrapper. It checks the NVIDIA driver, pinned package versions, unit tests, and a tiny CUDA pipeline before running its stage. It uses `config.yaml` and writes the original `artifacts/` paths; use it in a disposable clone if preserving the published outputs. The archived `make_paper_figure.py` plots the old adaptive panel and is not the current paper's sequential figure builder.

## Current sequential retained plug-in experiment

Use the reconstruction in [data.md](data.md), then run from its package directory:

```bash
cd camera_ready_v3_10page/revision_experiments/sequential_plugin
python run.py --prepared data/accepted_2012_reconstructed.parquet --check-only
python run.py --prepared data/accepted_2012_reconstructed.parquet --output results/reproduction
```

`--check-only` checks file existence, exact required package versions, the NVIDIA GPU, and CUDA-enabled XGBoost, without reading the cohort or fitting models. It exits 2 with actionable errors when prerequisites are missing. The full run subsequently validates the cohort counts and all 20 mask metadata records before each repetition.

The complete run preserves all 20 masks, 80/20 stratified splits, six policies, rounds 0–8, 2% hidden-pool batches, alpha .95, Gamma 3, unit costs, historical nuisance seeds, and seed 20260727. Its retained plug-in endpoints use the empirical losses of all currently known labels and the historical sensitivity bounds for remaining hidden labels. This differs from the earlier adaptive residual-corrected objective. It is a descriptive sequential study, without a simultaneous adaptive-coverage claim.

For an interrupted new run, repeat the same command with `--resume`; completed checkpoints are reused only when source, data, configuration, and runtime identity match. `--max-new-repetitions 1` creates a reusable first-repetition benchmark; omit that limit when resuming the full run. Use a new output directory rather than trying to resume the published `results/optimized/` record. Published manifests preserve scientific hashes but have host paths redacted for release, and a new checkout/environment has a different identity.

The published run used Python 3.12.4, the pinned packages, CUDA XGBoost 2.1.4, and an RTX 3060 with 12 GiB. Its 1,480 fits took approximately 2,138 seconds. GPU-dependent fitting can change subsequent acquisitions. One low-propensity trajectory differs from the older host by up to .001958 held-out CVaR; the preserved unoptimized implementation reproduces the repaired trajectory within 8.4e-17, and exact accepted-input equality rules out different input records. The specific original-host cause is unresolved. See `platform_variation_check.json` and the cohort identity evidence.

`investigate_platform_variation.py --prepared data/accepted_2012_reconstructed.parquet --check-only` performs prerequisites only. Without `--check-only` it refits the single repetition-15 low-propensity trajectory and writes new replay artifacts under `data/platform_replay/`; it is optional and not part of the lightweight verification.

The current results are the 1,080 rows in `results/optimized/sequential_plugin_runs.csv` and their paired summaries. At round 8 the TIV-minus-random paired mean differences are -.032887 estimated width, +.001424 held-out mean Brier loss, and -.014763 held-out CVaR. This is an accuracy–tail-risk tradeoff. The descriptive across-mask intervals are not simultaneous confidence guarantees.

## Public-release portability changes

The scientific estimators, learners, masks, split logic, aggregation, seeds, and committed scientific configurations are byte-identical to their source snapshots. Release changes are limited to documentation, packaging and helper entry points:

- `prepare_accepted.py` now accepts explicit source/output paths, checks the recorded source hash, supports `--check-only`, and saves a separate reconstruction manifest.
- `verify_completed_run.py` normalizes historical Windows path separators for other platforms; it and `validate_optimization.py` show help and print reports without overwriting the frozen verification records.
- `investigate_platform_variation.py` has an explicit CLI/main guard, external input path and preflight; importing it or requesting help cannot start GPU fitting.
- The historical `real_data_evaluation/make_paper_figure.py` has argument help and creates the requested output directory; its plot calculations are unchanged.
- Three CLI regression tests and the small unoptimized benchmark checkpoint/manifest complete the public verification paths. Host paths in historical JSON provenance were redacted; [provenance_adaptations.json](provenance_adaptations.json) records the affected hashes.

Figure extraction/build portability changes are described in [figures.md](figures.md).
