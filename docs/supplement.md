# Supplementary material and reproducibility archive

The [main paper](../camera_ready_v3_10page/main.pdf) is the complete single-PDF
camera-ready manuscript: it includes all Appendices A–F and the NeurIPS checklist.
Use this PDF for the manuscript submission so that all textual supplementary
material is included in the submitted paper.

The [standalone supplement](../camera_ready_v3_10page/supplement.pdf) is a companion
reading copy of Appendices A–F. Both documents share the proof, experimental-method,
and additional-experiment sources. Keep the companion PDF for convenient reading;
use the reproducibility ZIP for the code, saved results, and editable sources.
Additional tables are generated from the released results; no new simulation or
GPU run is implied by this presentation update.

## Reading and evidence map

| Material | Location | Released evidence |
|---|---|---|
| Identification, statistical certification, and revelation proofs | Appendices A–C | `camera_ready_v3_10page/appendices/{A_population,B_statistics,C_revelation}.tex` |
| Oracle finite-sample certificates and learned-endpoint accuracy | Appendix D.2 | `experiments/finite_sample_certificates.py`, `experiments/results/finite_sample_{config.json,replications.csv,summary.csv}` |
| Historical applicants and controlled masking | Appendices D.3–D.4 | `real_data_evaluation/artifacts/tables/part_a_*.csv`, `part_b_masking_certificates.csv`, `part_b_dominance.csv`, and preparation/mask metadata |
| Synthetic fixed-threshold frontier | Appendix D.5 | `experiments/results/experiment6_certification_frontier_curves.csv` and `experiment6_cost_to_certify_ratios.csv` |
| Fixed and corrected sequential audits | Appendix D.6 | Historical `part_b_fixed_audits.csv`; current `sequential_plugin/results/optimized/` round records, checkpoints, and paired terminal summaries |
| Sensitivity calibration and extensions | Appendix E | `camera_ready_v3_10page/appendices/E_scope.tex` |
| Additional mechanism, predictor, and revelation comparisons | Appendix F.1 | `experiments/results/experiment1_*.csv`, `experiment2_*.csv`, and `experiment3_*.csv` |
| Fixed predictor references and the separately optimized-width diagnostic | Appendix F.2 | `experiments/results/experiment6_model_selection.csv` and `experiment6_optimized_bridge.csv` |
| Diabetes covariate example | Appendix F.3 | `experiments/results/experiment7_*.csv`, using 442 records grouped into 14 cells |
| Additional-table extraction | Appendix F source and provenance | `camera_ready_v3_10page/figures/scripts/prepare_additional_experiments.py`, `supplement_additional_tables.tex`, and `supplement_additional_provenance.json` |

Here `sequential_plugin/` abbreviates
`camera_ready_v3_10page/revision_experiments/sequential_plugin/`.
Appendix D already contains the finite-sample and repaired sequential additions.
Appendix F adds saved diagnostic evidence that was not presented there, while
keeping the current headline results and their numerical sources distinct.

The study targets differ. Oracle interval coverage, learned-endpoint accuracy,
empirical masking coverage, fixed-threshold ambiguity reduction, and sequential
predictive performance are separate quantities. The corrected sequential study
uses retained plug-in evaluation with frozen historical sensitivity bounds;
its 1,080 records cover 20 masks, six policies, and nine rounds. Those
trajectories do not establish adaptive confidence coverage.

Appendix F's predictor comparison uses fixed forecast rules; the rule named
`Certificate` in the saved tables is the preselected upper-probability forecast,
not the output of a general optimization over predictors. Its recorded
`optimized_width` diagnostic minimizes each endpoint over a 61-point cutoff
grid. It is distinct from an exact continuous-cutoff minimum and from the
fixed-threshold allocation guarantee. The diabetes example is a controlled
selective-label demonstration with an oracle-calibrated sensitivity value,
not a clinical validation study.

## Reproduce the current results and supplementary tables

Follow [experiments.md](experiments.md) for environments, full commands, and
scientific scope, and [data.md](data.md) for externally obtained LendingClub
inputs. The synthetic studies need no borrower data. Current experiment 6 and
7 CSVs correspond to the combined `--main-figure` run; some older JSON summaries
come from different runs and are not replacements for these CSVs.

The additional-table generator reads the released CSVs and records their
hashes and aggregation. Run from the repository root:

```bash
python camera_ready_v3_10page/figures/scripts/prepare_additional_experiments.py
python scripts/build.py --supplement
```

The supplement build first compiles the main manuscript to establish its
cross-references and citation numbering. To rebuild all vector figures and
both documents:

```bash
python scripts/build.py --figures --paper --supplement
```

Keep `main.pdf` alongside `supplement.pdf` when moving the documents so that
references to main-only material remain available. Figure extraction and its
checks are documented in [figures.md](figures.md).

## Create or check the portable ZIP

From an unmodified release checkout, before rebuilding timestamped PDFs:

```bash
python scripts/verify_release.py
python scripts/package_supplement.py
```

The default output is `dist/hidden_tails_supplement_vVERSION.zip`, with the
version read from `RELEASE_MANIFEST.json`. An explicit destination is also
supported:

```bash
python scripts/package_supplement.py --output dist/hidden_tails_supplement.zip
python -m zipfile -e dist/hidden_tails_supplement.zip delivery
cd delivery
python scripts/verify_release.py
```

The ZIP contains the exact release-manifest file set plus the manifest itself:
main and supplementary PDFs, editable LaTeX/TikZ sources, plotting tables,
experiment implementations, saved numerical outputs, configurations, tests,
dependency files, reproduction guides, licenses, and provenance. The archive
has a flat repository root, so commands remain the same after extraction.
No Git clone or network access is required to check its file hashes and saved
sequential comparisons.

Packaging checks every input hash, rejects unsafe paths and private/runtime
assets, and verifies the ZIP's exact member set, CRCs, and member hashes before
replacing the output. Unlisted files are never collected. This excludes raw
borrower files, prepared cohorts, virtual environments, caches, private Git
history, and manuscript backups. The ZIP itself is generated under ignored
`dist/` and is not recursively included in a later archive.

If a published file has been rebuilt or edited, packaging refuses its changed
checksum. Use a clean release checkout to recreate that release's ZIP. After
rebuilding figures or PDFs, `python scripts/verify_release.py --results-only`
still checks the saved numerical results. Maintainers create a new release
manifest when intentionally publishing updated files.

Focused packaging checks use temporary fixtures only:

```bash
python -m unittest discover -s scripts -p test_package_supplement.py -v
```

## Historical archives and current evidence

- `real_data_evaluation/real_data_results_only.tar.gz` preserves the original
  LendingClub outputs. Its older adaptive procedure is not the current
  sequential experiment.
- `sequential_plugin/results/optimized/` supplies the current sequential paper
  results. The separate `results/checkpoints/repetition_000.csv` and
  `results/run_manifest.json` are the unoptimized first-repetition benchmark
  used to verify the computational optimization.
- `experiments/results/run_summary_main_figure.json` matches the current
  experiment 6/7 CSVs. `run_summary.json` retains the earlier full-sequence
  outputs, including experiments 1–5; its experiment 6 values use another
  random-number stream. `run_summary_semisynthetic.json` is an older
  569-record malignancy example, not the current diabetes experiment.
- Host-path redactions and public helper changes are documented in
  [provenance_adaptations.json](provenance_adaptations.json) and
  [experiments.md](experiments.md). Scientific source/output hashes retain
  their recorded meaning. Reproducing GPU fits across platforms need not
  reproduce every trajectory bit-for-bit.

The released ZIP uses the portable public helpers and includes their benchmark
inputs. Older local manuscript-source snapshots are historical packaging, not
the current reproduction entry point.
