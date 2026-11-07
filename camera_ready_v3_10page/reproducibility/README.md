# Reproduction material for the supplementary PDF

The current portable release is the reproducibility ZIP described in [docs/supplement.md](../../docs/supplement.md). Its repository-root layout contains both PDFs, manuscript/figure sources, experiment code, saved results, environments, and verification tools. No raw borrower records are bundled.

Start from the extracted archive root:

```bash
python scripts/verify_release.py
python camera_ready_v3_10page/figures/scripts/prepare_additional_experiments.py --check
python scripts/build.py --supplement
```

Use [the experiment guide](../../docs/experiments.md) for full reruns, recorded environments, and distinctions between current and archived outputs. The current sequential experiment is `camera_ready_v3_10page/revision_experiments/sequential_plugin/results/optimized/`. Its full 1,080 rows, paired summaries, checkpoints, unoptimized benchmark checkpoint, and validation helpers are included.

`implementation_inventory.tex` is a portable companion inventory, not an additional appendix input. The standalone PDF reuses Appendices A–E and adds Appendix F; its tables are generated from released CSVs. Current experiment 6/7 data match the combined `--main-figure` run. The older full-run frontier and 569-record malignancy summary are archival.

The release archive uses an explicit SHA-256 file list. Rebuilds can change PDF hashes, so verify the unmodified archive before rebuilding. Public helper changes preserve original numerical implementations and recorded scientific evidence while making paths and verification commands portable; see `docs/provenance_adaptations.json`.
