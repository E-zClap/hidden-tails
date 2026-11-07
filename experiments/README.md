# Synthetic experiments

This directory contains finite-cell Monte Carlo, population and learning illustrations, a certification-frontier stress test, and a semi-synthetic scikit-learn diabetes example. Source data for these experiments are generated from committed seeds or bundled with scikit-learn; no LendingClub download is needed.

From the repository root in the pinned Python environment:

```bash
python experiments/run_tail_observability_experiments.py --help
python experiments/finite_sample_certificates.py --help
```

See [the experiment guide](../docs/experiments.md) for the complete commands, mode-dependent random-number streams, original environment details, and output locations. Saved scientific CSVs and summaries are under `results/`. The population runner writes back to this directory's `results/` and `figures/`; use a separate clone for that rerun. The finite-sample runner supports `--out experiments/reproduction` for separate output.

Current manuscript figure reconstruction uses the editable figure sources and plotting tables described in [the figure guide](../docs/figures.md).
