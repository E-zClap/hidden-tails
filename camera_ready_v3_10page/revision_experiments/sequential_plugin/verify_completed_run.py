"""Independent consistency checks for the completed repaired experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "optimized"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optionally save the verification JSON; otherwise print only")
    args = parser.parse_args()
    manifest = json.loads((OUTPUT / "run_manifest.json").read_text())
    assert manifest["status"] == "complete", "Full rerun is not complete"
    assert manifest["completed_repetitions"] == list(range(20))
    frame = pd.read_csv(OUTPUT / "sequential_plugin_runs.csv")
    assert len(frame) == 1080
    assert not frame.duplicated(["repetition", "policy", "round"]).any()
    for _, trajectory in frame.groupby(["repetition", "policy"]):
        assert trajectory["round"].tolist() == list(range(9))
        assert trajectory.audited_labels.is_monotonic_increasing
    assert frame["policy"].nunique() == 6
    assert frame.estimator.eq("retained_plugin").all()
    assert frame.alpha.eq(0.95).all() and frame.gamma.eq(3.0).all()
    assert np.allclose(frame.cumulative_cost, frame.audited_labels, rtol=0, atol=1e-12)
    assert (frame.estimated_retained_lower >= -1e-12).all()
    assert (frame.estimated_retained_upper <= 1 + 1e-12).all()
    assert (frame.estimated_retained_width >= -1e-12).all()
    assert np.allclose(frame.estimated_retained_upper - frame.estimated_retained_lower,
                       frame.estimated_retained_width, rtol=0, atol=1e-12)
    assert (frame.mean_test_brier <= frame.true_test_cvar + 1e-12).all()
    for _, initial in frame.loc[frame["round"].eq(0)].groupby("repetition"):
        for metric in ["estimated_retained_lower", "estimated_retained_upper", "mean_test_brier", "true_test_cvar"]:
            assert initial[metric].max() - initial[metric].min() < 1e-12
    terminal = frame.loc[frame["round"].eq(8)]
    reported = pd.read_csv(OUTPUT / "terminal_comparison.csv").set_index("metric")
    independent = {}
    for metric in ["estimated_retained_width", "mean_test_brier", "true_test_cvar"]:
        pivot = terminal.pivot(index="repetition", columns="policy", values=metric)
        difference = pivot.tiv_per_cost.to_numpy() - pivot.random.to_numpy()
        assert abs(float(np.mean(difference)) - reported.loc[metric, "paired_difference_mean"]) < 1e-12
        assert abs(float(np.median(difference)) - reported.loc[metric, "paired_difference_median"]) < 1e-12
        independent[metric] = {"tiv_median": float(np.median(pivot.tiv_per_cost)),
                               "random_median": float(np.median(pivot.random)),
                               "paired_mean": float(np.mean(difference)),
                               "paired_median": float(np.median(difference))}
    output_hashes = {name: hashlib.sha256((OUTPUT / name).read_bytes()).hexdigest() == digest
                     for name, digest in manifest["outputs_sha256"].items()}
    assert all(output_hashes.values())
    originals = json.loads((ROOT / "original_source_sha256.json").read_text())
    original_root = ROOT.parents[2]
    preserved = {name: hashlib.sha256((original_root / name.replace("\\", "/")).read_bytes()).hexdigest() == digest
                 for name, digest in originals.items()}
    assert all(preserved.values())
    # Every non-TIV policy's acquisition rule is unaffected by the estimator change.
    old = pd.read_csv(original_root / "real_data_evaluation/artifacts/tables/part_b_adaptive_audits.csv")
    comparison = frame.merge(old, on=["repetition", "policy", "round"], suffixes=("_new", "_old"))
    unaffected = comparison.loc[comparison.policy.ne("tiv_per_cost")]
    report = {"checks_passed": True, "rows": len(frame), "repetitions": 20,
              "policies": 6, "rounds_per_policy": 9, "terminal_comparison": independent,
              "output_hashes_match": output_hashes, "original_sources_unchanged": preserved,
              "original_audited_labels_match": bool(np.array_equal(comparison.audited_labels_new, comparison.audited_labels_old)),
              "original_unaffected_cvar_max_abs_difference": float(np.max(np.abs(unaffected.true_test_cvar_new - unaffected.true_test_cvar_old))),
              "total_fit_seconds": float(sum(manifest["repetition_seconds"].values())),
              "prepared_sha256": manifest["identity"]["prepared_sha256"]}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
