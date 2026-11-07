"""Verify full-size output equivalence before resuming the optimized rerun."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optionally save the verification JSON; otherwise print only")
    args = parser.parse_args()
    before_path = ROOT / "results" / "checkpoints" / "repetition_000.csv"
    after_path = ROOT / "results" / "optimized" / "checkpoints" / "repetition_000.csv"
    before, after = pd.read_csv(before_path), pd.read_csv(after_path)
    pd.testing.assert_frame_equal(before, after, check_exact=False, rtol=0.0, atol=1e-12)
    numeric = before.select_dtypes(include="number").columns.tolist()
    maxima = {name: float(np.max(np.abs(before[name] - after[name]))) for name in numeric}
    manifest = json.loads((ROOT / "results" / "run_manifest.json").read_text())
    preserved_hash = hashlib.sha256((ROOT / "real_cvar_plugin" / "part_b_unoptimized.py").read_bytes()).hexdigest()
    normalized_hashes = {key.replace("\\", "/"): value for key, value in manifest["identity"]["sources_sha256"].items()}
    expected_hash = normalized_hashes["real_cvar_plugin/part_b.py"]
    if preserved_hash != expected_hash:
        raise ValueError("Unoptimized source snapshot differs from its benchmark manifest.")
    report = {
        "rows_compared": len(before), "policies_compared": before.policy.unique().tolist(),
        "rounds_compared": before["round"].unique().tolist(),
        "all_columns_equal_at_atol_1e_12": True, "numeric_maximum_absolute_differences": maxima,
        "unoptimized_checkpoint_sha256": hashlib.sha256(before_path.read_bytes()).hexdigest(),
        "optimized_checkpoint_sha256": hashlib.sha256(after_path.read_bytes()).hexdigest(),
        "unoptimized_source_matches_benchmark_manifest": True,
        "unoptimized_repetition_seconds": manifest["repetition_seconds"]["0"],
        "optimized_repetition_seconds": json.loads((ROOT / "results" / "optimized" / "run_manifest.json").read_text())["repetition_seconds"]["0"],
    }
    original_path = ROOT.parents[2] / "real_data_evaluation" / "artifacts" / "tables" / "part_b_adaptive_audits.csv"
    if original_path.exists():
        original = pd.read_csv(original_path)
        comparison = after.merge(original.loc[original.repetition.eq(0)], on=["repetition", "policy", "round"], suffixes=("_new", "_original"))
        unaffected = comparison.loc[comparison.policy.ne("tiv_per_cost")]
        report["original_unaffected_policy_cvar_max_abs_difference"] = float(np.max(np.abs(unaffected.true_test_cvar_new - unaffected.true_test_cvar_original)))
        report["original_audited_label_counts_match"] = bool(np.array_equal(comparison.audited_labels_new, comparison.audited_labels_original))
        report["original_initial_cvar_max_abs_difference"] = float(np.max(np.abs(comparison.loc[comparison["round"].eq(0), "true_test_cvar_new"] - comparison.loc[comparison["round"].eq(0), "true_test_cvar_original"])))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
