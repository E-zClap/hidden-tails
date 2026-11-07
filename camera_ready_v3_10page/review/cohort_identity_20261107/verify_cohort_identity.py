"""Read-only identity check of the restored original cohort; no model fitting."""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PLUGIN = ROOT / "camera_ready_v3_10page/revision_experiments/sequential_plugin"
sys.path.insert(0, str(PLUGIN))
from real_cvar_plugin.models import _preprocessor
from real_cvar_plugin.part_b import impose_mask


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def file_record(path: Path, expected: str, authority: str) -> dict:
    actual = sha256(path)
    assert actual == expected, f"Input hash mismatch: {path}"
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "sha256": actual, "expected_sha256": expected, "matches_expected": True,
            "expected_hash_authority": authority}


def support(frame: pd.DataFrame) -> dict:
    return {
        "n": len(frame), "score_below_660": int(frame.risk_score.lt(660).sum()),
        "score_missing": int(frame.risk_score.isna().sum()),
        "score_ge_660": int(frame.risk_score.ge(660).sum()),
        "dti_ge_35": int(frame.dti.ge(35).sum()),
        "amount_gt_35000": int(frame.requested_amount.gt(35000).sum()),
        "missing_numeric": {name: int(frame[name].isna().sum()) for name in
                            ["requested_amount", "dti", "risk_score", "employment_years"]},
        "dated_state_diagnostics": {state: int(frame.state.eq(state).sum()) for state in
                                   ["ID", "IN", "IA", "ME", "MS", "NE", "ND", "TN"]},
    }


def main() -> None:
    restored = ROOT / "real_data_evaluation/artifacts/prepared/lendingclub_2012.parquet"
    reconstructed = PLUGIN / "data/accepted_2012_reconstructed.parquet"
    manifest_path = PLUGIN / "results/optimized/run_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    cfg = manifest["identity"]["config"]
    authority = "User's restored-cluster report, relayed by root; this check performs no live remote read."
    records = {
        "restored_original_prepared": file_record(restored,
            "6a9a44ee9338068d50bf3e0e773a0debab9cd1816369c8dbed639a0f15817f86", authority),
        "completed_run_reconstructed_input": file_record(reconstructed,
            manifest["identity"]["prepared_sha256"], "Frozen completed-run manifest"),
        "accepted_raw": file_record(PLUGIN / "data/accepted_2007_to_2018Q4.csv.gz",
            "55c16f75120f897683f02e7aabcf080d0e4a20c4832feb1d592cfa941bd62a2d", authority),
        "rejected_raw": file_record(ROOT / "camera_ready_v3_10page/review/empirical_repair_qa/raw_support_check/rejected_2007_to_2018Q4.csv.gz",
            "c1f9f2a4ec5537ef0af44b24b2fb2f41f58fcaec45363ef32c6a2fb27347f6c5", authority),
    }
    original = pd.read_parquet(restored)
    rerun = pd.read_parquet(reconstructed)
    accepted = original.loc[original.S.eq(1) & original.Y.notna()].reset_index(drop=True)
    rejected = original.loc[original.S.eq(0)].reset_index(drop=True)
    assert original.shape == (390644, 15)
    assert rerun.shape == (53367, 12)
    assert len(accepted) == 53367 and len(rejected) == 337277
    assert int(accepted.Y.sum()) == 8644 and rejected.Y.isna().all()
    shared = list(rerun.columns)
    projected = accepted.loc[:, shared]
    pd.testing.assert_frame_equal(projected, rerun, check_exact=True, check_dtype=True)
    restored_hash, reconstructed_hash = row_hash(projected), row_hash(rerun)
    assert restored_hash == reconstructed_hash == manifest["identity"]["pseudo_population_sha256"]
    extra = [column for column in original.columns if column not in shared]
    assert extra == ["subgroup_fico", "subgroup_dti", "subgroup_amount"]

    # Instantiate, but do not fit, both preprocessors used by the classifier families.
    preprocessors = {}
    for scale_numeric in [False, True]:
        transformer = _preprocessor(cfg, scale_numeric)
        columns = [column for _, _, selected in transformer.transformers for column in selected]
        assert transformer.remainder == "drop"
        assert not set(extra).intersection(columns)
        preprocessors[str(scale_numeric)] = {"selected_columns": columns, "remainder": transformer.remainder,
                                            "no_fit_performed": True}
    part_b_path = PLUGIN / "real_cvar_plugin/part_b.py"
    tree = ast.parse(part_b_path.read_text())
    accessed_columns = sorted({node.slice.value for node in ast.walk(tree)
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
        and node.value.id in {"data", "train", "test", "masked"}
        and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)})
    assert not set(extra).intersection(accessed_columns)
    source_checks = {}
    for relative, expected in manifest["identity"]["sources_sha256"].items():
        path = PLUGIN / relative.replace("\\", "/")
        source_checks[relative] = sha256(path) == expected
    assert all(source_checks.values())
    # The data loader creates the three subgroup columns only in add_subgroups;
    # model feature_columns reads numeric/categorical from the preserved config.
    original_data = ROOT / "real_data_evaluation/src/real_cvar/data.py"
    assert sha256(original_data) == sha256(PLUGIN / "real_cvar_plugin/data.py")

    recorded_masks = json.loads((PLUGIN / "results/optimized/mask_metadata.json").read_text())
    mask_checks = []
    for repetition in range(20):
        masked, metadata = impose_mask(accepted, cfg, repetition)
        train_idx, test_idx = train_test_split(
            np.arange(len(masked)), test_size=float(cfg["part_b"]["test_fraction"]),
            stratify=2 * masked.S.to_numpy(dtype=np.int8) + masked.Y.to_numpy(dtype=np.int8),
            random_state=int(cfg["seed"]) + repetition * 1297,
        )
        hashes = {"selection_sha256": hashlib.sha256(masked.S.to_numpy(dtype=np.int8).tobytes()).hexdigest(),
                  "train_indices_sha256": hashlib.sha256(train_idx.astype("<i8").tobytes()).hexdigest(),
                  "test_indices_sha256": hashlib.sha256(test_idx.astype("<i8").tobytes()).hexdigest()}
        reference = recorded_masks[repetition]
        assert all(hashes[key] == reference[key] for key in hashes)
        assert all(np.isclose(value, reference[key], rtol=0, atol=1e-12) for key, value in metadata.items())
        mask_checks.append({"repetition": repetition, "all_mask_and_split_hashes_match": True, **hashes})

    counts = {"combined": support(original), "accepted": support(accepted), "rejected": support(rejected)}
    assert counts["combined"]["score_below_660"] == counts["rejected"]["score_below_660"] == 180903
    assert counts["combined"]["score_missing"] == counts["rejected"]["score_missing"] == 23097
    assert counts["accepted"]["score_below_660"] == counts["accepted"]["score_missing"] == 0
    assert counts["rejected"]["dti_ge_35"] == 67491 and counts["accepted"]["dti_ge_35"] == 0
    assert counts["rejected"]["amount_gt_35000"] == 169 and counts["accepted"]["amount_gt_35000"] == 0
    assert manifest_path.read_bytes() == manifest_bytes
    report = {
        "checked_utc": datetime.now(timezone.utc).isoformat(), "checks_passed": True,
        "inputs": records, "restored_shape": list(original.shape), "accepted_shape": list(accepted.shape),
        "reconstructed_shape": list(rerun.shape), "shared_columns_in_original_order": shared,
        "exact_values_missingness_dtypes_and_row_order_equal": True,
        "restored_accepted_shared_columns_row_sha256": restored_hash,
        "reconstructed_row_sha256": reconstructed_hash,
        "completed_manifest_pseudo_population_sha256": manifest["identity"]["pseudo_population_sha256"],
        "row_hash_matches_completed_manifest": True, "extra_subgroup_columns": extra,
        "preprocessor_column_selection": preprocessors,
        "sequential_direct_dataframe_column_accesses": accessed_columns,
        "extra_subgroups_unused_by_models_masks_splits_and_costs": True,
        "all_current_production_sources_match_completed_manifest": source_checks,
        "original_data_loader_equals_used_snapshot": True,
        "all_20_mask_and_split_hashes_match_completed_run": mask_checks,
        "support_counts": counts, "accepted_default_count": int(accepted.Y.sum()),
        "support_interpretation": "Direct observed counts confirm the recorded score-support gap. DTI/amount/state aggregates are diagnostics, not a claim that a fully supported target cohort has been established.",
        "completed_manifest_unchanged_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "models_fitted": 0, "production_files_modified": False,
        "conclusion": "The completed repaired sequential experiment used exactly the original accepted cohort's 12 shared columns, including values, missingness, dtypes and row order. The additional three descriptive subgroup columns are unused. All 20 masks and splits are identical. No sequential rerun is needed.",
    }
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "cohort_identity_evidence.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"checks_passed": True, "restored_bytes": records["restored_original_prepared"]["bytes"],
                      "row_sha256": restored_hash, "accepted_rows": len(accepted), "defaults": int(accepted.Y.sum()),
                      "extra_columns": extra, "all_20_mask_split_hashes_match": True,
                      "rerun_needed": False}, indent=2))


if __name__ == "__main__":
    main()
