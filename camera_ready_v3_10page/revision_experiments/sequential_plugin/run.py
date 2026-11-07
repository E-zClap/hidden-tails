"""Sequential-only rerun with the committed masks, splits, seeds and settings."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

# Do not create caches beside either original or isolated source files.
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from aggregation import summarize
from real_cvar_plugin.config import load_config
from real_cvar_plugin.part_b import _adaptive_repetition, impose_mask

ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight(prepared: Path) -> dict:
    errors, versions = [], {}
    if not prepared.is_file():
        errors.append(f"prepared cohort missing: {prepared}")
    # These affect the fitted trajectory, input decoding or numerical summaries.
    required = {"numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "xgboost", "PyYAML", "joblib"}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        name, expected = line.strip().split("==")
        if name not in required:
            continue
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
        if versions[name] != expected:
            errors.append(f"{name}: require {expected}, installed {versions[name]}")
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        gpu = probe.stdout.strip()
        if probe.returncode != 0 or not gpu:
            errors.append("nvidia-smi did not report an available GPU")
    except (OSError, subprocess.TimeoutExpired):
        gpu = None
        errors.append("nvidia-smi unavailable")
    if versions.get("xgboost"):
        import xgboost
        if not xgboost.build_info().get("USE_CUDA", False):
            errors.append("XGBoost is not CUDA-enabled")
    return {"ready": not errors, "errors": errors, "versions": versions,
            "python": sys.version, "platform": platform.platform(), "gpu": gpu,
            "prepared": str(prepared)}


def validate_cohort(data: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    required = {"S", "Y", *cfg["data"]["features"]["numeric"], *cfg["data"]["features"]["categorical"]}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Prepared cohort missing columns: {sorted(missing)}")
    pseudo = data.loc[data["S"].eq(1) & data["Y"].notna()].reset_index(drop=True)
    reference = json.loads((ROOT / "reference_preparation_metadata.json").read_text())
    if len(pseudo) != reference["accepted_mature_count"] or int(pseudo.Y.sum()) != reference["default_count_among_accepted"]:
        raise ValueError("Input does not match original accepted cohort counts (53,367 rows; 8,644 defaults).")
    if not pseudo.Y.isin([0, 1]).all():
        raise ValueError("Accepted outcomes must be binary.")
    return pseudo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path, help="Original prepared lendingclub_2012.parquet (read only)")
    parser.add_argument("--output", type=Path, default=ROOT / "results", help="Directory inside this package's results/ subtree")
    parser.add_argument("--check-only", action="store_true", help="Audit prerequisites without loading data or fitting models")
    parser.add_argument("--resume", action="store_true", help="Resume completed repetition checkpoints with identical provenance")
    parser.add_argument("--max-new-repetitions", type=int, help="Stop after this many new checkpoints, for a reusable runtime benchmark")
    args = parser.parse_args()
    if args.max_new_repetitions is not None and args.max_new_repetitions < 1:
        parser.error("--max-new-repetitions must be positive")
    prepared, output = args.prepared.expanduser().resolve(), args.output.expanduser().resolve()
    allowed = (ROOT / "results").resolve()
    if not output.is_relative_to(allowed):
        parser.error(f"Output must be inside {allowed}")
    audit = preflight(prepared)
    print(json.dumps(audit, indent=2), flush=True)
    if args.check_only or not audit["ready"]:
        return 0 if audit["ready"] else 2

    cfg, _ = load_config(ROOT / "config.yaml")
    cfg["data"]["prepared_path"] = str(prepared)
    cfg["output"]["directory"] = str(output)
    data = pd.read_parquet(prepared)
    pseudo = validate_cohort(data, cfg)
    source_files = [ROOT / "run.py", ROOT / "aggregation.py", ROOT / "config.yaml", ROOT / "requirements.txt",
                    *sorted((ROOT / "real_cvar_plugin").glob("*.py"))]
    identity = {
        "prepared_sha256": sha256(prepared),
        "pseudo_population_sha256": hashlib.sha256(pd.util.hash_pandas_object(pseudo, index=True).values.tobytes()).hexdigest(),
        "sources_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_files},
        "config": cfg, "runtime": audit,
        "estimator": "retained_plugin", "orthogonal": False,
    }
    manifest_path = output / "run_manifest.json"
    if output.exists() and any(output.iterdir()):
        if not args.resume or not manifest_path.exists():
            parser.error("Output is nonempty; use a new results subdirectory or --resume with its manifest.")
        existing = json.loads(manifest_path.read_text())
        if existing["identity"] != identity:
            parser.error("Resume provenance differs from the current data, settings, source or environment.")
        manifest = existing
    else:
        output.mkdir(parents=True, exist_ok=True)
        manifest = {"identity": identity, "status": "running",
                    "started_utc": datetime.now(timezone.utc).isoformat(), "completed_repetitions": []}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    reference_masks = json.loads((ROOT / "reference_part_b_mask_metadata.json").read_text())
    frames, mask_rows = [], []
    new_repetitions = 0
    repetitions = int(cfg["part_b"]["repetitions"])
    rounds = int(cfg["part_b"]["adaptive_rounds"])
    for repetition in range(repetitions):
        masked, metadata = impose_mask(pseudo, cfg, repetition)
        reference = reference_masks[repetition]
        for key, value in metadata.items():
            if not np.isclose(value, reference[key], rtol=0.0, atol=1e-12):
                raise ValueError(f"Mask {repetition} differs from original metadata at {key}.")
        train_idx, test_idx = train_test_split(
            np.arange(len(masked)), test_size=float(cfg["part_b"]["test_fraction"]),
            stratify=2 * masked["S"].to_numpy(dtype=np.int8) + masked["Y"].to_numpy(dtype=np.int8),
            random_state=int(cfg["seed"]) + repetition * 1297,
        )
        metadata.update({
            "repetition": repetition,
            "selection_sha256": hashlib.sha256(masked.S.to_numpy(dtype=np.int8).tobytes()).hexdigest(),
            "train_indices_sha256": hashlib.sha256(train_idx.astype("<i8").tobytes()).hexdigest(),
            "test_indices_sha256": hashlib.sha256(test_idx.astype("<i8").tobytes()).hexdigest(),
        })
        mask_rows.append(metadata)
        checkpoint = checkpoints / f"repetition_{repetition:03d}.csv"
        if repetition in manifest["completed_repetitions"]:
            frame = pd.read_csv(checkpoint)
        else:
            print(f"Repetition {repetition + 1}/{repetitions}: six policies, {rounds + 1} rounds", flush=True)
            started = time.perf_counter()
            frame = pd.DataFrame(_adaptive_repetition(masked, cfg, repetition))
            elapsed = time.perf_counter() - started
            temporary = checkpoint.with_suffix(".tmp")
            frame.to_csv(temporary, index=False)
            temporary.replace(checkpoint)
            manifest["completed_repetitions"].append(repetition)
            manifest.setdefault("repetition_seconds", {})[str(repetition)] = elapsed
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            new_repetitions += 1
            print(f"Repetition {repetition + 1} checkpoint complete in {elapsed:.1f} seconds", flush=True)
        expected_rows = len(cfg["part_b"]["policies"]) * (rounds + 1)
        if len(frame) != expected_rows or set(frame["round"]) != set(range(rounds + 1)):
            raise ValueError(f"Incomplete repetition checkpoint: {checkpoint}")
        frames.append(frame)
        if (args.max_new_repetitions is not None and new_repetitions >= args.max_new_repetitions
                and len(manifest["completed_repetitions"]) < repetitions):
            print("Checkpoint limit reached. Continue with the same command plus --resume and without --max-new-repetitions.", flush=True)
            return 0
    trajectories = pd.concat(frames, ignore_index=True)
    trajectories.to_csv(output / "sequential_plugin_runs.csv", index=False)
    for name, frame in summarize(trajectories, rounds, repetitions).items():
        frame.to_csv(output / f"{name}.csv", index=False)
    (output / "mask_metadata.json").write_text(json.dumps(mask_rows, indent=2) + "\n")
    manifest.update({"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
                     "rows": len(trajectories),
                     "outputs_sha256": {path.name: sha256(path) for path in sorted(output.glob("*.csv"))}})
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Sequential-only rerun complete: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
