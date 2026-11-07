"""Reconstruct sequential inputs from the public accepted source, without rejected rows."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import numpy as np

from real_cvar_plugin.config import load_config
from real_cvar_plugin.data import _read_standardized, inspect_schema
from real_cvar_plugin.part_b import impose_mask
from run import sha256, validate_cohort

ROOT = Path(__file__).resolve().parent
SOURCE_URL = "https://www.kaggle.com/api/v1/datasets/download/wordsforthewise/lending-club/accepted_2007_to_2018Q4.csv.gz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "accepted_2007_to_2018Q4.csv.gz",
                        help="Accepted CSV.GZ obtained separately from the source URL in data/source_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "accepted_2012_reconstructed.parquet",
                        help="Destination Parquet file; a sibling .manifest.json records this reconstruction")
    parser.add_argument("--check-only", action="store_true",
                        help="Verify source bytes, SHA-256 and schema without preparing rows or writing files")
    args = parser.parse_args()
    source, prepared = args.source.expanduser().resolve(), args.output.expanduser().resolve()
    if not source.is_file():
        parser.error(f"Accepted source missing: {source}. Obtain the file listed in data/source_manifest.json.")
    reference_source = json.loads((ROOT / "data" / "source_manifest.json").read_text())
    source_hash = sha256(source)
    if source.stat().st_size != reference_source["source_bytes"] or source_hash != reference_source["source_sha256"]:
        parser.error("Source bytes/SHA-256 differ from the published source_manifest.json; use the recorded source snapshot.")
    if source == prepared:
        parser.error("--output must differ from --source")
    cfg, _ = load_config(ROOT / "config.yaml")
    mapping = inspect_schema(source, accepted=True)
    if args.check_only:
        print(json.dumps({"ready": True, "source_sha256": source_hash, "schema_mapping": mapping}, indent=2))
        return
    data = _read_standardized(source, mapping, True, int(cfg["cohort_year"]), int(cfg["data"]["chunksize"]))
    data["source_row_id"] = "A" + data.index.astype(str)
    pseudo = validate_cohort(data, cfg)
    reference = json.loads((ROOT / "reference_part_b_mask_metadata.json").read_text())
    checks = []
    for repetition in range(int(cfg["part_b"]["repetitions"])):
        masked, metadata = impose_mask(pseudo, cfg, repetition)
        for key, value in metadata.items():
            if not np.isclose(value, reference[repetition][key], rtol=0.0, atol=1e-12):
                raise ValueError(f"Mask {repetition} differs at {key}: {value} versus {reference[repetition][key]}")
        checks.append({"repetition": repetition, "matches_original_metadata": True,
                       "selection_sha256": hashlib.sha256(masked.S.to_numpy(dtype=np.int8).tobytes()).hexdigest(), **metadata})
    prepared.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(prepared, index=False)
    manifest = {
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": SOURCE_URL, "source_bytes": source.stat().st_size,
        "source_sha256": source_hash, "prepared_sha256": sha256(prepared),
        "loader_sha256": sha256(ROOT / "real_cvar_plugin" / "data.py"),
        "accepted_mature_count": len(data), "default_count": int(data.Y.sum()),
        "schema_mapping": mapping, "mask_checks": checks,
        "original_selected_features_and_order_preserved": True,
        "excluded_unused_columns": ["subgroup_fico", "subgroup_dti", "subgroup_amount"],
        "note": "Original accepted-loader output and A-prefixed row identifiers. Combined-cohort descriptive subgroups are not used by any sequential fit, mask, split or metric and are not reconstructed.",
    }
    prepared.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
