"""Focused replay of the sole baseline trajectory differing materially from the old host."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from real_cvar_plugin.config import load_config
from real_cvar_plugin.models import fit_probability_model
from real_cvar_plugin import part_b_unoptimized
from run import preflight, validate_cohort

ROOT = Path(__file__).resolve().parent
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True, help="Reconstructed accepted cohort Parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "platform_replay", help="Directory for new replay artifacts")
    parser.add_argument("--check-only", action="store_true", help="Audit prerequisites without reading the cohort or fitting models")
    args = parser.parse_args()
    prepared, output = args.prepared.expanduser().resolve(), args.output.expanduser().resolve()
    audit = preflight(prepared)
    print(json.dumps(audit, indent=2), flush=True)
    if args.check_only or not audit["ready"]:
        return 0 if audit["ready"] else 2
    output.mkdir(parents=True, exist_ok=True)
    cfg, _ = load_config(ROOT / "config.yaml")
    repetition = 15
    pseudo = validate_cohort(pd.read_parquet(prepared), cfg)
    masked, _ = part_b_unoptimized.impose_mask(pseudo, cfg, repetition)
    train_idx, test_idx = train_test_split(
        np.arange(len(masked)), test_size=float(cfg["part_b"]["test_fraction"]),
        stratify=2 * masked["S"].to_numpy(dtype=np.int8) + masked["Y"].to_numpy(dtype=np.int8),
        random_state=int(cfg["seed"]) + repetition * 1297,
    )
    train = masked.iloc[train_idx].reset_index(drop=True)
    test = masked.iloc[test_idx].reset_index(drop=True)
    selected = train["S"].to_numpy(dtype=bool)
    y = train["Y"].to_numpy(dtype=np.int8)
    started = time.perf_counter()
    p1, _ = fit_probability_model(
        cfg["evaluation"]["nuisance_outcome_model"], cfg, train.loc[selected], y[selected], train,
        int(cfg["seed"]) + repetition * 100003 + 7001,
    )
    pi, _ = fit_probability_model(
        cfg["evaluation"]["nuisance_propensity_model"], cfg, train, selected.astype(np.int8), train,
        int(cfg["seed"]) + repetition * 100003 + 7002,
    )
    p1 = np.clip(p1, *cfg["evaluation"]["p1_clip"])
    pi = np.clip(pi, *cfg["evaluation"]["propensity_clip"])
    replay = pd.DataFrame(part_b_unoptimized._adaptive_policy(
        train, test, selected, p1, pi, "low_propensity", cfg, repetition,
    ))
    replay.to_csv(output / "platform_replay_low_propensity_rep15.csv", index=False)
    optimized = pd.read_csv(ROOT / "results/optimized/sequential_plugin_runs.csv")
    optimized = optimized.loc[optimized.repetition.eq(repetition) & optimized.policy.eq("low_propensity")].reset_index(drop=True)
    numeric = replay.select_dtypes(include="number").columns.tolist()
    maxima = {column: float(np.max(np.abs(replay[column] - optimized[column]))) for column in numeric}
    pd.testing.assert_frame_equal(replay, optimized, check_exact=False, rtol=0, atol=1e-12)
    report = {"repetition": repetition, "policy": "low_propensity", "seconds": time.perf_counter() - started,
              "unoptimized_replay_matches_optimized_at_atol_1e_12": True,
              "maximum_absolute_differences": maxima,
              "interpretation": "The unused-fit optimization does not explain this original-host difference. Exact historical audited row identities and fitted models were not retained, so the precise cross-platform divergence cannot be reconstructed from the original results-only archive."}
    (output / "platform_variation_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
