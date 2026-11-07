"""Verify release files and saved sequential results without external packages."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "camera_ready_v3_10page/revision_experiments/sequential_plugin/results/optimized"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def table(name):
    with (RESULTS / name).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-only", action="store_true", help="Skip file hashes after rebuilding outputs")
    args = parser.parse_args()
    if not args.results_only:
        manifest = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
        for name, expected in manifest["files"].items():
            path = (ROOT / name).resolve()
            require(ROOT in path.parents, f"Manifest path outside release: {name}")
            require(path.is_file(), f"Missing release file: {name}")
            require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"Changed release file: {name}")
        print(f"Release checksums: {len(manifest['files'])} files verified")
    rows = table("sequential_plugin_runs.csv")
    require(len(rows) == 1080, "Expected 1,080 sequential observations")
    policies = {"tiv_per_cost", "random", "uncertainty", "low_propensity", "predicted_loss", "sensitivity_width"}
    expected_keys = {(r, p, t) for r in range(20) for p in policies for t in range(9)}
    indexed = {(int(r["repetition"]), r["policy"], int(r["round"])): r for r in rows}
    require(set(indexed) == expected_keys, "Missing, duplicate, or unexpected trajectory coordinates")
    for row in rows:
        lo, hi, width = (float(row[name]) for name in ("estimated_retained_lower", "estimated_retained_upper", "estimated_retained_width"))
        require(0 <= lo <= hi <= 1, "Endpoint range violation")
        require(math.isclose(hi - lo, width, abs_tol=1e-12), "Endpoint/width mismatch")
        require(0 <= float(row["mean_test_brier"]) <= float(row["true_test_cvar"]) <= 1, "Brier/CVaR ordering violation")
        require(float(row["alpha"]) == .95 and float(row["gamma"]) == 3, "Unexpected sensitivity or tail level")
    pairs = table("terminal_pairs.csv")
    metrics = {"estimated_retained_width", "mean_test_brier", "true_test_cvar"}
    require(len(pairs) == 60 and {(int(p["repetition"]), p["metric"]) for p in pairs} == {(r,m) for r in range(20) for m in metrics}, "Incomplete paired terminal table")
    for pair in pairs:
        rep, metric = int(pair["repetition"]), pair["metric"]
        tiv, random = indexed[(rep,"tiv_per_cost",8)], indexed[(rep,"random",8)]
        require(int(pair["round"]) == 8, "Unexpected terminal round")
        require(tiv["audited_labels"] == random["audited_labels"] == pair["audited_labels"], "Unmatched terminal budgets")
        for name, value in (("tiv_value", float(tiv[metric])), ("random_value", float(random[metric])), ("tiv_minus_random", float(tiv[metric])-float(random[metric]))):
            require(math.isclose(float(pair[name]), value, abs_tol=1e-12), f"Paired mismatch: {rep}/{metric}/{name}")
    print("Saved results: 20 masks x 6 policies x 9 rounds; 60 paired comparisons verified")


if __name__ == "__main__":
    main()
