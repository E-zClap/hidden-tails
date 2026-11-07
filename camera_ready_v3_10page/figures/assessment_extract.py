"""Recreate the source tables for Figures 2--3 and the assessment supplements.

Run from any directory with `python path/to/assessment_extract.py`.
This script reads archived experiment outputs and never modifies them.
Only Python's standard library is required.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
SOURCES = {
    "learning": ROOT / "experiments/results/finite_sample_summary.csv",
    "historical": ROOT / "real_data_evaluation/artifacts/tables/part_a_certificates.csv",
    "masking": ROOT / "real_data_evaluation/artifacts/tables/part_b_masking_certificates.csv",
}
MODELS = ("gradient_boosting", "logistic", "conservative_logistic")
ALPHAS = (0.9, 0.95)
GAMMAS = (1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0)


def read(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write(name, fields, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def boolean(value):
    assert value in ("True", "False"), value
    return value == "True"


def coverage(rows, expected):
    result = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[float(row["gamma"])].append(row)
    assert tuple(sorted(grouped)) == GAMMAS
    for gamma in GAMMAS:
        group = grouped[gamma]
        assert len(group) == expected, (gamma, len(group), expected)
        assert {int(row["repetition"]) for row in group} == set(range(20))
        plugin_count = sum(boolean(row["covered"]) for row in group)
        fold_count = sum(boolean(row["enlarged_covered"]) for row in group)
        result.append({
            "gamma": gamma,
            "plugin": plugin_count / expected,
            "fold": fold_count / expected,
            "plugin_count": plugin_count,
            "fold_count": fold_count,
            "n_model_mask_rows": expected,
            "n_masks": 20,
        })
    return result


def main():
    learning = read(SOURCES["learning"])
    historical = read(SOURCES["historical"])
    masking = read(SOURCES["masking"])
    assert len(learning) == 5 and len(historical) == 576 and len(masking) == 960
    assert {row["model"] for row in historical} == set(MODELS)
    assert {row["model"] for row in masking} == set(MODELS)
    assert {row["enlargement_method"] for row in masking} == {"crossfit_fold_t"}
    keys = [(int(r["repetition"]), r["model"], float(r["alpha"]), float(r["gamma"])) for r in masking]
    assert len(keys) == len(set(keys)), "Duplicate model-mask-alpha-Gamma rows"
    for row in masking:
        assert boolean(row["covered"]) == (float(row["lower"]) <= float(row["true_cvar"]) <= float(row["upper"]))
        assert boolean(row["enlarged_covered"]) == (float(row["enlarged_lower"]) <= float(row["true_cvar"]) <= float(row["enlarged_upper"]))
        assert boolean(row["is_true_gamma"]) == (float(row["gamma"]) == 3.0)

    learning_fields = ["n", "id_width", "cert_width", "crossfit_rmse_upper", "crossfit_rmse_lower"]
    learning_rows = []
    for row in sorted(learning, key=lambda r: int(r["n"])):
        assert int(row["replications"]) == 4800
        result = {field: row[field] for field in learning_fields}
        result["sqrt_n_ref"] = 0.055 * math.sqrt(1000 / int(row["n"]))
        learning_rows.append(result)
    assert len({r["id_width"] for r in learning_rows}) == 1
    write("learning_summary.csv", learning_fields + ["sqrt_n_ref"], learning_rows)
    enlargement_polygon = [{"n": r["n"], "width": r["cert_width"]} for r in learning_rows]
    enlargement_polygon.extend({"n": r["n"], "width": r["id_width"]} for r in reversed(learning_rows))
    enlargement_polygon.append(enlargement_polygon[0].copy())
    write("learning_width_enlargement.csv", ["n", "width"], enlargement_polygon)

    all_population = [r for r in historical if r["subset_type"] == "all" and r["subset"] == "all"]
    assert len(all_population) == 48
    interval_rows = []
    for position, model in enumerate(MODELS):
        matched = [r for r in all_population if r["model"] == model and float(r["alpha"]) == 0.95 and float(r["gamma"]) == 2.0]
        assert len(matched) == 1
        row = matched[0]
        interval_rows.append({"model": model, "y": 3 - position, "accepted": row["accepted_only_cvar"], "lower": row["lower"], "upper": row["upper"]})
    write("assessment_historical.csv", ["model", "y", "accepted", "lower", "upper"], interval_rows)

    coverage_fields = ["gamma", "plugin", "fold", "plugin_count", "fold_count", "n_model_mask_rows", "n_masks"]
    pooled = coverage([r for r in masking if float(r["alpha"]) == 0.95], 60)
    write("assessment_coverage.csv", coverage_fields, pooled)
    for model in MODELS:
        for alpha in ALPHAS:
            suffix = f"{model}_{round(alpha * 100)}"
            matched = sorted([r for r in all_population if r["model"] == model and float(r["alpha"]) == alpha], key=lambda r: float(r["gamma"]))
            assert tuple(float(r["gamma"]) for r in matched) == GAMMAS
            rows = [{"gamma": r["gamma"], "accepted": r["accepted_only_cvar"], "lower": r["lower"], "upper": r["upper"]} for r in matched]
            write(f"assessment_sensitivity_{suffix}.csv", ["gamma", "accepted", "lower", "upper"], rows)
            polygon = [{"gamma": r["gamma"], "risk": r["upper"]} for r in matched]
            polygon.extend({"gamma": r["gamma"], "risk": r["lower"]} for r in reversed(matched))
            polygon.append(polygon[0].copy())
            write(f"assessment_sensitivity_polygon_{suffix}.csv", ["gamma", "risk"], polygon)
            rows = coverage([r for r in masking if r["model"] == model and float(r["alpha"]) == alpha], 20)
            write(f"assessment_coverage_{suffix}.csv", coverage_fields, rows)

    provenance = {
        "script": "assessment_extract.py",
        "sources": {name: {"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in SOURCES.items()},
        "learning": {"rows": 5, "replications_per_n": 4800, "identification_floor": learning[0]["id_width"], "reference_guide": "sqrt_n_ref = 0.055 * sqrt(1000/n); a visual n^(-1/2) guide normalized to 0.055 at n = 1000, not a fit or uncertainty bound", "transformations": "Original mean width and endpoint RMSE columns copied without rounding or smoothing. Any fill between cert_width and id_width denotes their difference, not a confidence band.", "width_enlargement_polygon": "learning_width_enlargement.csv contains the five cert_width coordinates in increasing n, the five id_width coordinates in decreasing n, and the initial coordinate repeated to close the polygon; it encodes only the difference between the saved mean widths."},
        "historical": {"population_filter": "subset_type == all and subset == all", "source_population_n": sorted({int(r["n"]) for r in all_population}), "source_selected_n": sorted({int(r["n_selected"]) for r in all_population}), "main_figure_filter": {"alpha": 0.95, "gamma": 2.0}, "interval_columns": ["lower", "upper"], "point_column": "accepted_only_cvar", "row_order_top_to_bottom": list(MODELS), "conservative_logistic_label": "Default-weighted logistic regression", "supplemental_alpha_values": list(ALPHAS), "gamma_values": list(GAMMAS), "sensitivity_polygon": "Each assessment_sensitivity_polygon_{model}_{90|95}.csv has eight upper endpoint coordinates in increasing Gamma, eight lower endpoint coordinates in decreasing Gamma, and the initial coordinate repeated to close the polygon. Shading spans the linearly connected plotting coordinates of the saved sensitivity endpoints; it is not an uncertainty band."},
        "masking": {"alpha_main": 0.95, "masks": 20, "predictors": 3, "rows_per_gamma_main": 60, "aggregation": "Unweighted mean of covered or enlarged_covered over 20 masking repetitions times three predictors at each Gamma; equivalently mean across mask-level averages, because each mask has all three predictors.", "supplemental_aggregation": "Unweighted mean across 20 masking repetitions separately for each predictor and alpha.", "dependence": "The three predictors share each masking repetition; 60 rows are not 60 independent masks. No uncertainty bands are computed.", "enlargement_method": "crossfit_fold_t", "true_gamma": 3.0, "gamma_values": list(GAMMAS), "validation": "Every Boolean coverage flag was recomputed from its saved interval and true_cvar and matched exactly; all model-mask-alpha-Gamma keys are unique."},
        "checks": "Original source hashes are recorded. No experiment was rerun. Source data were only read; derived CSV outputs retain original decimal values where copied. All expected counts and Gamma grids are asserted.",
    }
    (OUT / "assessment_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"historical": interval_rows, "pooled_coverage": pooled, "output_csv_count": 22, "source_sha256": {k: v["sha256"] for k, v in provenance["sources"].items()}}, indent=2))


if __name__ == "__main__":
    main()
