"""Build supplementary tables from the recorded experiments, without rerunning fits.

Uses only Python's standard library. Run with --check to verify committed outputs.
The source root defaults to the repository, falling back to the source-ZIP snapshot.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, median

PAPER = Path(__file__).resolve().parents[2]
NAMES = [
    "experiment1_rank_reversal.csv", "experiment2_learning_raw.csv",
    "experiment2_learning_summary.csv", "experiment2_choice_frequencies.csv",
    "experiment3_region_tiv_scores.csv", "experiment3_tiv_audit.csv",
    "experiment6_model_selection.csv", "experiment6_optimized_bridge.csv",
    "experiment6_certification_frontier_curves.csv",
    "experiment7_semisynthetic_cells.csv", "experiment7_semisynthetic_model_selection.csv",
    "experiment7_semisynthetic_optimized_bridge.csv",
    "experiment7_semisynthetic_certification_curves.csv", "run_summary_main_figure.json",
]


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def val(row, key):
    return float(row[key])


def near(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)


def fmt(value, digits=4):
    return f"{float(value):.{digits}f}"


def table(command, columns, header, rows, caption, label):
    return "\n".join([
        rf"\newcommand{{\{command}}}{{%", r"\begin{table}[!htbp]",
        r"\centering\small\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{{columns}}}", r"\toprule", header, r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}", "}", "",
    ])


def build(source_root):
    results = source_root / "experiments/results"
    data = {name: read_csv(results / name) for name in NAMES if name.endswith(".csv")}
    summary = json.loads((results / "run_summary_main_figure.json").read_text(encoding="utf-8"))
    mappings = {
        ("experiment6", "models"): "experiment6_model_selection.csv",
        ("experiment6", "bridge"): "experiment6_optimized_bridge.csv",
        ("experiment6", "curves"): "experiment6_certification_frontier_curves.csv",
        ("experiment7", "cells"): "experiment7_semisynthetic_cells.csv",
        ("experiment7", "models"): "experiment7_semisynthetic_model_selection.csv",
        ("experiment7", "bridge"): "experiment7_semisynthetic_optimized_bridge.csv",
        ("experiment7", "curves"): "experiment7_semisynthetic_certification_curves.csv",
    }
    for (experiment, part), name in mappings.items():
        recorded = summary[experiment][part]
        assert len(recorded) == len(data[name]), name
        for expected, actual in zip(recorded, data[name]):
            assert set(expected) == set(actual), name
            for key, value in expected.items():
                assert near(value, actual[key]) if isinstance(value, (float, int)) else value == actual[key], (name, key)

    raw = data["experiment2_learning_raw.csv"]
    learned = data["experiment2_learning_summary.csv"]
    choices = data["experiment2_choice_frequencies.csv"]
    assert len(raw) == 1000 and len(learned) == len(choices) == 4
    for row in learned:
        selected = [item for item in raw if item["learner"] == row["learner"]]
        assert len(selected) == 250
        assert near(fmean(val(item, "true_cvar") for item in selected), row["true_cvar_mean"])
    assert all(int(row["count"]) == 250 and near(row["frequency"], 1) for row in choices)
    audit = data["experiment3_tiv_audit.csv"]
    assert len(audit) == 80 and len({row["policy"] for row in audit}) == 5
    for row in audit:
        assert near(val(row, "upper") - val(row, "lower"), row["width"])
    synth = data["experiment6_model_selection.csv"]
    diabetes = data["experiment7_semisynthetic_model_selection.csv"]
    cells = data["experiment7_semisynthetic_cells.csv"]
    assert len(synth) == 360 and len(diabetes) == 3 and len(cells) == 14
    assert sum(int(row["n"]) for row in cells) == 442
    assert near(sum(val(row, "mass") for row in cells), 1)
    assert len({row["gamma_eval"] for row in cells}) == 1
    bridges = [data["experiment6_optimized_bridge.csv"], data["experiment7_semisynthetic_optimized_bridge.csv"]]
    assert [len(rows) for rows in bridges] == [2520, 21]
    for rows in bridges:
        assert all(-1e-12 <= val(row, "optimized_width") <= val(row, "uniform_bound") + 1e-12 for row in rows)
    for name in ["experiment6_certification_frontier_curves.csv", "experiment7_semisynthetic_certification_curves.csv"]:
        rows = data[name]
        tiv = {(row.get("world"), row["budget_fraction"]): row for row in rows if row["policy"] == "TIV/cost"}
        oracle = {(row.get("world"), row["budget_fraction"]): row for row in rows if row["policy"] == "Oracle frontier"}
        assert tiv.keys() == oracle.keys()
        assert all(near(row["debt"], oracle[key]["debt"]) for key, row in tiv.items())

    hidden = next(row for row in data["experiment1_rank_reversal.csv"] if row["predictor"] == "f hidden-tail")
    out = ["% Generated by figures/scripts/prepare_additional_experiments.py; do not edit.\n"]
    for name, value in [("AdditionalHiddenObserved", fmt(hidden["observed_label_cvar"])),
                        ("AdditionalHiddenFull", fmt(hidden["true_full_population_cvar"])),
                        ("AdditionalDiabetesGamma", fmt(cells[0]["gamma_eval"], 2)),
                        ("AdditionalDiabetesSelected", fmt(sum(val(row, "mass") * val(row, "reveal_probability") for row in cells)))]:
        out.append(rf"\newcommand{{\{name}}}{{{value}}}" + "\n")
    by_learner = {row["learner"]: row for row in learned}
    order = [("Observed-label CVaR", "Observed labels", ".25"), ("IPW CVaR", "IPW", ".25"),
             ("Conservative envelope", "Upper envelope", ".65"), ("Full-label oracle CVaR", "Full-label oracle", ".65")]
    rows = []
    for key, label, f_b in order:
        row = by_learner[key]
        nums = [fmt(row[k]) for k in ["observed_cvar_mean", "true_cvar_mean", "upper_envelope_mean", "width_mean"]]
        rows.append(" & ".join([label, f_b, *nums]) + r"\\")
    out.append(table("SupplementFiniteChoiceTable", "@{}lrcccc@{}",
        r"Rule & $f(B)$ & Observed CVaR & Full CVaR & Upper endpoint & Width\\", rows,
        r"Finite-candidate selection over 250 samples of size 1,400. Values average exact population evaluations of the chosen predictor; each rule selected the same candidate in every replication. The upper endpoint is a population sensitivity envelope, without statistical enlargement.",
        "tab:additional-selection"))

    policies = [("Top TIV/cost", "TIV"), ("Random", "Random"), ("High density", "Density"),
                ("Uncertainty", "Uncertainty"), ("High predicted loss", "Predicted loss")]
    rows = []
    for metric, title in [("relative_fixed_threshold_width_reduction", "Fixed-threshold gap removed"), ("relative_width_reduction", "Separately optimized empirical width removed")]:
        rows.append(r"\multicolumn{6}{l}{\emph{" + title + r"} (\%)}\\")
        for budget in [600, 1200, 1800]:
            nums = [fmt(100 * val(next(row for row in audit if row["policy"] == policy and int(row["budget"]) == budget), metric), 1) for policy, _ in policies]
            rows.append(" & ".join([f"{budget:,}", *nums]) + r"\\")
        if metric.startswith("relative_fixed"): rows.append(r"\midrule")
    out.append(table("SupplementFixedAuditTable", "@{}rccccc@{}",
        r"Labels & TIV & Random & Density & Uncertainty & Predicted loss\\", rows,
        r"One synthetic pool of 7,000 records, fixed predictor, and unit cost per hidden label. Reductions are normalized by that pool's initial gap or initial optimized width. These two targets need not rank policies identically; at 600 labels random revelation removes slightly more optimized width than TIV.",
        "tab:additional-audit"))

    rows = []
    roles = [("Observed-selected", r"Selected-label probability $p_1$"), ("Certificate", r"Sensitivity upper probability $q_+(p_1)$"), ("Oracle", r"Full-label probability $p_{\rm full}$")]
    for source, title in [(synth, "Synthetic populations: mean over 120 worlds"), (diabetes, "Diabetes-derived finite population")]:
        rows.append(r"\multicolumn{4}{l}{\emph{" + title + r"}}\\")
        for role, label in roles:
            subset = [row for row in source if row["role"] == role]
            nums = [fmt(fmean(val(row, key) for row in subset)) for key in ["observed_cvar", "true_cvar", "upper_certificate"]]
            rows.append(" & ".join([label, *nums]) + r"\\")
        if source is synth: rows.append(r"\midrule")
    out.append(table("SupplementReferenceForecastTable", "@{}lccc@{}",
        r"Specified forecast & Observed CVaR & Full CVaR & Upper endpoint\\", rows,
        r"Three specified reference forecasts, evaluated with absolute loss and $\alpha=.85$. The CSV role names ``Observed-selected,'' ``Certificate,'' and ``Oracle'' identify these formulas; they do not denote optimization over a candidate class. Synthetic entries use $\Gamma=12$; diabetes entries use its oracle-chosen sensitivity level.",
        "tab:additional-forecasts"))

    rows = []
    for budget in [0, .25, .5, .75, 1]:
        nums = []
        for source in bridges:
            subset = [row for row in source if near(row["budget_fraction"], budget)]
            assert len(subset) == (120 if source is bridges[0] else 1)
            nums += [fmt(median(val(row, key) for row in subset)) for key in ["optimized_width", "uniform_bound"]]
        rows.append(" & ".join([f"{100*budget:.0f}\\%", *nums]) + r"\\")
    out.append(table("SupplementGridBridgeTable", "@{}rcccc@{}",
        r" & \multicolumn{2}{c}{Synthetic median} & \multicolumn{2}{c}{Diabetes population}\\" + "\n" +
        r"Full-cost budget & Grid width & $W_{f,0}$ & Grid width & $W_{f,0}$\\", rows,
        r"Width connection along the fixed-$\eta=.5$ TIV-per-cost allocation. Each endpoint is minimized on the recorded 61-point cutoff grid. Synthetic columns are separate medians across 120 worlds; diabetes has one finite population. The sufficient bound is $W_{f,0}$, which is not itself a CVaR risk and can exceed one.",
        "tab:additional-bridge"))

    curves = data["experiment7_semisynthetic_certification_curves.csv"]
    policy_order = ["TIV/cost", "Oracle frontier", "Random", "High density", "Low reveal prob.", "High pred. loss", "Uncertainty", "Width only", "Contrast only"]
    display = {"Oracle frontier": "Fractional-knapsack frontier", "Low reveal prob.": "Lowest observation probability", "High pred. loss": "Predicted absolute loss", "Width only": "Sensitivity width", "Contrast only": "Excess-loss contrast"}
    rows = []
    for policy in policy_order:
        subset = [row for row in curves if row["policy"] == policy and val(row, "relative_debt") <= .25]
        assert subset,policy
        budget = min(val(row, "budget_fraction") for row in subset)
        rows.append(display.get(policy,policy) + " & " + f"{100*budget:.0f}\\%" + r"\\")
    out.append(table("SupplementDiabetesCostTable", "@{}lr@{}",
        r"Allocation rule & First recorded cost to 25\% gap\\", rows,
        r"Diabetes-derived population: first hit on the 2\%-spaced full-cost grid. Random is the mean remaining gap over 100 random cell orderings, so its entry is the first crossing of that mean curve, not the mean of 100 crossing costs. Acquisition is divisible at the cell level.",
        "tab:additional-diabetes-cost"))

    tex = "".join(out)
    provenance = {
        "description": "Supplement-only recorded mechanism, frontier-diagnostic, and diabetes tables; no experiment rerun.",
        "source_files": {"experiments/results/" + name: hashlib.sha256((results / name).read_bytes()).hexdigest() for name in NAMES},
        "current_frontier_and_diabetes_run": "python experiments/run_tail_observability_experiments.py --main-figure",
        "current_frontier_and_diabetes_summary": "experiments/results/run_summary_main_figure.json",
        "mechanism_run": "python experiments/run_tail_observability_experiments.py (full-sequence experiments 1-3)",
        "archival_summaries_not_used": ["experiments/results/run_summary.json: experiment6", "experiments/results/run_summary_semisynthetic.json: earlier 569-record malignancy example"],
        "numeric_checks": {"experiment2_records": len(raw), "experiment3_records": len(audit), "experiment6_worlds": 120,
                           "experiment6_bridge_records": len(bridges[0]), "diabetes_records": 442, "diabetes_cells": len(cells),
                           "diabetes_gamma": val(cells[0], "gamma_eval"), "diabetes_bridge_records": len(bridges[1]),
                           "current_summary_matches_csvs": True, "all_grid_widths_bounded_by_W_f0": True,
                           "tiv_and_fractional_frontier_match": True},
        "interpretation": ["Experiment1 is not a pairwise rank reversal.", "Experiment6/7 reference forecasts are specified formulas, not argmin-selected learners.",
                           "Recorded bridge endpoints use a 61-point cutoff grid.", "Diabetes observation is imposed; measured progression is binarized; no held-out clinical validation.",
                           "No finite-sample coverage claim is added."],
    }
    return tex, json.dumps(provenance, indent=2) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    root = args.source_root or PAPER.parent
    if not (root / "experiments/results/experiment7_semisynthetic_cells.csv").is_file():
        root = PAPER / "reproducibility/project"
    tex, provenance = build(root)
    targets = [(PAPER / "figures/supplement_additional_tables.tex", tex),
               (PAPER / "figures/supplement_additional_provenance.json", provenance)]
    for path, content in targets:
        if args.check:
            assert path.read_bytes() == content.encode("utf-8"), f"Stale generated output: {path}"
        else:
            path.write_bytes(content.encode("utf-8"))
    print("Additional experiment tables and numeric checks: PASS" + (" (no files written)" if args.check else ""))


if __name__ == "__main__":
    main()
