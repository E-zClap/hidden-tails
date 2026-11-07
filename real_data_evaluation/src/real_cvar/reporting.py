from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required result table is missing: {path}")
    return pd.read_csv(path)


def _save(fig: plt.Figure, directory: Path, name: str) -> None:
    fig.savefig(directory / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def build_report(cfg: dict[str, Any]) -> None:
    output = Path(cfg["output"]["directory"])
    tables = output / "tables"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")

    part_a = _read(tables / "part_a_certificates.csv")
    all_a = part_a[part_a["subset_type"] == "all"]
    fig, axes = plt.subplots(1, len(cfg["evaluation"]["alphas"]), figsize=(8.0, 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, alpha in zip(axes, map(float, cfg["evaluation"]["alphas"])):
        cur = all_a[np.isclose(all_a["alpha"], alpha)]
        sns.lineplot(data=cur, x="gamma", y="width", hue="model", marker="o", ax=ax)
        ax.set_title(rf"$\alpha={alpha:.2f}$")
        ax.set_xscale("log")
        ax.set_xlabel(r"Sensitivity $\Gamma$")
        ax.set_ylabel("Certificate width")
    fig.tight_layout()
    _save(fig, figures, "part_a_width_vs_gamma")

    subgroup = _read(tables / "part_a_subgroups.csv")
    max_alpha = max(map(float, cfg["evaluation"]["alphas"]))
    gamma0 = float(cfg["part_b"]["gamma0"])
    plot_sub = subgroup[np.isclose(subgroup["alpha"], max_alpha) & np.isclose(subgroup["gamma"], gamma0)]
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    sns.barplot(data=plot_sub, x="subset", y="width", hue="model", ax=ax)
    ax.set_xlabel("Predeclared subgroup")
    ax.set_ylabel("Certificate width")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    _save(fig, figures, "part_a_subgroup_widths")

    static = _read(tables / "part_b_masking_certificates.csv")
    true_gamma = static[static["is_true_gamma"].astype(str).str.lower().eq("true")]
    coverage = (
        true_gamma.groupby(["alpha", "model"])[["covered", "enlarged_covered", "width", "enlarged_width"]]
        .mean()
        .reset_index()
    )
    coverage.to_csv(tables / "part_b_coverage_summary.csv", index=False)
    melted = coverage.melt(
        id_vars=["alpha", "model"],
        value_vars=["covered", "enlarged_covered"],
        var_name="interval",
        value_name="coverage",
    )
    fig, axes = plt.subplots(1, len(cfg["evaluation"]["alphas"]), figsize=(9.0, 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, alpha in zip(axes, map(float, cfg["evaluation"]["alphas"])):
        sns.barplot(data=melted[np.isclose(melted["alpha"], alpha)], x="model", y="coverage", hue="interval", ax=ax)
        ax.axhline(float(cfg["evaluation"]["confidence_level"]), color="black", linestyle="--", linewidth=1)
        ax.set_ylim(0.0, 1.03)
        ax.set_title(rf"$\alpha={alpha:.2f}$")
        ax.set_ylabel(r"Empirical coverage at true $\Gamma_0$")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    _save(fig, figures, "part_b_coverage")

    fixed_audits = _read(tables / "part_b_fixed_audits.csv")
    fixed_summary = fixed_audits.groupby(["policy", "round"]).agg(
        certificate_width_median=("certificate_width", "median"),
        certificate_width_q25=("certificate_width", lambda x: x.quantile(0.25)),
        certificate_width_q75=("certificate_width", lambda x: x.quantile(0.75)),
        fixed_threshold_width_median=("fixed_threshold_width", "median"),
        fixed_threshold_width_q25=("fixed_threshold_width", lambda x: x.quantile(0.25)),
        fixed_threshold_width_q75=("fixed_threshold_width", lambda x: x.quantile(0.75)),
        audited_fraction_hidden_median=("audited_fraction_hidden", "median"),
    ).reset_index()
    fixed_summary.to_csv(tables / "part_b_fixed_audit_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    sns.lineplot(
        data=fixed_audits,
        x="audited_fraction_hidden",
        y="fixed_threshold_width",
        hue="policy",
        estimator="median",
        ax=ax,
    )
    ax.set_xlabel("Audited fraction of hidden pool")
    ax.set_ylabel("Fixed-threshold ambiguity width")
    ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, figures, "part_b_fixed_audits")

    adaptive = _read(tables / "part_b_adaptive_audits.csv")
    summary = adaptive.groupby(["policy", "round"]).agg(
        certificate_width_median=("certificate_width", "median"),
        certificate_width_q25=("certificate_width", lambda x: x.quantile(0.25)),
        certificate_width_q75=("certificate_width", lambda x: x.quantile(0.75)),
        true_test_cvar_median=("true_test_cvar", "median"),
        true_test_cvar_q25=("true_test_cvar", lambda x: x.quantile(0.25)),
        true_test_cvar_q75=("true_test_cvar", lambda x: x.quantile(0.75)),
        audited_fraction_hidden_median=("audited_fraction_hidden", "median"),
    ).reset_index()
    summary.to_csv(tables / "part_b_adaptive_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))
    sns.lineplot(data=adaptive, x="audited_fraction_hidden", y="certificate_width", hue="policy", estimator="median", ax=axes[0])
    sns.lineplot(data=adaptive, x="audited_fraction_hidden", y="true_test_cvar", hue="policy", estimator="median", ax=axes[1], legend=False)
    axes[0].set_xlabel("Audited fraction of hidden pool")
    axes[0].set_ylabel("Certificate width")
    axes[1].set_xlabel("Audited fraction of hidden pool")
    axes[1].set_ylabel("True held-out CVaR")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    _save(fig, figures, "part_b_adaptive_retraining")

    diagnostics = _read(tables / "part_a_overlap_diagnostics.csv").iloc[0]
    _read(tables / "part_a_rankings.csv")
    _read(tables / "part_a_breakdown.csv")
    dominance = _read(tables / "part_b_dominance.csv")
    dominance_true = dominance[np.isclose(dominance["gamma"], gamma0)]
    certified = dominance_true[dominance_true["certified"].astype(str).str.lower().eq("true")]
    correct_rate = float(pd.to_numeric(certified["correct"], errors="coerce").mean()) if len(certified) else np.nan
    lines = [
        "# Real-data evaluation results",
        "",
        "This file is generated from the run artifacts; it contains no hand-entered numerical claims.",
        "",
        "## Part A: actual selective labels",
        "",
        f"The prepared cohort contains {int(diagnostics['n']):,} applications: "
        f"{int(diagnostics['n_selected']):,} accepted mature loans and {int(diagnostics['n_unselected']):,} rejected applications.",
        f"The cross-fitted propensity AUC is {diagnostics['propensity_auc']:.3f}. "
        f"Fractions clipped below/above the predeclared overlap limits are "
        f"{diagnostics['fraction_clipped_low']:.3%}/{diagnostics['fraction_clipped_high']:.3%}.",
        "",
        "Pairwise breakdown values are in `tables/part_a_breakdown.csv`; accepted-only versus certificate rank changes are in `tables/part_a_rankings.csv`.",
        "",
        "## Part B: controlled masking",
        "",
        f"At the known masking sensitivity Gamma0={gamma0:g}, the empirical correctness rate among non-abstaining certified dominance claims is "
        + ("not estimable (no claims were certified)." if np.isnan(correct_rate) else f"{correct_rate:.3%}."),
        "Coverage and sharpness summaries are in `tables/part_b_coverage_summary.csv`.",
        "Fixed-model TIV/cost trajectories are in `tables/part_b_fixed_audit_summary.csv`. Adaptive retraining trajectories are in `tables/part_b_adaptive_summary.csv`, with both cost-to-width comparisons in `tables/part_b_cost_to_width.csv`.",
        "",
        "## Interpretation guardrail",
        "",
        "The cross-fold t enlargement is evaluated empirically; it is not presented as an exact simultaneous confidence theorem. "
        "The iterative retraining study is an adaptive empirical extension, not a claim of globally optimal active learning.",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
