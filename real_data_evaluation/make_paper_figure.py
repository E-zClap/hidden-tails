"""Three-panel LendingClub figure for the camera-ready, drawn from the run artifacts (artifacts/tables)."""
import argparse
import csv
import pathlib
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

ALPHA = 0.95
MODELS = {"gradient_boosting": ("gradient boosting", "#b2182b"),
          "logistic": ("logistic", "#2166ac"),
          "conservative_logistic": ("default-weighted logistic", "#4d9221")}
POLICIES = {"tiv_per_cost": ("TIV / cost", "#b2182b", "-"), "random": ("random", "0.35", "--"),
            "low_propensity": ("low propensity", "#2166ac", "-."), "uncertainty": ("uncertainty", "#999999", ":")}


def read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def main(artifacts, out_stem):
    tables = pathlib.Path(artifacts) / "tables"
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5})
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.95))  # printed size: full 5.5-inch text width

    # (a) actual selective labels: sharp interval per model over Gamma, accepted-only CVaR as a marker
    ax = axes[0]
    rows = [r for r in read(tables / "part_a_certificates.csv") if r["subset_type"] == "all" and float(r["alpha"]) == ALPHA]
    for k, (model, (label, color)) in enumerate(MODELS.items()):
        cur = sorted((float(r["gamma"]), float(r["lower"]), float(r["upper"]), float(r["accepted_only_cvar"]))
                     for r in rows if r["model"] == model and float(r["gamma"]) in (1.0, 1.5, 2.0, 3.0, 5.0))
        g = np.array([c[0] for c in cur]) * (1 + 0.05 * (k - 1))
        ax.vlines(g, [c[1] for c in cur], [c[2] for c in cur], color=color, lw=2.0, label=label)
        ax.scatter(np.r_[g, g], [c[1] for c in cur] + [c[2] for c in cur], marker="_", s=30, color=color, lw=1.4)
        ax.axhline(cur[0][3], color=color, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([1, 1.5, 2, 3, 5], ["1", "1.5", "2", "3", "5"])
    ax.set_ylim(0.63, 0.97)
    ax.set_xlabel(r"sensitivity $\Gamma$")
    ax.set_ylabel(r"CVaR$_{0.95}$ certificate")
    ax.set_title("(a) applicants (actual labels)", fontsize=8.5)
    ax.legend(frameon=False, fontsize=5.8, loc="upper left")

    # (b) controlled masking: coverage of the true CVaR over Gamma, raw and enlarged
    ax = axes[1]
    cov = defaultdict(list)
    for r in read(tables / "part_b_masking_certificates.csv"):
        if float(r["alpha"]) == ALPHA:
            cov[float(r["gamma"])].append((r["covered"] == "True", r["enlarged_covered"] == "True"))
    gs = sorted(cov)
    ax.plot(gs, [np.mean([c[0] for c in cov[g]]) for g in gs], "o-", color="#2166ac", ms=3.5, label="plug-in")
    ax.plot(gs, [np.mean([c[1] for c in cov[g]]) for g in gs], "s-", color="#b2182b", ms=3.5, label="enlarged")
    ax.axvline(3.0, color="0.5", lw=0.8, ls="--")
    ax.text(1.0, 0.86, r"true $\Gamma_0=3$" + "\n(dashed)", fontsize=6, color="0.35")
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([1, 2, 3, 5, 12], ["1", "2", "3", "5", "12"])
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel(r"assumed $\Gamma$")
    ax.set_ylabel("coverage of true CVaR")
    ax.set_title("(b) controlled masking", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6, loc="lower right", handlelength=1.4)

    # (c) audits with retraining: median certificate width per round
    ax = axes[2]
    summary = read(tables / "part_b_adaptive_summary.csv")
    for policy, (label, color, ls) in POLICIES.items():
        cur = sorted((int(r["round"]), float(r["certificate_width_median"])) for r in summary if r["policy"] == policy)
        ax.plot([2 * c[0] for c in cur], [c[1] for c in cur], ls, color=color, lw=1.5, label=label)
    ax.set_xlabel("hidden labels revealed (%)")
    ax.set_ylabel("median certificate width")
    ax.set_title("(c) audit and retrain", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6, loc="lower left")

    fig.tight_layout(w_pad=1.4)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_stem}.{ext}", dpi=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__ + " Historical adaptive panel; see docs/experiments.md for the current sequential analysis.")
    parser.add_argument("artifacts", help="Directory containing historical artifacts/tables")
    parser.add_argument("out_stem", help="Output path without .pdf/.png extension")
    args = parser.parse_args()
    pathlib.Path(args.out_stem).parent.mkdir(parents=True, exist_ok=True)
    main(args.artifacts, args.out_stem)
