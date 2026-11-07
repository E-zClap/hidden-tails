"""Synthetic experiments and figures for tail-risk certification under selective labels.

Use ``--main-figure`` to regenerate only the conceptual certification figure and
the certification-frontier stress test used as the main figures in the paper.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "DejaVu Sans",
})


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
ALPHA = 0.85
BETA = 1.0 - ALPHA
GAMMA = 50.0
SEED = 20260503
FRONTIER_WORLDS = 120
FRONTIER_CELLS = 120
FRONTIER_GAMMA0 = 10.0
FRONTIER_GAMMA_EVAL = 12.0
FRONTIER_ETA = 0.5
SEMISYNTH_CELLS = 14
SEMISYNTH_RANDOM_POLICIES = 100
SEMISYNTH_DIABETES_QUANTILE = 0.75

def cvar_variational(losses: np.ndarray, alpha: float, weights: np.ndarray | None = None) -> float:
    """Exact finite-support CVaR via the Rockafellar--Uryasev objective."""
    losses = np.asarray(losses, dtype=float)
    if weights is None:
        weights = np.ones_like(losses, dtype=float) / max(len(losses), 1)
    else:
        weights = np.asarray(weights, dtype=float)
        total = weights.sum()
        if total <= 0:
            return np.nan
        weights = weights / total

    beta = 1.0 - alpha
    etas = np.unique(np.concatenate(([0.0, 1.0], losses)))
    values = [eta + np.sum(weights * np.maximum(losses - eta, 0.0)) / beta for eta in etas]
    return float(np.min(values))


def sensitivity_bounds(p1: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    p1 = np.clip(np.asarray(p1, dtype=float), 1e-8, 1.0 - 1e-8)
    lo = p1 / (p1 + gamma * (1.0 - p1))
    hi = gamma * p1 / (1.0 - p1 + gamma * p1)
    return lo, hi


def save_figure(fig: plt.Figure, stem: str, *, tight: bool = False) -> None:
    """Save each figure as PNG for reports and PDF for LaTeX inclusion."""
    save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.015} if tight else {}
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=220, **save_kwargs)
    fig.savefig(FIG_DIR / f"{stem}.pdf", **save_kwargs)


def make_conceptual_certification_figure() -> None:
    """Conceptual map of hidden-tail certification used in the introduction."""
    fig = plt.figure(figsize=(12.6, 3.4))
    gs = fig.add_gridspec(1, 3, left=0.02, right=0.99, top=0.97, bottom=0.08, wspace=0.16)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor("none")

    rng = np.random.default_rng(SEED)
    blue = "#377eb8"
    red = "#d95f5f"
    pale_red = "#f2b8b5"
    gray = "#6b7280"
    dark = "#1f2937"
    gold = "#d99a23"
    green = "#2a9d72"

    # Figure-level arrows turn the panels into one pipeline.
    for left_ax, right_ax in [(axes[0], axes[1]), (axes[1], axes[2])]:
        left_pos = left_ax.get_position()
        right_pos = right_ax.get_position()
        y_mid = 0.5 * (left_pos.y0 + left_pos.y1)
        axes[0].annotate(
            "",
            xy=(right_pos.x0 - 0.012, y_mid),
            xytext=(left_pos.x1 + 0.012, y_mid),
            xycoords=fig.transFigure,
            textcoords=fig.transFigure,
            arrowprops=dict(arrowstyle="->", lw=1.2, color=gray),
        )

    for ax, label in zip(axes, ["(a)", "(b)", "(c)"]):
        ax.text(0.015, 0.965, label, fontsize=9.2, fontweight="semibold",
                color=dark, ha="left", va="top")

    # Panel (a): selective observation hides tail losses.
    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    central = rng.normal(loc=(0.31, 0.52), scale=(0.105, 0.115), size=(22, 2))
    central = np.clip(central, 0.08, 0.72)
    tail = rng.normal(loc=(0.76, 0.56), scale=(0.075, 0.095), size=(10, 2))
    tail = np.clip(tail, 0.55, 0.94)
    ax.add_patch(plt.Rectangle((0.08, 0.20), 0.45, 0.58, fc=blue, ec="none", alpha=0.06))
    ax.add_patch(plt.Rectangle((0.59, 0.20), 0.33, 0.58, fc=red, ec="none", alpha=0.08))
    ax.scatter(central[:, 0], central[:, 1], s=28, c=blue, alpha=0.86, edgecolor="white", linewidth=0.4)
    ax.scatter(tail[:, 0], tail[:, 1], s=42, c=pale_red, alpha=0.75, edgecolor="white", linewidth=0.4)
    hidden_tail = tail[::2]
    ax.scatter(hidden_tail[:, 0], hidden_tail[:, 1], s=44, c=gray, alpha=0.55, edgecolor="white", linewidth=0.4)
    ax.text(0.76, 0.37, "hidden\nlabels", ha="center", va="center", fontsize=8.4, color=gray, fontweight="semibold")
    gate = plt.Rectangle((0.55, 0.20), 0.018, 0.58, fc=dark, ec="none", alpha=0.62)
    ax.add_patch(gate)
    ax.text(0.305, 0.81, "observed labels", fontsize=8.6, color=blue, fontweight="semibold", ha="center")
    ax.text(0.559, 0.145, "historical\nselection", fontsize=8.0, color=dark, ha="center", va="top")
    ax.text(0.64, 0.81, "hidden tail", fontsize=8.6, color=red, fontweight="semibold")

    # Panel (b): point estimate becomes an interval certificate.
    ax = axes[1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    x0 = 0.56
    ax.annotate("", xy=(x0, 0.84), xytext=(x0, 0.18), arrowprops=dict(arrowstyle="->", lw=1.1, color=dark, alpha=0.8))
    y_lo, y_hi, y_obs, y_true = 0.36, 0.74, 0.25, 0.58
    ax.plot([x0, x0], [y_lo, y_hi], color=gold, lw=7.0, alpha=0.25, solid_capstyle="butt")
    ax.plot([x0, x0], [y_lo, y_hi], color=gold, lw=2.2, solid_capstyle="butt")
    ax.hlines([y_lo, y_hi], x0 - 0.06, x0 + 0.06, color=gold, lw=2.2)
    ax.scatter([x0], [y_obs], s=55, c=blue, zorder=4)
    ax.scatter([x0], [y_true], s=72, c="white", edgecolor=red, linewidth=2.0, zorder=4)
    ax.text(x0 + 0.10, y_obs, "selected-label\nCVaR", fontsize=8.2, color=blue, va="center", ha="left")
    ax.text(x0 + 0.10, y_true, "target CVaR\ninterval", fontsize=8.5, color=red, va="center")
    ax.text(x0 + 0.10, y_hi, "upper certificate", fontsize=8.4, color=dark, va="center")
    ax.text(x0 + 0.10, y_lo, "lower certificate", fontsize=8.4, color=dark, va="center")
    for k, lab in enumerate([r"$q_1$", r"$q_2$", r"$q_3$"]):
        y = 0.72 - 0.14 * k
        ax.text(0.16, y, lab, ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="#eef2f7", ec="#9ca3af"))
        ax.annotate("", xy=(x0 - 0.09, 0.64 - 0.07 * k), xytext=(0.23, y),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="#9ca3af"))
    ax.text(0.08, 0.26, "compatible\nhidden laws", fontsize=8.5, color=gray)

    # Panel (c): revelation ranked by TIV/cost shrinks certification debt.
    ax = axes[2]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([0.14, 0.14], [0.34, 0.78], color=gold, lw=4.0, alpha=0.75)
    ax.text(0.14, 0.28, r"$W_{f,\eta}(0)$", fontsize=8.5, color=gold, ha="center")
    ax.annotate("", xy=(0.30, 0.56), xytext=(0.20, 0.56),
                arrowprops=dict(arrowstyle="->", lw=1.3, color=gray))
    ax.text(0.485, 0.76, r"rank by $\tau_{f,\eta}(x)/c(x)$", fontsize=8.6, color=dark, ha="center")
    card_y = 0.56
    for i, x in enumerate([0.35, 0.44, 0.53, 0.62], start=1):
        color = green if i <= 2 else "#e5e7eb"
        edge = green if i <= 2 else "#9ca3af"
        ax.text(x, card_y, rf"$x_{{({i})}}$", ha="center", va="center", fontsize=9.5, color=dark,
                bbox=dict(boxstyle="round,pad=0.32", fc=color, ec=edge, alpha=0.35 if i <= 2 else 0.25))
    ax.text(0.485, 0.42, "reveal top regions", fontsize=8.2, color=dark, ha="center")
    ax.annotate("", xy=(0.78, 0.56), xytext=(0.68, 0.56),
                arrowprops=dict(arrowstyle="->", lw=1.3, color=gray))
    ax.plot([0.84, 0.84], [0.47, 0.66], color=green, lw=4.0, alpha=0.85)
    ax.text(0.84, 0.71, r"$W_{f,\eta}(\lambda)\leq\varepsilon$", fontsize=8.5, color=green, ha="center")
    save_figure(fig, "conceptual_tail_risk_certification", tight=True)
    plt.close(fig)


@dataclass(frozen=True)
class SyntheticSystem:
    names: tuple[str, ...]
    weights: np.ndarray
    p_true: np.ndarray
    s_y1: np.ndarray
    s_y0: np.ndarray

    @property
    def n_regions(self) -> int:
        return len(self.names)

    @property
    def pi(self) -> np.ndarray:
        return self.p_true * self.s_y1 + (1.0 - self.p_true) * self.s_y0

    @property
    def p1(self) -> np.ndarray:
        return self.p_true * self.s_y1 / self.pi

    @property
    def p0(self) -> np.ndarray:
        return self.p_true * (1.0 - self.s_y1) / (1.0 - self.pi)

    def sample(self, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        x = rng.choice(self.n_regions, size=n, p=self.weights)
        y = rng.binomial(1, self.p_true[x])
        s_prob = np.where(y == 1, self.s_y1[x], self.s_y0[x])
        s = rng.binomial(1, s_prob)
        yobs = s * y
        return {"x": x, "y": y, "s": s, "yobs": yobs}


def main_system() -> SyntheticSystem:
    """Five-region system used for experiments 2--3.

    Region B is the hidden-tail region: adverse labels are much less likely to be
    revealed. The implied odds-ratio shift is about 48.5, so the experiment uses
    Gamma=50. Region E is deliberately prediction-uncertain but has zero TIV for
    the candidate class because all candidates predict 0.5 there.
    """

    return SyntheticSystem(
        names=("A common-safe", "B hidden-tail", "C easy", "D moderate", "E uncertain"),
        weights=np.array([0.45, 0.15, 0.15, 0.10, 0.15]),
        p_true=np.array([0.04, 0.75, 0.10, 0.40, 0.50]),
        s_y1=np.array([0.80, 0.03, 0.80, 0.80, 0.80]),
        s_y0=np.array([0.80, 0.60, 0.80, 0.80, 0.80]),
    )


def candidate_predictions() -> dict[str, np.ndarray]:
    base = np.array([0.05, 0.05, 0.10, 0.40, 0.50])
    return {
        "low-B observed learner": base.copy(),
        "mild-B robust learner": np.array([0.05, 0.25, 0.10, 0.40, 0.50]),
        "high-B conservative learner": np.array([0.05, 0.65, 0.10, 0.40, 0.50]),
        "very-high-B learner": np.array([0.05, 0.85, 0.10, 0.40, 0.50]),
    }


def losses_for_prediction(pred: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.abs(pred[x] - y)


def exact_true_cvar(system: SyntheticSystem, pred: np.ndarray, alpha: float = ALPHA) -> float:
    losses = []
    weights = []
    for j in range(system.n_regions):
        losses.extend([abs(pred[j] - 0.0), abs(pred[j] - 1.0)])
        weights.extend([system.weights[j] * (1.0 - system.p_true[j]), system.weights[j] * system.p_true[j]])
    return cvar_variational(np.array(losses), alpha, np.array(weights))


def exact_observed_cvar(system: SyntheticSystem, pred: np.ndarray, alpha: float = ALPHA) -> float:
    losses = []
    weights = []
    norm = float(np.sum(system.weights * system.pi))
    for j in range(system.n_regions):
        wj = system.weights[j] * system.pi[j] / norm
        p1 = system.p1[j]
        losses.extend([abs(pred[j] - 0.0), abs(pred[j] - 1.0)])
        weights.extend([wj * (1.0 - p1), wj * p1])
    return cvar_variational(np.array(losses), alpha, np.array(weights))


def exact_envelope(
    system: SyntheticSystem,
    pred: np.ndarray,
    alpha: float = ALPHA,
    gamma: float = GAMMA,
    upper: bool = True,
) -> tuple[float, float]:
    p_lo, p_hi = sensitivity_bounds(system.p1, gamma)
    candidate_losses = np.unique(np.concatenate(([0.0, 1.0], pred, 1.0 - pred)))
    best_value = np.inf
    best_eta = 0.0
    for eta in candidate_losses:
        total = 0.0
        for j in range(system.n_regions):
            a0 = max(abs(pred[j] - 0.0) - eta, 0.0)
            a1 = max(abs(pred[j] - 1.0) - eta, 0.0)
            selected = system.p1[j] * a1 + (1.0 - system.p1[j]) * a0
            if upper:
                q = p_hi[j] if a1 >= a0 else p_lo[j]
            else:
                q = p_lo[j] if a1 >= a0 else p_hi[j]
            unselected = q * a1 + (1.0 - q) * a0
            total += system.weights[j] * (system.pi[j] * selected + (1.0 - system.pi[j]) * unselected)
        value = eta + total / (1.0 - alpha)
        if value < best_value:
            best_value = value
            best_eta = eta
    return float(best_value), float(best_eta)


def empirical_conservative_objective(
    sample: dict[str, np.ndarray],
    system: SyntheticSystem,
    pred: np.ndarray,
    alpha: float = ALPHA,
    gamma: float = GAMMA,
) -> tuple[float, float]:
    x, y, s = sample["x"], sample["y"], sample["s"]
    p_lo, p_hi = sensitivity_bounds(system.p1, gamma)
    candidate_losses = np.unique(np.concatenate(([0.0, 1.0], pred, 1.0 - pred)))
    best_value = np.inf
    best_eta = 0.0
    for eta in candidate_losses:
        observed_loss = np.maximum(np.abs(pred[x] - y) - eta, 0.0)
        a0 = np.maximum(np.abs(pred[x] - 0.0) - eta, 0.0)
        a1 = np.maximum(np.abs(pred[x] - 1.0) - eta, 0.0)
        q = np.where(a1 >= a0, p_hi[x], p_lo[x])
        hidden = q * a1 + (1.0 - q) * a0
        h = np.where(s == 1, observed_loss, hidden)
        value = eta + np.mean(h) / (1.0 - alpha)
        if value < best_value:
            best_value = value
            best_eta = eta
    return float(best_value), float(best_eta)


def choose_learners(sample: dict[str, np.ndarray], system: SyntheticSystem, preds: dict[str, np.ndarray]) -> dict[str, str]:
    x, y, s = sample["x"], sample["y"], sample["s"]
    chosen: dict[str, str] = {}

    full_scores = {
        name: cvar_variational(losses_for_prediction(pred, x, y), ALPHA)
        for name, pred in preds.items()
    }
    chosen["Full-label oracle CVaR"] = min(full_scores, key=full_scores.get)

    selected = s == 1
    obs_scores = {
        name: cvar_variational(losses_for_prediction(pred, x[selected], y[selected]), ALPHA)
        for name, pred in preds.items()
    }
    chosen["Observed-label CVaR"] = min(obs_scores, key=obs_scores.get)

    ipw_scores = {}
    weights = 1.0 / system.pi[x[selected]]
    for name, pred in preds.items():
        ipw_scores[name] = cvar_variational(losses_for_prediction(pred, x[selected], y[selected]), ALPHA, weights)
    chosen["IPW CVaR"] = min(ipw_scores, key=ipw_scores.get)

    env_scores = {
        name: empirical_conservative_objective(sample, system, pred, ALPHA, GAMMA)[0]
        for name, pred in preds.items()
    }
    chosen["Conservative envelope"] = min(env_scores, key=env_scores.get)
    return chosen


def experiment_1() -> pd.DataFrame:
    alpha = ALPHA
    gamma = GAMMA
    w_b = 0.18
    pi_a = 0.95
    pi_b = 0.04
    p1_a = 0.0
    p1_b = 1.0
    p_true_a = 0.0
    p_true_b = 1.0
    weights = np.array([1.0 - w_b, w_b])
    pi = np.array([pi_a, pi_b])
    p1 = np.array([p1_a, p1_b])
    p_true = np.array([p_true_a, p_true_b])
    p_lo, p_hi = sensitivity_bounds(p1, gamma)

    preds = {
        "f hidden-tail": np.array([0.0, 0.0]),
        "g tail-safe": np.array([0.0, 1.0]),
    }

    rows = []
    selected_norm = float(np.sum(weights * pi))
    for name, pred in preds.items():
        true_losses = np.array([abs(pred[0] - 0.0), abs(pred[1] - 1.0)])
        true_weights = weights.copy()
        observed_losses = np.array([abs(pred[0] - 0.0), abs(pred[1] - 1.0)])
        observed_weights = weights * pi / selected_norm

        def envelope(upper: bool) -> float:
            best = np.inf
            for eta in np.unique(np.concatenate(([0.0, 1.0], pred, 1.0 - pred))):
                total = 0.0
                for j in range(2):
                    a0 = max(abs(pred[j] - 0.0) - eta, 0.0)
                    a1 = max(abs(pred[j] - 1.0) - eta, 0.0)
                    selected_term = p1[j] * a1 + (1.0 - p1[j]) * a0
                    if upper:
                        q = p_hi[j] if a1 >= a0 else p_lo[j]
                    else:
                        q = p_lo[j] if a1 >= a0 else p_hi[j]
                    hidden_term = q * a1 + (1.0 - q) * a0
                    total += weights[j] * (pi[j] * selected_term + (1.0 - pi[j]) * hidden_term)
                best = min(best, eta + total / (1.0 - alpha))
            return float(best)

        rows.append(
            {
                "predictor": name,
                "observed_label_cvar": cvar_variational(observed_losses, alpha, observed_weights),
                "true_full_population_cvar": cvar_variational(true_losses, alpha, true_weights),
                "sharp_upper_envelope": envelope(True),
                "sharp_lower_envelope": envelope(False),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(RES_DIR / "experiment1_rank_reversal.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    metrics = ["observed_label_cvar", "true_full_population_cvar", "sharp_upper_envelope"]
    labels = ["Observed-label CVaR", "True full-pop. CVaR", "Sharp upper envelope"]
    xloc = np.arange(len(metrics))
    width = 0.34
    for offset, predictor in [(-width / 2, "f hidden-tail"), (width / 2, "g tail-safe")]:
        vals = [df.loc[df["predictor"] == predictor, m].iloc[0] for m in metrics]
        ax.bar(xloc + offset, vals, width=width, label=predictor)
    ax.set_xticks(xloc)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(f"CVaR$_{{{alpha:.2f}}}$ / envelope")
    ax.set_title("Hidden-tail rank reversal")
    ax.legend(frameon=True)
    fig.tight_layout()
    save_figure(fig, "experiment1_hidden_tail_rank_reversal")
    plt.close(fig)

    pi_grid = np.linspace(0.01, 0.70, 80)
    observed_f = []
    true_f = []
    upper_f = []
    for pib in pi_grid:
        selected_norm = weights[0] * pi_a + weights[1] * pib
        observed_weights = np.array([weights[0] * pi_a, weights[1] * pib]) / selected_norm
        observed_f.append(cvar_variational(np.array([0.0, 1.0]), alpha, observed_weights))
        true_f.append(cvar_variational(np.array([0.0, 1.0]), alpha, weights))
        hidden_error_prob = weights[1] * (pib * p1_b + (1 - pib) * 1.0)
        upper_f.append(min(1.0, hidden_error_prob / BETA))

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(pi_grid, observed_f, label="Observed-label CVaR of f", linewidth=2.2)
    ax.plot(pi_grid, true_f, label="True CVaR of f", linewidth=2.2)
    ax.plot(pi_grid, upper_f, "--", label="Upper envelope of f", linewidth=2.2)
    ax.set_xlabel("Selection probability in hidden region B")
    ax.set_ylabel(f"CVaR$_{{{alpha:.2f}}}$")
    ax.set_title("Observed tail risk remains low when the dangerous region is rarely labeled")
    ax.legend(frameon=True)
    fig.tight_layout()
    save_figure(fig, "experiment1_selection_sweep")
    plt.close(fig)
    return df


def experiment_2(rng: np.random.Generator, n_rep: int = 250, n_train: int = 1400) -> pd.DataFrame:
    system = main_system()
    preds = candidate_predictions()
    records = []
    for rep in range(n_rep):
        sample = system.sample(n_train, rng)
        choices = choose_learners(sample, system, preds)
        for learner, pred_name in choices.items():
            pred = preds[pred_name]
            upper, eta_u = exact_envelope(system, pred, ALPHA, GAMMA, upper=True)
            lower, _ = exact_envelope(system, pred, ALPHA, GAMMA, upper=False)
            records.append(
                {
                    "rep": rep,
                    "learner": learner,
                    "chosen_predictor": pred_name,
                    "true_cvar": exact_true_cvar(system, pred, ALPHA),
                    "observed_cvar": exact_observed_cvar(system, pred, ALPHA),
                    "upper_envelope": upper,
                    "lower_envelope": lower,
                    "identification_width": upper - lower,
                    "eta_upper": eta_u,
                }
            )

    df = pd.DataFrame(records)
    df.to_csv(RES_DIR / "experiment2_learning_raw.csv", index=False)
    summary = (
        df.groupby("learner")
        .agg(
            true_cvar_mean=("true_cvar", "mean"),
            true_cvar_se=("true_cvar", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
            observed_cvar_mean=("observed_cvar", "mean"),
            upper_envelope_mean=("upper_envelope", "mean"),
            width_mean=("identification_width", "mean"),
            most_common_predictor=("chosen_predictor", lambda x: x.value_counts().index[0]),
        )
        .reset_index()
    )
    summary.to_csv(RES_DIR / "experiment2_learning_summary.csv", index=False)

    order = ["Observed-label CVaR", "IPW CVaR", "Conservative envelope", "Full-label oracle CVaR"]
    plot_df = summary.set_index("learner").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    xloc = np.arange(len(order))
    width = 0.25
    for k, (metric, label) in enumerate(
        [
            ("observed_cvar_mean", "Observed-label CVaR"),
            ("true_cvar_mean", "True full-pop. CVaR"),
            ("upper_envelope_mean", "Sharp upper envelope"),
        ]
    ):
        ax.bar(xloc + (k - 1) * width, plot_df[metric], width=width, label=label)
    ax.set_xticks(xloc)
    ax.set_xticklabels(order, rotation=12, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(f"CVaR$_{{{ALPHA:.2f}}}$ / envelope")
    ax.set_title("Conservative learning under selective labels")
    ax.legend(frameon=True, ncol=1)
    fig.tight_layout()
    save_figure(fig, "experiment2_conservative_vs_naive")
    plt.close(fig)

    choice_table = (
        df.groupby(["learner", "chosen_predictor"]).size().reset_index(name="count")
        .assign(frequency=lambda z: z["count"] / n_rep)
    )
    choice_table.to_csv(RES_DIR / "experiment2_choice_frequencies.csv", index=False)
    return summary


def empirical_audited_envelope(
    system: SyntheticSystem,
    pred: np.ndarray,
    sample: dict[str, np.ndarray],
    audited: np.ndarray,
    alpha: float = ALPHA,
    gamma: float = GAMMA,
    upper: bool = True,
) -> tuple[float, float]:
    x, y, s = sample["x"], sample["y"], sample["s"]
    known = (s == 1) | audited
    p_lo, p_hi = sensitivity_bounds(system.p1, gamma)
    candidate_losses = np.unique(np.concatenate(([0.0, 1.0], pred, 1.0 - pred)))
    best_value = np.inf
    best_eta = 0.0
    for eta in candidate_losses:
        known_excess = np.maximum(np.abs(pred[x] - y) - eta, 0.0)
        a0 = np.maximum(np.abs(pred[x] - 0.0) - eta, 0.0)
        a1 = np.maximum(np.abs(pred[x] - 1.0) - eta, 0.0)
        if upper:
            q = np.where(a1 >= a0, p_hi[x], p_lo[x])
        else:
            q = np.where(a1 >= a0, p_lo[x], p_hi[x])
        hidden = q * a1 + (1.0 - q) * a0
        h = np.where(known, known_excess, hidden)
        value = eta + np.mean(h) / (1.0 - alpha)
        if value < best_value:
            best_value = value
            best_eta = eta
    return float(best_value), float(best_eta)


def experiment_3(rng: np.random.Generator, n_pool: int = 7000) -> pd.DataFrame:
    system = main_system()
    preds = candidate_predictions()
    # Use the population observed-label learner because this is the debt-heavy
    # predictor from Experiment 2. TIV is most informative when there is residual
    # hidden-tail ambiguity to repay; the conservative learner already suppresses
    # much of this debt.
    observed_scores = {name: exact_observed_cvar(system, pred, ALPHA) for name, pred in preds.items()}
    pred_name = min(observed_scores, key=observed_scores.get)
    pred = preds[pred_name]
    _, eta = exact_envelope(system, pred, ALPHA, GAMMA, upper=False)

    sample = system.sample(n_pool, rng)
    x = sample["x"]
    hidden_idx = np.flatnonzero(sample["s"] == 0)
    p_lo, p_hi = sensitivity_bounds(system.p1, GAMMA)
    a0 = np.maximum(np.abs(pred[x] - 0.0) - eta, 0.0)
    a1 = np.maximum(np.abs(pred[x] - 1.0) - eta, 0.0)
    tiv = (1.0 - system.pi[x]) * (p_hi[x] - p_lo[x]) * np.abs(a1 - a0) / BETA
    # In a pool of already-hidden units, the marginal reduction per audited
    # label is the local width itself; the (1-pi) factor in population TIV
    # determines how much mass from each region enters such a hidden pool.
    hidden_pool_tiv = (p_hi[x] - p_lo[x]) * np.abs(a1 - a0) / BETA
    uncertainty = -np.abs(pred[x] - 0.5)
    central_p = 0.5 * (p_lo[x] + p_hi[x])
    central_loss = central_p * np.abs(pred[x] - 1.0) + (1.0 - central_p) * np.abs(pred[x] - 0.0)
    density = system.weights[x]

    policies = {
        "Top TIV/cost": hidden_pool_tiv,
        "Random": rng.random(n_pool),
        "Uncertainty": uncertainty,
        "High predicted loss": central_loss,
        "High density": density,
    }
    budgets = np.unique(np.round(np.linspace(0, min(1800, len(hidden_idx)), 16)).astype(int))
    records = []
    base_upper, _ = empirical_audited_envelope(system, pred, sample, np.zeros(n_pool, dtype=bool), ALPHA, GAMMA, True)
    base_lower, _ = empirical_audited_envelope(system, pred, sample, np.zeros(n_pool, dtype=bool), ALPHA, GAMMA, False)
    base_width = base_upper - base_lower
    fixed_local_width = (p_hi[x] - p_lo[x]) * np.abs(a1 - a0) / BETA

    for policy, score in policies.items():
        order = hidden_idx[np.argsort(score[hidden_idx])[::-1]]
        for budget in budgets:
            audited = np.zeros(n_pool, dtype=bool)
            if budget > 0:
                audited[order[:budget]] = True
            upper, _ = empirical_audited_envelope(system, pred, sample, audited, ALPHA, GAMMA, True)
            lower, _ = empirical_audited_envelope(system, pred, sample, audited, ALPHA, GAMMA, False)
            width = upper - lower
            fixed_width = float(np.mean(np.where((sample["s"] == 0) & (~audited), fixed_local_width, 0.0)))
            records.append(
                {
                    "policy": policy,
                    "budget": int(budget),
                    "upper": upper,
                    "lower": lower,
                    "width": width,
                    "width_reduction": base_width - width,
                    "relative_width_reduction": (base_width - width) / base_width if base_width > 0 else 0.0,
                    "fixed_threshold_width": fixed_width,
                    "fixed_threshold_width_reduction": float(np.mean(np.where(sample["s"] == 0, fixed_local_width, 0.0))) - fixed_width,
                    "relative_fixed_threshold_width_reduction": (
                        (float(np.mean(np.where(sample["s"] == 0, fixed_local_width, 0.0))) - fixed_width)
                        / float(np.mean(np.where(sample["s"] == 0, fixed_local_width, 0.0)))
                        if float(np.mean(np.where(sample["s"] == 0, fixed_local_width, 0.0))) > 0
                        else 0.0
                    ),
                    "chosen_predictor": pred_name,
                    "eta": eta,
                    "base_width": base_width,
                }
            )

    df = pd.DataFrame(records)
    df.to_csv(RES_DIR / "experiment3_tiv_audit.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for policy in ["Top TIV/cost", "High predicted loss", "Uncertainty", "High density", "Random"]:
        cur = df[df["policy"] == policy]
        ax.plot(cur["budget"], cur["relative_fixed_threshold_width_reduction"], marker="o", linewidth=2.0, label=policy)
    ax.set_xlabel("Additional hidden labels revealed")
    ax.set_ylabel("Fraction of fixed-threshold tail ambiguity removed")
    ax.set_title("TIV-per-cost fixed-threshold ambiguity reduction")
    ax.set_ylim(bottom=-0.02)
    ax.legend(frameon=True)
    fig.tight_layout()
    save_figure(fig, "experiment3_tiv_guided_revelation")
    plt.close(fig)

    region_scores = pd.DataFrame(
        {
            "region": system.names,
            "weight": system.weights,
            "pi": system.pi,
            "p1": system.p1,
            "p_lo": p_lo,
            "p_hi": p_hi,
            "prediction": pred,
            "tiv_at_eta": [
                (1.0 - system.pi[j])
                * (p_hi[j] - p_lo[j])
                * abs(max(abs(pred[j] - 1.0) - eta, 0.0) - max(abs(pred[j] - 0.0) - eta, 0.0))
                / BETA
                for j in range(system.n_regions)
            ],
        }
    )
    region_scores.to_csv(RES_DIR / "experiment3_region_tiv_scores.csv", index=False)
    return df


def simulate_lower_bound_setting(
    rng: np.random.Generator,
    n: int,
    beta: float,
    pi0: float,
    d: int = 16,
    reps: int = 240,
) -> float:
    effective = n * beta * pi0
    gamma = min(0.20, 0.16 * np.sqrt(d / max(effective, 1e-9)))
    excesses = []
    for _ in range(reps):
        theta = rng.choice([-1, 1], size=d)
        # Draw full observations. Only T=1 and S=1 labels are informative.
        t = rng.binomial(1, beta, size=n)
        j = rng.integers(0, d, size=n)
        p = 0.5 + gamma * theta[j]
        y = rng.binomial(1, p)
        s = rng.binomial(1, pi0, size=n)
        informative = (t == 1) & (s == 1)
        scores = np.zeros(d)
        for jj in range(d):
            mask = informative & (j == jj)
            # Labels in {0,1}; subtract 0.5 so majority estimates theta_j.
            scores[jj] = np.sum(y[mask] - 0.5)
        theta_hat = np.where(scores >= 0.0, 1, -1)
        mistakes = np.mean(theta_hat != theta)
        excesses.append(2.0 * gamma * mistakes)
    return float(np.mean(excesses))


def experiment_4(rng: np.random.Generator) -> pd.DataFrame:
    settings = []
    for n in [600, 900, 1400, 2200, 3500, 5500]:
        for beta in [0.10, 0.15, 0.25]:
            for pi0 in [0.20, 0.35, 0.55]:
                settings.append((n, beta, pi0))
    rows = []
    for n, beta, pi0 in settings:
        excess = simulate_lower_bound_setting(rng, n, beta, pi0)
        rows.append(
            {
                "n": n,
                "beta": beta,
                "alpha": 1.0 - beta,
                "pi0": pi0,
                "effective_revealed_tail_labels": n * beta * pi0,
                "x_rate": 1.0 / np.sqrt(n * beta * pi0),
                "excess_cvar": excess,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(RES_DIR / "experiment4_lower_bound_scaling.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    colors = {0.10: "#4C78A8", 0.15: "#F58518", 0.25: "#54A24B"}
    for beta, cur in df.groupby("beta"):
        ax.scatter(cur["x_rate"], cur["excess_cvar"], label=fr"$\beta={beta:.2f}$", s=42, alpha=0.8, color=colors[beta])
    coef = np.polyfit(df["x_rate"], df["excess_cvar"], deg=1)
    xs = np.linspace(df["x_rate"].min(), df["x_rate"].max(), 100)
    ax.plot(xs, coef[0] * xs + coef[1], color="black", linewidth=2.0, label="linear fit")
    ax.set_xlabel(r"$1/\sqrt{n\pi_0(1-\alpha)}$")
    ax.set_ylabel("Excess CVaR of majority learner")
    ax.set_title("Difficulty tracks revealed-tail-label scarcity")
    ax.legend(frameon=True)
    fig.tight_layout()
    save_figure(fig, "experiment4_revealed_tail_scaling")
    plt.close(fig)
    return df


def make_main_three_panel_figure() -> None:
    """Compact paper-ready figure for the main synthetic section."""
    exp1 = pd.read_csv(RES_DIR / "experiment1_rank_reversal.csv")
    exp2 = pd.read_csv(RES_DIR / "experiment2_learning_summary.csv")
    exp3 = pd.read_csv(RES_DIR / "experiment3_tiv_audit.csv")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.65))
    colors = {
        "blue": "#4C78A8",
        "orange": "#F58518",
        "green": "#54A24B",
        "red": "#E45756",
        "purple": "#9467BD",
    }

    ax = axes[0]
    metrics = ["observed_label_cvar", "true_full_population_cvar", "sharp_upper_envelope"]
    labels = ["Obs.", "True", "Upper"]
    xloc = np.arange(len(metrics))
    width = 0.34
    for offset, predictor, color in [
        (-width / 2, "f hidden-tail", colors["blue"]),
        (width / 2, "g tail-safe", colors["orange"]),
    ]:
        vals = [exp1.loc[exp1["predictor"] == predictor, m].iloc[0] for m in metrics]
        ax.bar(xloc + offset, vals, width=width, label=predictor.replace(" ", "\n"), color=color)
    ax.set_xticks(xloc)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("CVaR / envelope", fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(frameon=True, fontsize=5.6, loc="upper left")
    ax.text(-0.17, 1.06, "(a)", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")

    ax = axes[1]
    order = ["Observed-label CVaR", "IPW CVaR", "Conservative envelope", "Full-label oracle CVaR"]
    short = ["Obs.", "IPW", "Cons.", "Oracle"]
    plot_df = exp2.set_index("learner").loc[order].reset_index()
    xloc = np.arange(len(order))
    width = 0.24
    for k, (metric, label, color) in enumerate(
        [
            ("observed_cvar_mean", "Observed", colors["blue"]),
            ("true_cvar_mean", "True", colors["orange"]),
            ("upper_envelope_mean", "Upper", colors["green"]),
        ]
    ):
        ax.bar(xloc + (k - 1) * width, plot_df[metric], width=width, label=label, color=color)
    ax.set_xticks(xloc)
    ax.set_xticklabels(short, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(frameon=True, fontsize=5.6, loc="upper right")
    ax.text(-0.17, 1.06, "(b)", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")

    ax = axes[2]
    policy_order = ["Top TIV/cost", "High density", "Random", "High predicted loss", "Uncertainty"]
    policy_colors = {
        "Top TIV/cost": colors["blue"],
        "High density": colors["red"],
        "Random": colors["purple"],
        "High predicted loss": colors["orange"],
        "Uncertainty": colors["green"],
    }
    for policy in policy_order:
        cur = exp3[exp3["policy"] == policy]
        ax.plot(
            cur["budget"],
            cur["relative_fixed_threshold_width_reduction"],
            linewidth=1.35,
            label=policy,
            color=policy_colors[policy],
        )
    ax.set_xlabel("Hidden labels revealed", fontsize=7)
    ax.set_ylabel("Ambiguity removed", fontsize=7)
    ax.set_ylim(-0.02, 1.05)
    ax.tick_params(axis="both", labelsize=7)
    ax.legend(frameon=True, fontsize=5.2, loc="lower right")
    ax.text(-0.17, 1.06, "(c)", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")

    fig.tight_layout(w_pad=1.0)
    save_figure(fig, "synthetic_three_panel")
    plt.close(fig)


def experiment_5_sensitivity() -> pd.DataFrame:
    """Exact-population sensitivity of the sharp interval width."""
    base_system = main_system()
    pred = candidate_predictions()["mild-B robust learner"]

    records: list[dict[str, float | str]] = []

    def envelope_from_observed(
        weights: np.ndarray,
        pi: np.ndarray,
        p1: np.ndarray,
        alpha: float,
        gamma: float,
        upper: bool,
    ) -> float:
        p_lo, p_hi = sensitivity_bounds(p1, gamma)
        candidate_losses = np.unique(np.concatenate(([0.0, 1.0], pred, 1.0 - pred)))
        best_value = np.inf
        for eta in candidate_losses:
            total = 0.0
            for j in range(len(weights)):
                a0 = max(abs(pred[j] - 0.0) - eta, 0.0)
                a1 = max(abs(pred[j] - 1.0) - eta, 0.0)
                selected = p1[j] * a1 + (1.0 - p1[j]) * a0
                if upper:
                    q = p_hi[j] if a1 >= a0 else p_lo[j]
                else:
                    q = p_lo[j] if a1 >= a0 else p_hi[j]
                hidden = q * a1 + (1.0 - q) * a0
                total += weights[j] * (pi[j] * selected + (1.0 - pi[j]) * hidden)
            best_value = min(best_value, eta + total / (1.0 - alpha))
        return float(best_value)

    for gamma in np.geomspace(1.0, 100.0, 60):
        upper, _ = exact_envelope(base_system, pred, ALPHA, float(gamma), upper=True)
        lower, _ = exact_envelope(base_system, pred, ALPHA, float(gamma), upper=False)
        records.append({"sweep": "Gamma", "x": float(gamma), "width": upper - lower})

    for alpha in np.linspace(0.55, 0.95, 60):
        upper, _ = exact_envelope(base_system, pred, float(alpha), GAMMA, upper=True)
        lower, _ = exact_envelope(base_system, pred, float(alpha), GAMMA, upper=False)
        records.append({"sweep": "alpha", "x": float(alpha), "width": upper - lower})

    for pi0 in np.linspace(0.05, 1.0, 60):
        pi = np.full(base_system.n_regions, pi0)
        upper = envelope_from_observed(base_system.weights, pi, base_system.p1, ALPHA, GAMMA, True)
        lower = envelope_from_observed(base_system.weights, pi, base_system.p1, ALPHA, GAMMA, False)
        records.append({"sweep": "pi", "x": float(pi0), "width": upper - lower})

    df = pd.DataFrame(records)
    df.to_csv(RES_DIR / "experiment5_sensitivity.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.2))
    specs = [
        ("Gamma", r"Sensitivity $\Gamma$", True),
        ("alpha", r"Tail level $\alpha$", False),
        ("pi", r"Common reveal prob. $\pi$", False),
    ]
    for ax, (label, xlabel, logx) in zip(axes, specs):
        cur = df[df["sweep"] == label]
        ax.plot(cur["x"], cur["width"], color="#4C78A8", linewidth=2.0)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="both", labelsize=7)
    axes[0].set_ylabel("Sharp CVaR interval width", fontsize=7)
    for ax, panel in zip(axes, ["(a)", "(b)", "(c)"]):
        ax.text(0.03, 0.96, panel, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    fig.tight_layout(w_pad=1.0)
    save_figure(fig, "experiment5_sensitivity")
    plt.close(fig)
    return df


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-8, 1.0 - 1e-8)
    return np.log(p / (1.0 - p))


@dataclass(frozen=True)
class FrontierWorld:
    mu: np.ndarray
    pi: np.ndarray
    p1: np.ndarray
    p0: np.ndarray
    cost: np.ndarray
    score: np.ndarray

    @property
    def p_full(self) -> np.ndarray:
        return self.pi * self.p1 + (1.0 - self.pi) * self.p0


def generate_frontier_world(rng: np.random.Generator, k: int = 120, gamma0: float = 10.0) -> FrontierWorld:
    """Random finite-cell selective-label world for certification stress tests."""
    mu = rng.dirichlet(np.full(k, 0.75))
    score = rng.normal(size=k)
    hidden_score = score + 0.35 * rng.normal(size=k)
    hidden = hidden_score >= np.quantile(hidden_score, 0.65)

    p1 = sigmoid(-1.35 + 0.75 * score + 0.45 * rng.normal(size=k))
    p1[hidden] = sigmoid(-2.05 + 0.35 * score[hidden] + 0.30 * rng.normal(size=hidden.sum()))
    p1 = np.clip(p1, 0.04, 0.72)

    delta = rng.normal(0.0, 0.18, size=k)
    delta[hidden] = rng.uniform(0.80 * np.log(gamma0), np.log(gamma0), size=hidden.sum())
    p0 = sigmoid(logit(p1) + delta)
    p0 = np.clip(p0, 0.02, 0.98)

    pi = rng.uniform(0.62, 0.95, size=k)
    pi[hidden] = rng.uniform(0.02, 0.16, size=hidden.sum())

    cost = rng.uniform(0.45, 1.75, size=k)
    expensive = rng.random(k) < 0.20
    cost[expensive] += rng.uniform(0.75, 1.75, size=expensive.sum())
    return FrontierWorld(mu=mu, pi=pi, p1=p1, p0=p0, cost=cost, score=score)


def finite_cvar_from_label_law(mu: np.ndarray, p: np.ndarray, pred: np.ndarray, alpha: float = ALPHA) -> float:
    losses = np.concatenate([np.abs(pred - 0.0), np.abs(pred - 1.0)])
    weights = np.concatenate([mu * (1.0 - p), mu * p])
    return cvar_variational(losses, alpha, weights)


def finite_observed_cvar(world: FrontierWorld, pred: np.ndarray, alpha: float = ALPHA) -> float:
    selected_mu = world.mu * world.pi
    selected_mu = selected_mu / selected_mu.sum()
    return finite_cvar_from_label_law(selected_mu, world.p1, pred, alpha)


def finite_envelope(
    world: FrontierWorld,
    pred: np.ndarray,
    gamma: float,
    alpha: float = ALPHA,
    upper: bool = True,
    lam: np.ndarray | None = None,
    eta_grid: np.ndarray | None = None,
) -> tuple[float, float]:
    if lam is None:
        lam = np.zeros_like(world.mu)
    p_lo, p_hi = sensitivity_bounds(world.p1, gamma)
    if eta_grid is None:
        eta_grid = np.unique(np.concatenate(([0.0, 1.0], pred, 1.0 - pred)))
    eta = eta_grid[:, None]
    a0 = np.maximum(np.abs(pred[None, :] - 0.0) - eta, 0.0)
    a1 = np.maximum(np.abs(pred[None, :] - 1.0) - eta, 0.0)
    selected = world.p1[None, :] * a1 + (1.0 - world.p1[None, :]) * a0
    revealed = world.p0[None, :] * a1 + (1.0 - world.p0[None, :]) * a0
    if upper:
        q = np.where(a1 >= a0, p_hi[None, :], p_lo[None, :])
    else:
        q = np.where(a1 >= a0, p_lo[None, :], p_hi[None, :])
    hidden = q * a1 + (1.0 - q) * a0
    total = np.sum(
        world.mu[None, :]
        * (
            world.pi[None, :] * selected
            + (1.0 - world.pi[None, :])
            * (lam[None, :] * revealed + (1.0 - lam[None, :]) * hidden)
        ),
        axis=1,
    )
    values = eta_grid + total / (1.0 - alpha)
    idx = int(np.argmin(values))
    return float(values[idx]), float(eta_grid[idx])


def frontier_candidates(world: FrontierWorld, gamma: float) -> dict[str, np.ndarray]:
    p_lo, p_hi = sensitivity_bounds(world.p1, gamma)
    return {
        "selected-label score": world.p1,
        "midpoint certificate score": 0.5 * (p_lo + p_hi),
        "upper-certificate score": p_hi,
        "oracle full-label score": world.p_full,
    }


def select_frontier_predictors(world: FrontierWorld, gamma: float) -> dict[str, tuple[str, np.ndarray]]:
    candidates = frontier_candidates(world, gamma)
    return {
        "Observed-selected": ("selected-label score", candidates["selected-label score"]),
        "Certificate": ("upper-certificate score", candidates["upper-certificate score"]),
        "Oracle": ("oracle full-label score", candidates["oracle full-label score"]),
    }


def fixed_tiv_terms(world: FrontierWorld, pred: np.ndarray, eta: float, gamma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_lo, p_hi = sensitivity_bounds(world.p1, gamma)
    a0 = np.maximum(np.abs(pred - 0.0) - eta, 0.0)
    a1 = np.maximum(np.abs(pred - 1.0) - eta, 0.0)
    delta = np.abs(a1 - a0)
    width = p_hi - p_lo
    tau = (1.0 - world.pi) * width * delta / BETA
    return tau, width, delta


def lambda_for_budget(score: np.ndarray, world: FrontierWorld, budget: float) -> np.ndarray:
    lam = np.zeros_like(world.mu)
    order = np.argsort(score)[::-1]
    remaining = float(budget)
    for j in order:
        if remaining <= 1e-14:
            break
        cj = world.mu[j] * world.cost[j]
        if cj <= 0:
            if score[j] > -np.inf:
                lam[j] = 1.0
            continue
        take = min(1.0, remaining / cj)
        lam[j] = take
        remaining -= take * cj
    return lam


def certification_debt(world: FrontierWorld, tau: np.ndarray, lam: np.ndarray) -> float:
    return float(np.sum(world.mu * (1.0 - lam) * tau))


def uniform_threshold_debt(world: FrontierWorld, pred: np.ndarray, lam: np.ndarray, gamma: float) -> float:
    p_lo, p_hi = sensitivity_bounds(world.p1, gamma)
    eta_grid = np.linspace(0.0, 1.0, 61)
    eta = eta_grid[:, None]
    a0 = np.maximum(np.abs(pred[None, :] - 0.0) - eta, 0.0)
    a1 = np.maximum(np.abs(pred[None, :] - 1.0) - eta, 0.0)
    delta = np.abs(a1 - a0)
    debts = np.sum(
        world.mu[None, :]
        * (1.0 - lam[None, :])
        * (1.0 - world.pi[None, :])
        * (p_hi[None, :] - p_lo[None, :])
        * delta
        / BETA,
        axis=1,
    )
    return float(np.max(debts))


def cost_to_tolerance(cur: pd.DataFrame, tolerance: float) -> float:
    hit = cur[cur["relative_debt"] <= tolerance]
    if hit.empty:
        return float(cur["budget_fraction"].max())
    return float(hit["budget_fraction"].iloc[0])


def compatible_gamma(p0: np.ndarray, p1: np.ndarray) -> float:
    """Smallest odds-ratio sensitivity budget containing p0 around p1."""
    p0 = np.clip(np.asarray(p0, dtype=float), 1e-6, 1.0 - 1e-6)
    p1 = np.clip(np.asarray(p1, dtype=float), 1e-6, 1.0 - 1e-6)
    shift = np.abs(logit(p0) - logit(p1))
    return float(np.exp(np.max(shift)))


def make_semisynthetic_diabetes_world(rng: np.random.Generator, k: int = SEMISYNTH_CELLS) -> tuple[FrontierWorld, pd.DataFrame, float]:
    """Real covariates/outcomes with an imposed selective-label channel.

    We use the bundled scikit-learn diabetes dataset as a fully labeled finite
    population and binarize the outcome as high disease progression. Selection
    is then imposed by construction so that high-progression high-risk cases are
    less likely to have labels revealed historically.
    """
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    data = load_diabetes()
    x = data.data
    y = (data.target >= np.quantile(data.target, SEMISYNTH_DIABETES_QUANTILE)).astype(float)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=SEED))
    model.fit(x, y)
    risk = model.predict_proba(x)[:, 1]

    quantiles = np.quantile(risk, np.linspace(0.0, 1.0, k + 1))
    quantiles[0] -= 1e-9
    quantiles[-1] += 1e-9
    cell = np.digitize(risk, quantiles[1:-1], right=True)

    # Outcome-dependent selection: high-risk, high-progression cases are hidden most often.
    s_prob = 0.85 - 0.25 * risk - 0.55 * y - 0.20 * y * risk
    s_prob += rng.normal(0.0, 0.01, size=len(y))
    s_prob = np.clip(s_prob, 0.03, 0.95)

    rows = []
    mu = np.zeros(k)
    pi = np.zeros(k)
    p1 = np.zeros(k)
    p0 = np.zeros(k)
    score = np.zeros(k)
    cost = np.zeros(k)
    n = len(y)
    for j in range(k):
        mask = cell == j
        if not np.any(mask):
            continue
        mu[j] = mask.mean()
        score[j] = float(np.mean(risk[mask]))
        pi[j] = float(np.mean(s_prob[mask]))
        selected_mass = float(np.sum(s_prob[mask]))
        hidden_mass = float(np.sum(1.0 - s_prob[mask]))
        p1[j] = float(np.sum(y[mask] * s_prob[mask]) / max(selected_mass, 1e-12))
        p0[j] = float(np.sum(y[mask] * (1.0 - s_prob[mask])) / max(hidden_mass, 1e-12))
        # Higher-risk strata are more expensive to review but not prohibitively so.
        cost[j] = 0.65 + 1.15 * score[j] + 0.20 * (j % 3 == 0)
        rows.append(
            {
                "cell": j,
                "n": int(np.sum(mask)),
                "mass": mu[j],
                "mean_model_risk": score[j],
                "reveal_probability": pi[j],
                "selected_high_progression_rate": p1[j],
                "hidden_high_progression_rate": p0[j],
                "full_high_progression_rate": float(np.mean(y[mask])),
                "audit_cost": cost[j],
            }
        )

    keep = mu > 0
    world = FrontierWorld(
        mu=mu[keep] / mu[keep].sum(),
        pi=np.clip(pi[keep], 1e-4, 1.0 - 1e-4),
        p1=np.clip(p1[keep], 1e-4, 1.0 - 1e-4),
        p0=np.clip(p0[keep], 1e-4, 1.0 - 1e-4),
        cost=np.maximum(cost[keep], 1e-4),
        score=score[keep],
    )
    gamma_eval = max(2.0, 1.10 * compatible_gamma(world.p0, world.p1))
    return world, pd.DataFrame(rows), gamma_eval


def experiment_7_semisynthetic_diabetes_case(rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    """Semi-synthetic selective-label case study using real diabetes covariates."""
    world, cells, gamma_eval = make_semisynthetic_diabetes_world(rng)
    budget_grid = np.linspace(0.0, 1.0, 51)
    bridge_grid = np.linspace(0.0, 1.0, 21)
    candidates = select_frontier_predictors(world, gamma_eval)
    model_rows: list[dict[str, float | str]] = []
    for role, (_, pred) in candidates.items():
        model_rows.append(
            {
                "role": role,
                "observed_cvar": finite_observed_cvar(world, pred),
                "true_cvar": finite_cvar_from_label_law(world.mu, world.p_full, pred),
                "upper_certificate": finite_envelope(world, pred, gamma_eval, upper=True)[0],
            }
        )

    pred = frontier_candidates(world, gamma_eval)["selected-label score"]
    tau, width, contrast = fixed_tiv_terms(world, pred, FRONTIER_ETA, gamma_eval)
    w0 = float(np.sum(world.mu * tau))
    expected_selected_loss = world.p1 * np.abs(pred - 1.0) + (1.0 - world.p1) * np.abs(pred - 0.0)
    deterministic_scores = {
        "TIV/cost": np.where(world.cost > 0, tau / world.cost, np.inf),
        "Oracle frontier": np.where(world.cost > 0, tau / world.cost, np.inf),
        "High density": world.mu,
        "Low reveal prob.": -world.pi,
        "High pred. loss": expected_selected_loss,
        "Uncertainty": world.p1 * (1.0 - world.p1),
        "Width only": width,
        "Contrast only": contrast,
    }
    total_cost = float(np.sum(world.mu * world.cost))
    curve_rows: list[dict[str, float | str]] = []
    for policy, score in deterministic_scores.items():
        for frac in budget_grid:
            lam = lambda_for_budget(score, world, frac * total_cost)
            debt = certification_debt(world, tau, lam)
            curve_rows.append(
                {
                    "policy": policy,
                    "budget_fraction": float(frac),
                    "debt": debt,
                    "relative_debt": debt / w0,
                }
            )

    # Average random cell-order audits to keep the curve stable.
    random_debts = {float(frac): [] for frac in budget_grid}
    for _ in range(SEMISYNTH_RANDOM_POLICIES):
        score = rng.random(len(world.mu))
        for frac in budget_grid:
            lam = lambda_for_budget(score, world, frac * total_cost)
            random_debts[float(frac)].append(certification_debt(world, tau, lam))
    for frac in budget_grid:
        debt = float(np.mean(random_debts[float(frac)]))
        curve_rows.append(
            {
                "policy": "Random",
                "budget_fraction": float(frac),
                "debt": debt,
                "relative_debt": debt / w0,
            }
        )

    bridge_rows: list[dict[str, float]] = []
    d0 = uniform_threshold_debt(world, pred, np.zeros(len(world.mu)), gamma_eval)
    for frac in bridge_grid:
        lam = lambda_for_budget(deterministic_scores["TIV/cost"], world, frac * total_cost)
        upper = finite_envelope(world, pred, gamma_eval, upper=True, lam=lam, eta_grid=np.linspace(0, 1, 61))[0]
        lower = finite_envelope(world, pred, gamma_eval, upper=False, lam=lam, eta_grid=np.linspace(0, 1, 61))[0]
        d_unif = uniform_threshold_debt(world, pred, lam, gamma_eval)
        bridge_rows.append(
            {
                "budget_fraction": float(frac),
                "optimized_width": upper - lower,
                "uniform_bound": d_unif,
                "relative_optimized_width": (upper - lower) / d0 if d0 > 0 else 0.0,
                "relative_uniform_bound": d_unif / d0 if d0 > 0 else 0.0,
            }
        )

    curves = pd.DataFrame(curve_rows)
    models = pd.DataFrame(model_rows)
    bridge = pd.DataFrame(bridge_rows)
    cells = cells.assign(gamma_eval=gamma_eval)
    cells.to_csv(RES_DIR / "experiment7_semisynthetic_cells.csv", index=False)
    curves.to_csv(RES_DIR / "experiment7_semisynthetic_certification_curves.csv", index=False)
    models.to_csv(RES_DIR / "experiment7_semisynthetic_model_selection.csv", index=False)
    bridge.to_csv(RES_DIR / "experiment7_semisynthetic_optimized_bridge.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))

    ax = axes[0, 0]
    xloc = np.arange(len(cells))
    ax.plot(xloc, cells["full_high_progression_rate"], color="#F58518", linewidth=1.8, label="Full")
    ax.plot(xloc, cells["selected_high_progression_rate"], color="#4C78A8", linewidth=1.8, label="Selected")
    ax.plot(xloc, cells["hidden_high_progression_rate"], color="#E45756", linewidth=1.8, label="Hidden")
    ax2 = ax.twinx()
    ax2.plot(xloc, cells["reveal_probability"], color="#6b7280", linewidth=1.2, linestyle="--", label="Reveal prob.")
    ax.set_xlabel("Risk-score cell", fontsize=7)
    ax.set_ylabel("High-progression rate", fontsize=7)
    ax2.set_ylabel("Reveal probability", fontsize=7)
    ax.set_ylim(-0.03, 1.03)
    ax2.set_ylim(-0.03, 1.03)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=True, fontsize=5.8, loc="upper left")
    ax.text(-0.16, 1.05, "(a)", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")

    ax = axes[0, 1]
    model_summary = models.set_index("role").loc[["Observed-selected", "Certificate", "Oracle"]]
    width_bar = 0.24
    groups = np.arange(3)
    for i, (metric, label, color) in enumerate(
        [
            ("observed_cvar", "Observed", "#4C78A8"),
            ("true_cvar", "True", "#F58518"),
            ("upper_certificate", "Upper cert.", "#54A24B"),
        ]
    ):
        ax.bar(groups + (i - 1) * width_bar, model_summary[metric], width=width_bar, label=label, color=color)
    ax.set_xticks(groups)
    ax.set_xticklabels(["Obs.-sel.", "Cert.", "Oracle"], fontsize=7)
    ax.set_ylabel("CVaR / certificate", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True, fontsize=5.8)
    ax.text(-0.16, 1.05, "(b)", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")

    ax = axes[1, 0]
    colors = {
        "TIV/cost": "#1f77b4",
        "Oracle frontier": "#000000",
        "Random": "#9467bd",
        "High density": "#d62728",
        "Low reveal prob.": "#8c564b",
        "High pred. loss": "#ff7f0e",
        "Uncertainty": "#2ca02c",
        "Width only": "#17becf",
        "Contrast only": "#bcbd22",
    }
    order = ["TIV/cost", "Oracle frontier", "Random", "High density", "Low reveal prob.", "High pred. loss", "Uncertainty", "Width only", "Contrast only"]
    for policy in order:
        cur = curves[curves["policy"] == policy].sort_values("budget_fraction")
        ax.plot(
            cur["budget_fraction"],
            cur["relative_debt"],
            label=policy,
            color=colors[policy],
            linestyle="--" if policy == "Oracle frontier" else "-",
            linewidth=1.8 if policy in {"TIV/cost", "Oracle frontier"} else 1.0,
        )
    ax.set_xlabel("Audit cost / full audit cost", fontsize=7)
    ax.set_ylabel("Remaining cert. debt", fontsize=7)
    ax.set_ylim(-0.02, 1.03)
    ax.legend(frameon=True, fontsize=4.8, ncol=2, loc="upper right")
    ax.text(-0.16, 1.05, "(c)", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")

    ax = axes[1, 1]
    ax.plot(bridge["budget_fraction"], bridge["relative_optimized_width"], color="#F58518", linewidth=2.0, label="Optimized interval")
    ax.plot(bridge["budget_fraction"], bridge["relative_uniform_bound"], color="#4C78A8", linewidth=2.0, linestyle="--", label=r"$D_f^{unif}$ bound")
    ax.set_xlabel("TIV/cost audit budget", fontsize=7)
    ax.set_ylabel("Relative width", fontsize=7)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(frameon=True, fontsize=6.0)
    ax.text(-0.16, 1.05, "(d)", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")

    for ax in axes.ravel():
        ax.tick_params(axis="both", labelsize=6.5)
    fig.tight_layout(w_pad=1.05, h_pad=1.0)
    save_figure(fig, "semisynthetic_diabetes_case")
    plt.close(fig)
    return {"cells": cells, "curves": curves, "models": models, "bridge": bridge}


def experiment_6_certification_frontier_stress(
    rng: np.random.Generator,
    n_worlds: int = FRONTIER_WORLDS,
    k: int = FRONTIER_CELLS,
    gamma0: float = FRONTIER_GAMMA0,
    gamma_eval: float = FRONTIER_GAMMA_EVAL,
) -> dict[str, pd.DataFrame]:
    """Stress test Theorem 3 in random finite-cell populations."""
    budget_grid = np.linspace(0.0, 1.0, 51)
    bridge_grid = np.linspace(0.0, 1.0, 21)
    policies = [
        "TIV/cost",
        "Oracle frontier",
        "Random",
        "High density",
        "Low reveal prob.",
        "High pred. loss",
        "Uncertainty",
        "Width only",
        "Contrast only",
    ]
    curve_rows: list[dict[str, float | int | str]] = []
    model_rows: list[dict[str, float | int | str]] = []
    ratio_rows: list[dict[str, float | int | str]] = []
    bridge_rows: list[dict[str, float | int | str]] = []

    for world_id in range(n_worlds):
        world = generate_frontier_world(rng, k=k, gamma0=gamma0)
        selected = select_frontier_predictors(world, gamma_eval)
        for role, (_, pred) in selected.items():
            model_rows.append(
                {
                    "world": world_id,
                    "role": role,
                    "observed_cvar": finite_observed_cvar(world, pred),
                    "true_cvar": finite_cvar_from_label_law(world.mu, world.p_full, pred),
                    "upper_certificate": finite_envelope(world, pred, gamma_eval, upper=True)[0],
                }
            )

        pred = frontier_candidates(world, gamma_eval)["selected-label score"]
        eta = FRONTIER_ETA
        tau, width, contrast = fixed_tiv_terms(world, pred, eta, gamma_eval)
        w0 = float(np.sum(world.mu * tau))
        if w0 <= 1e-12:
            continue
        expected_selected_loss = world.p1 * np.abs(pred - 1.0) + (1.0 - world.p1) * np.abs(pred - 0.0)
        scores = {
            "TIV/cost": np.where(world.cost > 0, tau / world.cost, np.inf),
            "Oracle frontier": np.where(world.cost > 0, tau / world.cost, np.inf),
            "Random": rng.random(k),
            "High density": world.mu,
            "Low reveal prob.": -world.pi,
            "High pred. loss": expected_selected_loss,
            "Uncertainty": world.p1 * (1.0 - world.p1),
            "Width only": width,
            "Contrast only": contrast,
        }
        total_cost = float(np.sum(world.mu * world.cost))
        for policy in policies:
            for frac in budget_grid:
                lam = lambda_for_budget(scores[policy], world, frac * total_cost)
                debt = certification_debt(world, tau, lam)
                curve_rows.append(
                    {
                        "world": world_id,
                        "policy": policy,
                        "budget_fraction": float(frac),
                        "debt": debt,
                        "relative_debt": debt / w0,
                    }
                )

        world_curves = pd.DataFrame([r for r in curve_rows if r["world"] == world_id])
        tiv_curve = world_curves[world_curves["policy"] == "TIV/cost"].sort_values("budget_fraction")
        for tolerance in [0.50, 0.25, 0.10]:
            tiv_cost = max(cost_to_tolerance(tiv_curve, tolerance), 1e-8)
            for policy in policies:
                if policy in {"TIV/cost", "Oracle frontier"}:
                    continue
                cur = world_curves[world_curves["policy"] == policy].sort_values("budget_fraction")
                ratio_rows.append(
                    {
                        "world": world_id,
                        "policy": policy,
                        "tolerance": tolerance,
                        "ratio": cost_to_tolerance(cur, tolerance) / tiv_cost,
                    }
                )

        d0 = uniform_threshold_debt(world, pred, np.zeros(k), gamma_eval)
        for frac in bridge_grid:
            lam = lambda_for_budget(scores["TIV/cost"], world, frac * total_cost)
            upper = finite_envelope(world, pred, gamma_eval, upper=True, lam=lam, eta_grid=np.linspace(0, 1, 61))[0]
            lower = finite_envelope(world, pred, gamma_eval, upper=False, lam=lam, eta_grid=np.linspace(0, 1, 61))[0]
            d_unif = uniform_threshold_debt(world, pred, lam, gamma_eval)
            bridge_rows.append(
                {
                    "world": world_id,
                    "budget_fraction": float(frac),
                    "optimized_width": upper - lower,
                    "uniform_bound": d_unif,
                    "relative_optimized_width": (upper - lower) / d0 if d0 > 0 else 0.0,
                    "relative_uniform_bound": d_unif / d0 if d0 > 0 else 0.0,
                }
            )

    curves = pd.DataFrame(curve_rows)
    models = pd.DataFrame(model_rows)
    ratios = pd.DataFrame(ratio_rows)
    bridge = pd.DataFrame(bridge_rows)
    curves.to_csv(RES_DIR / "experiment6_certification_frontier_curves.csv", index=False)
    models.to_csv(RES_DIR / "experiment6_model_selection.csv", index=False)
    ratios.to_csv(RES_DIR / "experiment6_cost_to_certify_ratios.csv", index=False)
    bridge.to_csv(RES_DIR / "experiment6_optimized_bridge.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.35))
    colors = {
        "TIV/cost": "#1f77b4",
        "Oracle frontier": "#000000",
        "Random": "#9467bd",
        "High density": "#d62728",
        "Low reveal prob.": "#8c564b",
        "High pred. loss": "#ff7f0e",
        "Uncertainty": "#2ca02c",
        "Width only": "#17becf",
        "Contrast only": "#bcbd22",
    }

    ax = axes[0, 0]
    model_summary = (
        models.groupby("role")[["observed_cvar", "true_cvar", "upper_certificate"]]
        .median()
        .loc[["Observed-selected", "Certificate", "Oracle"]]
    )
    xloc = np.arange(3)
    width_bar = 0.24
    for i, (metric, label, color) in enumerate(
        [
            ("observed_cvar", "Observed", "#4C78A8"),
            ("true_cvar", "True", "#F58518"),
            ("upper_certificate", "Upper cert.", "#54A24B"),
        ]
    ):
        ax.bar(xloc + (i - 1) * width_bar, model_summary[metric], width=width_bar, label=label, color=color)
    ax.set_xticks(xloc)
    ax.set_xticklabels(["Obs.-sel.", "Cert.", "Oracle"], fontsize=7)
    ax.set_ylabel("CVaR / certificate", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True, fontsize=5.5)
    ax.text(-0.16, 1.05, "(a)", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")

    ax = axes[0, 1]
    plot_policies = ["TIV/cost", "Oracle frontier", "Random", "High density", "Low reveal prob.", "High pred. loss", "Uncertainty", "Width only", "Contrast only"]
    for policy in plot_policies:
        cur = curves[curves["policy"] == policy]
        summary = cur.groupby("budget_fraction")["relative_debt"].quantile([0.25, 0.5, 0.75]).unstack()
        linestyle = "--" if policy == "Oracle frontier" else "-"
        linewidth = 1.8 if policy in {"TIV/cost", "Oracle frontier"} else 1.0
        ax.plot(summary.index, summary[0.5], label=policy, color=colors[policy], linestyle=linestyle, linewidth=linewidth)
        if policy == "TIV/cost":
            ax.fill_between(summary.index.to_numpy(), summary[0.25].to_numpy(), summary[0.75].to_numpy(), color=colors[policy], alpha=0.16, linewidth=0)
    ax.set_xlabel("Audit cost / full audit cost", fontsize=7)
    ax.set_ylabel("Remaining cert. debt", fontsize=7)
    ax.set_ylim(-0.02, 1.03)
    ax.legend(frameon=True, fontsize=4.7, ncol=2, loc="upper right")
    ax.text(-0.16, 1.05, "(b)", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")

    ax = axes[1, 0]
    ratio_tol = 0.25
    ratio_plot = ratios[ratios["tolerance"] == ratio_tol]
    ratio_order = ["Random", "High density", "Low reveal prob.", "High pred. loss", "Uncertainty", "Width only", "Contrast only"]
    med = ratio_plot.groupby("policy")["ratio"].median().reindex(ratio_order)
    q25 = ratio_plot.groupby("policy")["ratio"].quantile(0.25).reindex(ratio_order)
    q75 = ratio_plot.groupby("policy")["ratio"].quantile(0.75).reindex(ratio_order)
    xpos = np.arange(len(ratio_order))
    ax.bar(xpos, med, color=[colors[p] for p in ratio_order], alpha=0.88)
    ax.errorbar(xpos, med, yerr=[med - q25, q75 - med], fmt="none", ecolor="black", elinewidth=0.8, capsize=2)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(xpos)
    ax.set_xticklabels(["Rand.", "Density", "Low pi", "Loss", "Uncert.", "Width", "Contrast"], rotation=20, ha="right", fontsize=6.2)
    ax.set_ylabel("Cost ratio to 25% debt", fontsize=7)
    ax.set_ylim(0, max(2.0, float(np.nanmax(q75)) * 1.12))
    ax.text(-0.16, 1.05, "(c)", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")

    ax = axes[1, 1]
    bridge_summary = bridge.groupby("budget_fraction")[["relative_optimized_width", "relative_uniform_bound"]].median()
    ax.plot(bridge_summary.index, bridge_summary["relative_optimized_width"], color="#F58518", linewidth=2.0, label="Optimized interval")
    ax.plot(bridge_summary.index, bridge_summary["relative_uniform_bound"], color="#4C78A8", linewidth=2.0, linestyle="--", label=r"$D_f^{unif}$ bound")
    ax.set_xlabel("TIV/cost audit budget", fontsize=7)
    ax.set_ylabel("Relative width", fontsize=7)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(frameon=True, fontsize=6.0)
    ax.text(-0.16, 1.05, "(d)", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")

    for ax in axes.ravel():
        ax.tick_params(axis="both", labelsize=6.5)
    fig.tight_layout(w_pad=1.05, h_pad=1.0)
    save_figure(fig, "synthetic_three_panel")
    save_figure(fig, "certification_frontier_stress_test")
    plt.close(fig)

    return {"curves": curves, "models": models, "ratios": ratios, "bridge": bridge}


def write_run_summary(outputs: dict[str, object], filename: str = "run_summary.json") -> None:
    def convert(value: object) -> object:
        if isinstance(value, pd.DataFrame):
            return value.to_dict(orient="records")
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        return value

    serializable = {}
    for key, value in outputs.items():
        serializable[key] = convert(value)
    with open(RES_DIR / filename, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main-figure",
        action="store_true",
        help="Regenerate only the conceptual figure and certification-frontier stress test used in the main paper.",
    )
    parser.add_argument(
        "--semisynthetic",
        action="store_true",
        help="Regenerate only the semi-synthetic diabetes selective-label case study.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    if args.semisynthetic:
        exp7 = experiment_7_semisynthetic_diabetes_case(rng)
        write_run_summary({"experiment7": exp7}, "run_summary_semisynthetic.json")
        print("Wrote semi-synthetic case-study figure to", FIG_DIR / "semisynthetic_diabetes_case.png")
        print("Wrote semi-synthetic results to", RES_DIR)
        return

    if args.main_figure:
        make_conceptual_certification_figure()
        exp6 = experiment_6_certification_frontier_stress(rng)
        exp7 = experiment_7_semisynthetic_diabetes_case(rng)
        write_run_summary({"experiment6": exp6, "experiment7": exp7}, "run_summary_main_figure.json")
        print("Wrote conceptual figure to", FIG_DIR / "conceptual_tail_risk_certification.png")
        print("Wrote certification-frontier stress-test figure to", FIG_DIR / "synthetic_three_panel.png")
        print("Wrote semi-synthetic case-study figure to", FIG_DIR / "semisynthetic_diabetes_case.png")
        print("Wrote stress-test results to", RES_DIR)
        return

    make_conceptual_certification_figure()
    exp1 = experiment_1()
    exp2 = experiment_2(rng)
    exp3 = experiment_3(rng)
    exp4 = experiment_4(rng)
    exp5 = experiment_5_sensitivity()
    exp6 = experiment_6_certification_frontier_stress(rng)
    exp7 = experiment_7_semisynthetic_diabetes_case(rng)
    write_run_summary({
        "experiment1": exp1,
        "experiment2": exp2,
        "experiment3": exp3,
        "experiment4": exp4,
        "experiment5": exp5,
        "experiment6": exp6,
        "experiment7": exp7,
    })

    print("Wrote figures to", FIG_DIR)
    print("Wrote results to", RES_DIR)


if __name__ == "__main__":
    main()
