"""Finite-sample certification of the sharp CVaR interval: identification width versus statistical inflation.

Random finite-cell worlds satisfy the Gamma odds-ratio model. In each world the sharp population interval
[rho_L, rho_U] and the CVaR under the data-generating law are computed exactly. Repeated selectively labeled samples
are then drawn and both endpoints are estimated
  (i)  with oracle nuisances, together with the nonasymptotic outer certificate C = [rho_L_hat - r, rho_U_hat + r],
       where r = beta^{-1} sqrt(log(4|T|/delta)/(2n)) (Hoeffding over the two endpoints and the finite set T of
       candidate thresholds, on which both Rockafellar-Uryasev objectives are piecewise linear), and
  (ii) with two-fold cross-fitted cell-frequency nuisances and the locally robust score of Theorem 4.

Outputs: results/finite_sample_replications.csv, results/finite_sample_summary.csv, results/finite_sample_config.json,
figures/finite_sample_decomposition.{pdf,png}.
"""
import argparse
import csv
import json
import math
import pathlib
import platform
import time

import numpy as np


def q_plus(p, gamma):
    return gamma * p / (1.0 - p + gamma * p)


def q_minus(p, gamma):
    return p / (p + gamma * (1.0 - p))


def dq_plus(p, gamma):
    return gamma / (1.0 + (gamma - 1.0) * p) ** 2


def dq_minus(p, gamma):
    return gamma / (gamma - (gamma - 1.0) * p) ** 2


def endpoint_maps(a0, a1, p, gamma):
    """Sign-free upper and lower endpoint values b and their derivatives in p."""
    d = a1 - a0
    d_pos, d_neg = np.maximum(d, 0.0), np.maximum(-d, 0.0)
    qp, qm = q_plus(p, gamma), q_minus(p, gamma)
    dqp, dqm = dq_plus(p, gamma), dq_minus(p, gamma)
    b = {"U": a0 + qp * d_pos - qm * d_neg, "L": a0 + qm * d_pos - qp * d_neg}
    db = {"U": dqp * d_pos - dqm * d_neg, "L": dqm * d_pos - dqp * d_neg}
    return b, db


def make_world(rng, n_cells, gamma, pi_min):
    mu = rng.dirichlet(np.ones(n_cells))
    pi = rng.uniform(pi_min, 0.95, n_cells)
    p1 = rng.uniform(0.05, 0.95, n_cells)
    # The data-generating hidden-label law is a random compatible point of the Gamma interval.
    p0 = q_minus(p1, gamma) + rng.uniform(0.0, 1.0, n_cells) * (q_plus(p1, gamma) - q_minus(p1, gamma))
    forecast = np.clip(p1 + rng.normal(0.0, 0.15, n_cells), 0.0, 1.0)
    loss0, loss1 = forecast ** 2, (1.0 - forecast) ** 2  # Brier loss, bounded in [0, 1]
    thresholds = np.unique(np.concatenate([[0.0, 1.0], loss0, loss1]))
    a0 = np.maximum(loss0[None, :] - thresholds[:, None], 0.0)  # shape (|T|, K)
    a1 = np.maximum(loss1[None, :] - thresholds[:, None], 0.0)
    return dict(mu=mu, pi=pi, p1=p1, p0=p0, T=thresholds, a0=a0, a1=a1)


def population_values(w, gamma, beta):
    b, _ = endpoint_maps(w["a0"], w["a1"], w["p1"], gamma)
    selected = w["pi"] * (w["p1"] * w["a1"] + (1.0 - w["p1"]) * w["a0"])
    rho = {}
    for s in ("U", "L"):
        m = (w["mu"] * (selected + (1.0 - w["pi"]) * b[s])).sum(axis=1)
        rho[s] = float(np.min(w["T"] + m / beta))
    hidden = w["p0"] * w["a1"] + (1.0 - w["p0"]) * w["a0"]
    m_true = (w["mu"] * (selected + (1.0 - w["pi"]) * hidden)).sum(axis=1)
    return rho["L"], rho["U"], float(np.min(w["T"] + m_true / beta))


def draw_counts(rng, w, n):
    probs = np.concatenate([w["mu"] * w["pi"] * w["p1"], w["mu"] * w["pi"] * (1.0 - w["p1"]), w["mu"] * (1.0 - w["pi"])])
    counts = rng.multinomial(n, probs / probs.sum()).reshape(3, -1)
    return counts[0], counts[1], counts[2]  # (S=1,Y=1), (S=1,Y=0), (S=0) per cell


def oracle_endpoints(w, counts, gamma, beta, n):
    c11, c10, c0 = counts
    b, _ = endpoint_maps(w["a0"], w["a1"], w["p1"], gamma)
    out = {}
    for s in ("U", "L"):
        m = (c11 * w["a1"] + c10 * w["a0"] + c0 * b[s]).sum(axis=1) / n
        out[s] = float(np.min(w["T"] + m / beta))
    return out["L"], out["U"]


def crossfit_endpoints(rng, w, counts, gamma, beta, n, pi_min):
    """Two-fold cross-fitted cell-frequency nuisances plugged into the locally robust score."""
    folds = []
    for c in counts:
        half = rng.binomial(c, 0.5)
        folds.append((half, c - half))
    sums = {"U": np.zeros(len(w["T"])), "L": np.zeros(len(w["T"]))}
    for k in (0, 1):
        c11, c10, c0 = (folds[i][k] for i in range(3))
        o11, o10, o0 = (folds[i][1 - k] for i in range(3))
        sel = o11 + o10
        p_hat = np.clip((o11 + 0.5) / (sel + 1.0), 0.01, 0.99)
        tot = sel + o0
        pi_hat = np.where(tot > 0, sel / np.maximum(tot, 1), 0.5)
        pi_hat = np.clip(pi_hat, pi_min / 2.0, 1.0)
        weight = (1.0 - pi_hat) / pi_hat
        b, db = endpoint_maps(w["a0"], w["a1"], p_hat, gamma)
        for s in ("U", "L"):
            score = (c11 * (w["a1"] + weight * db[s] * (1.0 - p_hat))
                     + c10 * (w["a0"] + weight * db[s] * (0.0 - p_hat))
                     + c0 * b[s])
            sums[s] += score.sum(axis=1)
    return (float(np.min(w["T"] + sums["L"] / n / beta)), float(np.min(w["T"] + sums["U"] / n / beta)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worlds", type=int, default=60)
    parser.add_argument("--reps", type=int, default=80)
    parser.add_argument("--cells", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--pi-min", type=float, default=0.2)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--n-grid", type=int, nargs="+", default=[1000, 3000, 10000, 30000, 100000])
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parent)
    parser.add_argument("--plot-only", action="store_true", help="redraw the figure from the summary CSV")
    args = parser.parse_args()

    beta = 1.0 - args.alpha
    results_dir, figures_dir = args.out / "results", args.out / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        with open(results_dir / "finite_sample_summary.csv", newline="") as fh:
            plot(list(csv.DictReader(fh)), figures_dir)
        return
    start = time.time()
    rows = []
    master = np.random.SeedSequence(args.seed)
    for world_id, world_seed in enumerate(master.spawn(args.worlds)):
        rng = np.random.default_rng(world_seed)
        w = make_world(rng, args.cells, args.gamma, args.pi_min)
        rho_l, rho_u, rho_true = population_values(w, args.gamma, beta)
        for n in args.n_grid:
            radius = math.sqrt(math.log(4 * len(w["T"]) / args.delta) / (2 * n)) / beta
            for rep in range(args.reps):
                counts = draw_counts(rng, w, n)
                hat_l, hat_u = oracle_endpoints(w, counts, args.gamma, beta, n)
                lo, hi = max(0.0, hat_l - radius), min(1.0, hat_u + radius)
                cf_l, cf_u = crossfit_endpoints(rng, w, counts, args.gamma, beta, n, args.pi_min)
                rows.append(dict(world=world_id, n=n, rep=rep, rho_lower=rho_l, rho_upper=rho_u,
                                 rho_true=rho_true, id_width=rho_u - rho_l, radius=radius,
                                 oracle_lower=hat_l, oracle_upper=hat_u, cert_lower=lo, cert_upper=hi,
                                 cert_width=hi - lo, covers_true=int(lo <= rho_true <= hi),
                                 covers_interval=int(lo <= rho_l and rho_u <= hi),
                                 crossfit_lower=cf_l, crossfit_upper=cf_u))
        print(f"world {world_id + 1}/{args.worlds} done ({time.time() - start:.1f}s)", flush=True)

    with open(results_dir / "finite_sample_replications.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for n in args.n_grid:
        sub = [r for r in rows if r["n"] == n]
        mean = lambda key: float(np.mean([r[key] for r in sub]))  # noqa: E731
        err_u = np.array([r["crossfit_upper"] - r["rho_upper"] for r in sub])
        err_l = np.array([r["crossfit_lower"] - r["rho_lower"] for r in sub])
        summary.append(dict(
            n=n, replications=len(sub), id_width=mean("id_width"),
            oracle_width=float(np.mean([r["oracle_upper"] - r["oracle_lower"] for r in sub])),
            crossfit_width=float(np.mean([r["crossfit_upper"] - r["crossfit_lower"] for r in sub])),
            radius=mean("radius"), cert_width=mean("cert_width"),
            inflation=float(np.mean([r["cert_width"] - r["id_width"] for r in sub])),
            covers_true=mean("covers_true"), covers_interval=mean("covers_interval"),
            crossfit_rmse_upper=float(np.sqrt(np.mean(err_u ** 2))),
            crossfit_rmse_lower=float(np.sqrt(np.mean(err_l ** 2))),
            crossfit_bias_upper=float(err_u.mean()), crossfit_bias_lower=float(err_l.mean())))
    with open(results_dir / "finite_sample_summary.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    config = dict(vars(args), out=str(args.out), beta=beta, python=platform.python_version(),
                  numpy=np.__version__, seconds=round(time.time() - start, 1))
    (results_dir / "finite_sample_config.json").write_text(json.dumps(config, indent=2))
    plot(summary, figures_dir)
    print(json.dumps(summary, indent=1))


def plot(summary, figures_dir):
    """Draw the figure at its printed size (0.8 of a 5.5-inch text width)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "legend.fontsize": 6.5})
    ns = np.array([float(s["n"]) for s in summary])
    col = lambda key: [float(s[key]) for s in summary]  # noqa: E731
    fig, axes = plt.subplots(1, 2, figsize=(4.4, 1.85))
    ax = axes[0]
    ax.plot(ns, col("cert_width"), "o-", color="#b2182b", ms=3, label="outer certificate")
    ax.plot(ns, col("crossfit_width"), "s--", color="#2166ac", ms=3, label="cross-fitted estimate")
    ax.plot(ns, col("id_width"), "k-", lw=1.0, label="sharp population width")
    ax.set_xscale("log")
    ax.set_xlabel("sample size $n$")
    ax.set_ylabel("interval width")
    ax.legend(frameon=False)
    ax.set_title("(a) identification vs. sampling")
    ax = axes[1]
    ax.plot(ns, col("crossfit_rmse_upper"), "o-", color="#b2182b", ms=3, label="upper endpoint")
    ax.plot(ns, col("crossfit_rmse_lower"), "s--", color="#2166ac", ms=3, label="lower endpoint")
    ax.plot(ns, col("crossfit_rmse_upper")[0] * np.sqrt(ns[0] / ns), ":", color="0.4", label=r"$\propto n^{-1/2}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sample size $n$")
    ax.set_ylabel("RMSE (cross-fitted)")
    ax.legend(frameon=False)
    ax.set_title("(b) both endpoints, same rate")
    fig.tight_layout(pad=0.3, w_pad=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"finite_sample_decomposition.{ext}", dpi=300)


if __name__ == "__main__":
    main()
