# -*- coding: utf-8 -*-
"""
make_rate_fig.py -- regenerate the rate-gap figure from the EXISTING
rate_curve.csv (no need to re-run the hours-long rate sweep).
    python make_rate_fig.py --result-root M:/MRI/Results_RDCCI_v7
Outputs: rate_gap_scatter.pdf / .png  (paper Fig: calibration failure is
predicted by posterior collapse, not by rate) + group stats to console.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-root", default="M:/MRI/Results_RDCCI_v7")
    args = ap.parse_args()
    root = Path(args.result_root)
    df = pd.read_csv(root / "rate_curve.csv")
    df["collapsed"] = df["rate_kl"] < 5.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    # ---- panel A: lam_rd sweep, per-seed scatter ----
    d = df[df["sweep"] == "lam_rd"]
    ax = axes[0]
    for col, ycol, lab in [("#2166ac", "gap_External", "oracle"),
                           ("#1a9850", "gap_sim_External", "CPCI-Sim")]:
        h = d[~d["collapsed"]]
        ax.scatter(h["rate_kl"], h[ycol], c=col, s=20, alpha=0.55,
                   label="%s, healthy" % lab, marker="o" if lab == "oracle" else "^")
        c = d[d["collapsed"]]
        ax.scatter(c["rate_kl"], c[ycol], facecolors="none", edgecolors="#b2182b",
                   s=55, linewidths=1.4, label="%s, collapsed" % lab,
                   marker="o" if lab == "oracle" else "^")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(5.0, color="black", ls="--", lw=1)
    ax.text(5.4, 0.75, "collapse threshold (5 nats)", fontsize=8, rotation=90, va="top")
    ax.set_xlabel("Generator rate KL (nats)")
    ax.set_ylabel("External coverage gap |cov - 0.90|")
    ax.set_title(r"(A) $\lambda_{RD}$ sweep: gap is bimodal, not power-law")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)

    # ---- panel B: latent sweep means ----
    ax = axes[1]
    g = df[df["sweep"] == "latent"].groupby("val").agg(
        gap_or=("gap_External", "mean"), gap_sim=("gap_sim_External", "mean"),
        n_col=("collapsed", "sum"), gap_or_h=(
            "gap_External", lambda s: s[~df.loc[s.index, "collapsed"]].mean()),
        gap_sim_h=("gap_sim_External", lambda s: s[~df.loc[s.index, "collapsed"]].mean()))
    xs = np.arange(len(g))
    ax.plot(xs, g["gap_or"], "o-", color="#2166ac", label="oracle (all seeds)")
    ax.plot(xs, g["gap_sim"], "^-", color="#1a9850", label="CPCI-Sim (all seeds)")
    ax.plot(xs, g["gap_or_h"], "o--", color="#2166ac", alpha=0.5,
            label="oracle (healthy only)")
    ax.plot(xs, g["gap_sim_h"], "^--", color="#1a9850", alpha=0.5,
            label="CPCI-Sim (healthy only)")
    for x, n in zip(xs, g["n_col"]):
        ax.text(x, max(g["gap_or"].max(), g["gap_sim"].max()) * 1.02,
                "%d/20 collapsed" % n, ha="center", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(v) for v in g.index])
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("External coverage gap")
    ax.set_title("(B) Latent sweep: gap rises via training instability")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(root / "rate_gap_scatter.pdf")
    fig.savefig(root / "rate_gap_scatter.png", dpi=300)
    print("figure saved: %s" % (root / "rate_gap_scatter.pdf"))

    # ---- group stats (for the paper text) ----
    print("\n% ---- collapse vs healthy, pooled over both sweeps ----")
    for name, m in [("healthy", ~df["collapsed"]), ("collapsed", df["collapsed"])]:
        sub = df[m]
        print("%-9s n=%3d  oracle gap=%.3f+-%.3f  SIM gap=%.3f+-%.3f"
              % (name, len(sub), sub["gap_External"].mean(), sub["gap_External"].std(),
                 sub["gap_sim_External"].mean(), sub["gap_sim_External"].std()))


if __name__ == "__main__":
    main()