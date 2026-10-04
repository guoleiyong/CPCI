# -*- coding: utf-8 -*-
"""
make_figs_and_tables.py -- generate the three-tier comparison figure and
LaTeX table rows from rd_cci_v8_cpci.py outputs.

Run AFTER `main` (and `rate`) have finished:
    python make_figs_and_tables.py --result-root M:/MRI/Results_RDCCI_v7

Outputs:
    cpci_three_tier.pdf / .png   (Fig: coverage with Wilson CI, 3 tiers x 3 cohorts)
    printed LaTeX table rows     (paste into paper_revision_draft.tex tab:mri)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def wilson_ci(k, n, z=1.96):
    p = k / max(n, 1)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-root", default="M:/MRI/Results_RDCCI_v7")
    args = ap.parse_args()
    root = Path(args.result_root)

    per = pd.read_csv(root / "three_tier_results_per_seed.csv")
    methods = ["CPCI-Sim", "CPCI-Oracle", "CPCI-Scaled",
               "Weighted-CP", "Oracle"]
    cohorts = ["PPMI-internal", "NEUROCON", "TaoWu"]
    nom = 0.90

    # ---------- figure ----------
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    colors = {"CPCI-Sim": "#2166ac", "CPCI-Oracle": "#999999",
              "CPCI-Scaled": "#b2182b", "Weighted-CP": "#66c2a5",
              "Oracle": "#1a9850"}
    xpos, xticks = [], []
    for ci, cname in enumerate(cohorts):
        for mi, meth in enumerate(methods):
            g = per[(per["cohort"] == cname) & (per["method"] == meth)]
            if not len(g):
                continue
            k, n = int(g["covered"].sum()), int(g["n"].sum())
            cov = g["coverage"].mean()
            lo, hi = wilson_ci(k, n)
            x = ci * (len(methods) + 1) + mi
            xpos.append(x); xticks.append((x, meth))
            ax.bar(x, cov - 0.5, bottom=0.5, width=0.8,
                   color=colors.get(meth, "#cccccc"), edgecolor="white")
            ax.errorbar(x, cov, yerr=[[cov - lo], [hi - cov]], fmt="none",
                        ecolor="black", elinewidth=1.2, capsize=3)
            ax.text(x, 0.515, "%.3f" % cov, ha="center", va="bottom",
                    fontsize=7, rotation=90)
    ax.axhline(nom, color="black", ls="--", lw=1)
    ax.text(len(xpos) - 0.5, nom + 0.005, "nominal 0.90", fontsize=8, ha="right")
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("Coverage")
    tick_x = [t[0] for t in xticks]
    tick_l = [t[1] for t in xticks]
    ax.set_xticks(tick_x)
    ax.set_xticklabels(tick_l, rotation=45, ha="right", fontsize=7)
    # cohort separators and labels
    for ci, cname in enumerate(cohorts):
        ax.text(ci * (len(methods) + 1) + 2, 1.005, cname, ha="center",
                fontsize=9, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(root / "cpci_three_tier.pdf")
    fig.savefig(root / "cpci_three_tier.png", dpi=300)
    print("figure saved: %s" % (root / "cpci_three_tier.pdf"))

    # ---------- LaTeX table rows ----------
    print("\n% ---- paste into tab:mri ----")
    for cname in cohorts:
        print(r"\midrule")
        first = True
        for meth in methods:
            g = per[(per["cohort"] == cname) & (per["method"] == meth)]
            if not len(g):
                continue
            k, n = int(g["covered"].sum()), int(g["n"].sum())
            lo, hi = wilson_ci(k, n)
            lab = (" %s" % cname) if first else ""
            first = False
            print(" %s & %-13s & %.3f [%.3f, %.3f] & %.2f \\\\" % (
                lab, meth, g["coverage"].mean(), lo, hi, g["width"].mean()))


if __name__ == "__main__":
    main()
