# -*- coding: utf-8 -*-
"""
make_figs_and_tables_v2.py -- revised version of make_figs_and_tables.py.

Reviewer-driven change (consistency with Table 'eff'): the old script drew
Wilson CIs on POOLED subjects across 20 splits, which double-counts the
same external subjects 20 times and contradicts the paper's own caveat
(and the subject-cluster bootstrap CIs used in Table 'eff'). This version
uses SUBJECT-CLUSTER bootstrap CIs (subject = cluster; per-subject coverage
frequency over seeds is the unit of variation), matching Table 'eff'.

Run AFTER `main`:
    python make_figs_and_tables_v2.py --result-root M:/MRI/Results_RDCCI_v7

Outputs:
    cpci_three_tier.pdf / .png  (Fig. tiers)
    deploy_curve.pdf            (Fig. deploy, if deploy_curve_*.csv exist)
    printed LaTeX rows for tab:balance / deploy numbers (to paste)
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def subject_bootstrap(df, cohort, method, n_boot=10000, seed=0):
    d = df[(df["cohort"] == cohort) & (df["method"] == method)]
    if not len(d):
        return None
    pv = d.groupby("sid").apply(
        lambda g: float(((g["true"] >= g["lo"]) & (g["true"] <= g["hi"])).mean()),
        include_groups=False).to_numpy()
    rng = np.random.RandomState(seed)
    boots = pv[rng.randint(0, len(pv), size=(n_boot, len(pv)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(pv.mean()), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-root", default="M:/MRI/Results_RDCCI_v7")
    args = ap.parse_args()
    root = Path(args.result_root)

    per = pd.read_csv(root / "three_tier_results_per_seed.csv")
    ints = pd.read_csv(root / "intervals_per_subject.csv")
    methods = ["CPCI-Sim", "CPCI-Oracle", "CPCI-Scaled", "Weighted-CP", "Oracle"]
    cohorts = ["PPMI-internal", "NEUROCON", "TaoWu"]
    nom = 0.90

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
            cov = g["coverage"].mean()
            r = subject_bootstrap(ints, cname, meth)
            lo, hi = (r[1], r[2]) if r else (cov, cov)
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
    ax.set_xticks([t[0] for t in xticks])
    ax.set_xticklabels([t[1] for t in xticks], rotation=45, ha="right", fontsize=7)
    for ci, cname in enumerate(cohorts):
        ax.text(ci * (len(methods) + 1) + 2, 1.005, cname, ha="center",
                fontsize=9, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(root / "cpci_three_tier.pdf")
    fig.savefig(root / "cpci_three_tier.png", dpi=300)
    print("figure saved: %s (subject-cluster bootstrap CIs)" % (root / "cpci_three_tier.pdf"))

    # caption note for the paper
    print("\n% ---- caption sentence for Fig. tiers ----")
    print("% Coverage is the mean over 20 splits; error bars are 95%% "
          "subject-cluster bootstrap CIs (subject = cluster), matching "
          "Table~\\ref{tab:eff}.")

    # ---- deploy curve (Fig. deploy) ----
    dep_files = sorted(root.glob("deploy_curve_*.csv"))
    if dep_files:
        fig, axes = plt.subplots(1, len(dep_files), figsize=(5.2 * len(dep_files), 3.8),
                                 sharey=True)
        if len(dep_files) == 1:
            axes = [axes]
        for ax, f in zip(axes, dep_files):
            d = pd.read_csv(f)
            for meth, mk in [("CPCI-Sim", "o"), ("RelInt", "s"),
                             ("CPCI-Oracle (no recal)", "^")]:
                g = d[d["method"] == meth].groupby("m")["coverage"].agg(["mean", "std"])
                if not len(g):
                    continue
                ax.errorbar(g.index, g["mean"], yerr=g["std"], marker=mk,
                            lw=1.8, capsize=3, label=meth)
            ax.axhline(nom, color="black", ls="--", lw=1)
            ax.set_xlabel("target-site calibration subjects $m$")
            ax.set_title(f.stem.replace("deploy_curve_", ""))
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("Coverage")
        axes[0].set_ylim(0, 1.02)
        axes[-1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(root / "deploy_curve.pdf")
        fig.savefig(root / "deploy_curve.png", dpi=300)
        print("deploy curve saved: %s" % (root / "deploy_curve.pdf"))
    else:
        print("deploy_curve_*.csv not found - run "
              "rd_cci_v8_cpci.py deploy --target <NEUROCON|TaoWu> first")

    # ---- balanced sensitivity numbers (tab:balance) ----
    bf = root / "balance_internal.csv"
    if bf.exists():
        b = pd.read_csv(bf)
        print("\n% ---- paste into tab:balance ----")
        print("balanced ($n=42$) & $%.3f\\pm%.3f$ & %.2f & --- \\\\" % (
            b["coverage_balanced"].mean(), b["coverage_balanced"].std(),
            b["width_balanced"].mean()))
        print("\n% ---- paste into Sec. 4.2 balanced-sensitivity sentence ----")
        print("%% leaves the internal coverage of CPCI-Sim at $%.3f\\pm%.3f$"
              % (b["coverage_balanced"].mean(), b["coverage_balanced"].std()))
    else:
        print("balance_internal.csv not found - run "
              "rd_cci_v8_cpci.py balance --n-target 42 --repeats 20 first")


if __name__ == "__main__":
    main()