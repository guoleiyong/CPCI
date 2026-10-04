# -*- coding: utf-8 -*-
"""
make_revision_figs.py -- analyses and figures for the CMIG revision:
  1) subject-cluster bootstrap CIs for external cohorts (fixes the
     repeated-subject Wilson CI issue)
  2) interval score (Winkler) per cohort/method
  3) coverage-width frontier under global width scaling (frontier.pdf)
  4) F_sim vs F_target score CDF comparison (cdf_comparison.pdf)
  5) direction-mismatch heatmap (stress2_heatmap.pdf)
  6) top AAL regions driving the effect direction (effect_regions.pdf)

Run AFTER: main (v7.4), stress2, cdf.
    python make_revision_figs.py --result-root M:/MRI/Results_RDCCI_v7
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ALPHA = 0.1
EXT = ["NEUROCON", "TaoWu"]
CORE = ["CPCI-Sim", "CPCI-Sim (composite)", "RelInt", "CPCI-Oracle",
        "CPCI-Scaled", "Weighted-CP", "Oracle", "MeanEffect", "Direct-tau"]
COLORS_EXTRA = {"CPCI-Sim (composite)": "#92c5de"}


def winkler(lo, hi, y, alpha=ALPHA):
    width = hi - lo
    below = (y < lo) * (2.0 / alpha) * (lo - y)
    above = (y > hi) * (2.0 / alpha) * (y - hi)
    return float(np.mean(width + below + above))


def subject_bootstrap(df, cohort, method, n_boot=10000, seed=0):
    """Cluster bootstrap with SUBJECT as the sampling unit: per-subject
    coverage frequency over seeds is the unit of variation."""
    d = df[(df["cohort"] == cohort) & (df["method"] == method)]
    if not len(d):
        return None
    pv = d.groupby("sid")["true"].count().rename("n").to_frame()
    pv["cov"] = d.groupby("sid").apply(
        lambda g: float(((g["true"] >= g["lo"]) & (g["true"] <= g["hi"])).mean()),
        include_groups=False)
    p = pv["cov"].to_numpy()
    rng = np.random.RandomState(seed)
    boots = p[rng.randint(0, len(p), size=(n_boot, len(p)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(p.mean()), float(lo), float(hi), len(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-root", default="M:/MRI/Results_RDCCI_v7")
    args = ap.parse_args()
    root = Path(args.result_root)
    int_df = pd.read_csv(root / "intervals_per_subject.csv")

    # ---------- 1+2: subject-bootstrap CIs + interval score ----------
    print("\n% ---- subject-cluster bootstrap CIs (external cohorts) ----")
    print("% method & cohort & coverage & [95% subject-cluster CI] & n_subj & Winkler score")
    rows_out = []
    for meth in CORE:
        for cname in EXT:
            r = subject_bootstrap(int_df, cname, meth)
            if r is None:
                continue
            cov, lo, hi, nsubj = r
            d = int_df[(int_df["cohort"] == cname) & (int_df["method"] == meth)]
            ws = winkler(d["lo"].to_numpy(), d["hi"].to_numpy(), d["true"].to_numpy())
            rows_out.append(dict(method=meth, cohort=cname, coverage=cov,
                                 ci_lo=lo, ci_hi=hi, n_subjects=nsubj,
                                 winkler=ws, width=float((d["hi"] - d["lo"]).mean())))
            print("%s & %s & %.3f & [%.3f, %.3f] & %d & %.3f \\\\" % (
                meth, cname, cov, lo, hi, nsubj, ws))
    pd.DataFrame(rows_out).to_csv(root / "revision_stats.csv", index=False)

    # ---------- split-level percentile CIs (replaces pooled Wilson in main table) ----------
    per = pd.read_csv(root / "three_tier_results_per_seed.csv")
    print("\n% ---- split-level stats for MAIN TABLE (20 splits: mean, SD, 2.5/97.5 pct) ----")
    for cname in ["PPMI-internal"] + EXT:
        for meth in ["CPCI-Sim", "CPCI-Sim (composite)", "RelInt", "CPCI-Oracle",
                     "CPCI-Scaled", "MeanEffect", "Direct-tau", "Weighted-CP",
                     "Oracle"]:
            g = per[(per["cohort"] == cname) & (per["method"] == meth)]
            if not len(g):
                continue
            c = g["coverage"].to_numpy()
            lo, hi = np.percentile(c, [2.5, 97.5])
            print("%s & %s & %.3f & %.3f & [%.3f, %.3f] & %.2f \\" % (
                cname, meth, c.mean(), c.std(), lo, hi, g["width"].mean()))

    # seed-level variability (model/split uncertainty)
    per = pd.read_csv(root / "three_tier_results_per_seed.csv")
    print("\n% ---- seed-level variability (mean+-SD over 20 splits) ----")
    for meth in ["CPCI-Sim", "RelInt", "CPCI-Oracle"]:
        for cname in EXT:
            g = per[(per["cohort"] == cname) & (per["method"] == meth)]
            if len(g):
                print("%s %s: %.3f +- %.3f (min %.3f, max %.3f)" % (
                    meth, cname, g["coverage"].mean(), g["coverage"].std(),
                    g["coverage"].min(), g["coverage"].max()))

    # ---------- 2b: clinical interpretability (zero-inclusion + width/effect) ----------
    print("\n% ---- clinical interpretability (evaluation truth) ----")
    print("% method & cohort & %% intervals containing zero & median width / median |true| & median width")
    zi_rows = []
    for meth in ["CPCI-Sim", "RelInt", "CPCI-Oracle", "Target-GT Oracle"]:
        mm = "Oracle" if meth == "Target-GT Oracle" else meth
        for cname in ["PPMI-internal"] + EXT:
            d = int_df[(int_df["cohort"] == cname) & (int_df["method"] == mm)]
            if not len(d):
                continue
            zero_frac = float((d["lo"] <= 0.0).mean())
            wid = (d["hi"] - d["lo"]).to_numpy()
            mt = np.abs(d["true"]).to_numpy()
            ratio = float(np.median(wid) / max(1e-8, np.median(mt)))
            zi_rows.append(dict(method=meth, cohort=cname, zero_frac=zero_frac,
                                width_over_effect=ratio, median_width=float(np.median(wid))))
            print("%s & %s & %.1f\\%% & %.2f & %.2f \\" % (
                meth, cname, 100 * zero_frac, ratio, np.median(wid)))
    pd.DataFrame(zi_rows).to_csv(root / "zero_inclusion.csv", index=False)

    # ---------- 3: coverage-width frontier (global width scaling) ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    colors = {"CPCI-Sim": "#2166ac", "CPCI-Sim (composite)": "#92c5de",
              "RelInt": "#762a83", "CPCI-Oracle": "#999999",
              "CPCI-Scaled": "#b2182b", "Weighted-CP": "#66c2a5",
              "Oracle": "#1a9850", "MeanEffect": "#f4a582", "Direct-tau": "#fed9a6"}
    ks = np.linspace(0.2, 3.0, 60)
    for ax, cname in zip(axes, EXT):
        for meth in CORE:
            d = int_df[(int_df["cohort"] == cname) & (int_df["method"] == meth)]
            if not len(d):
                continue
            half = (d["hi"].to_numpy() - d["lo"].to_numpy()) / 2.0
            mid = (d["hi"].to_numpy() + d["lo"].to_numpy()) / 2.0
            y = d["true"].to_numpy()
            covs, wids = [], []
            for k in ks:
                lo_k = np.maximum(0.0, mid - k * half)
                hi_k = mid + k * half
                covs.append(float(((lo_k <= y) & (y <= hi_k)).mean()))
                wids.append(float((hi_k - lo_k).mean()))
            ax.plot(wids, covs, color=colors.get(meth, "#ccc"), lw=1.8,
                    label=meth, ls="--" if meth in ("Weighted-CP", "Oracle") else "-")
        ax.axhline(0.9, color="black", ls=":", lw=1)
        ax.set_xlabel("Mean interval width")
        ax.set_title(cname)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Coverage")
    axes[0].set_ylim(0, 1.02)
    axes[1].legend(fontsize=7)
    fig.suptitle("Coverage-width frontier under global width scaling (20 splits pooled)")
    fig.tight_layout()
    fig.savefig(root / "frontier.pdf"); fig.savefig(root / "frontier.png", dpi=300)
    print("\n% ---- width required to reach nominal 90% coverage ----")
    for cname in EXT:
        for meth in CORE:
            d = int_df[(int_df["cohort"] == cname) & (int_df["method"] == meth)]
            if not len(d):
                continue
            half = (d["hi"].to_numpy() - d["lo"].to_numpy()) / 2.0
            mid = (d["hi"].to_numpy() + d["lo"].to_numpy()) / 2.0
            y = d["true"].to_numpy()
            w_at_90, cov_max = None, 0.0
            for k in np.linspace(0.2, 6.0, 240):
                lo_k = np.maximum(0.0, mid - k * half)
                hi_k = mid + k * half
                cv = float(((lo_k <= y) & (y <= hi_k)).mean())
                cov_max = max(cov_max, cv)
                if cv >= 0.90:
                    w_at_90 = float((hi_k - lo_k).mean())
                    break
            print("%s %s: width@90%% = %s (max coverage %.3f)" % (
                cname, meth,
                ("%.2f" % w_at_90) if w_at_90 else ">max-tested",
                cov_max))
    print("\nfrontier.pdf saved")

    # ---------- 4: score CDF comparison ----------
    try:
        cdf = pd.read_csv(root / "score_cdfs.csv")
        kinds = ["source-GT", "simulated", "target-GT (NEUROCON)", "target-GT (TaoWu)"]
        cols = {"source-GT": "#999999", "simulated": "#2166ac",
                "target-GT (NEUROCON)": "#1a9850", "target-GT (TaoWu)": "#66c2a5"}
        fig, ax = plt.subplots(figsize=(5.6, 4.4))
        # normalize per seed by the median SIMULATED error so the x-axis
        # "ITE error / simulated error" is unit-free and curves comparable
        for k in kinds:
            vv = []
            for sd0, g0 in cdf[cdf["kind"] == k].groupby("seed"):
                ref = cdf[(cdf["kind"] == "simulated") & (cdf["seed"] == sd0)]
                ref = float(np.median(ref["value"])) if len(ref) else 1.0
                vv.append((g0["value"].to_numpy() / max(ref, 1e-12)))
            v = np.concatenate(vv) if vv else np.array([])
            if not len(v):
                continue
            xs = np.sort(v)
            ys = np.arange(1, len(xs) + 1) / len(xs)
            ax.plot(xs, ys, color=cols[k], lw=2 if k == "simulated" else 1.5,
                    label=k, ls="-" if k in ("simulated",) else "--")
        ax.set_xlabel("ITE error / simulated error")
        ax.set_ylabel("CDF")
        ax.set_title("Simulated calibration errors wrap the target-GT tail")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(root / "cdf_comparison.pdf"); fig.savefig(root / "cdf_comparison.png", dpi=300)
        print("cdf_comparison.pdf saved")
    except FileNotFoundError:
        print("score_cdfs.csv missing - run `python rd_cci_v7_cpci.py cdf`")

    # ---------- 5: direction-mismatch heatmap ----------
    try:
        s2 = pd.read_csv(root / "stress2_results.csv")
        ext_mask = s2["cohort"].isin(EXT)
        piv = s2[ext_mask].groupby(["theta", "method"])["coverage"].mean().reset_index()
        thetas = sorted(piv["theta"].unique())
        meths = ["CPCI-Sim", "RelInt", "CPCI-Oracle"]
        M = np.array([[piv[(piv["theta"] == t) & (piv["method"] == m)]["coverage"].iloc[0]
                       for t in thetas] for m in meths])
        fig, ax = plt.subplots(figsize=(7.2, 2.6))
        im = ax.imshow(M, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(thetas)))
        ax.set_xticklabels(["%d" % t for t in thetas])
        ax.set_xlabel("Effect-direction mismatch (degrees)")
        ax.set_yticks(range(len(meths))); ax.set_yticklabels(meths)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, "%.2f" % M[i, j], ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="Coverage (external, pooled)")
        ax.set_title("Coverage vs effect-direction mismatch (severity matched)")
        fig.tight_layout()
        fig.savefig(root / "stress2_heatmap.pdf"); fig.savefig(root / "stress2_heatmap.png", dpi=300)
        print("stress2_heatmap.pdf saved")
    except FileNotFoundError:
        print("stress2_results.csv missing - run `python rd_cci_v7_cpci.py stress2`")

    # ---------- 6: top AAL regions ----------
    try:
        eff = pd.read_csv(root / "effect_direction.csv")
        det = pd.read_csv(root / "feature_filter_detail.csv")
        if not len(det):
            raise ValueError(
                "feature_filter_detail.csv is EMPTY (a failed run wiped "
                "it). Re-run `main` with valid features to regenerate it, "
                "then re-run this script.")
        kept = det[det["keep"]].reset_index(drop=True)
        eff = eff[eff["j"] < len(kept)]  # guard against stale index files
        eff["region"] = eff["j"].map(lambda j: str(kept.iloc[int(j)]["feature"]))
        eff["region"] = eff["region"].str.replace("__std", "", regex=False)
        agg = eff.groupby("region")["d"].apply(lambda s: float(np.mean(np.abs(s)))).sort_values()
        top = agg.tail(15)
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        ax.barh(range(len(top)), top.to_numpy(), color="#2166ac")
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index, fontsize=8)
        ax.set_xlabel("Mean |effect direction| over 20 splits (standardized units)")
        ax.set_title("Top AAL regions driving the counterfactual effect")
        fig.tight_layout()
        fig.savefig(root / "effect_regions.pdf"); fig.savefig(root / "effect_regions.png", dpi=300)
        print("effect_regions.pdf saved")
    except FileNotFoundError:
        print("effect_direction.csv missing - rerun `main` (v7.4)")


if __name__ == "__main__":
    main()