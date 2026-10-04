# -*- coding: utf-8 -*-
"""make_table2_ppmi.py -- fill the PPMI column of Table 2 (tab:acq) with
demographics computed from the ANALYZED 754-subject sample.

Joins the feature CSV's file list (subject IDs) with ppmi_subjects.csv
(diagnosis, manufacturer) and the IDA search export (age, sex), then
prints LaTeX-ready values for the Age and Female rows.

Usage:
  python make_table2_ppmi.py --feat-csv M:/MRI/Results_RDCCI_v7/features/PPMI_roi_features.csv \
      --subjects ppmi_subjects.csv --ida idaSearch_9_21_2026.xlsx
"""
import argparse
import re
from pathlib import Path

import pandas as pd


def _find_file(name, hint_dirs):
    """Resolve a metadata file: use the path as given, else search the
    script directory and common data locations; fail with a clear list."""
    cand = [Path(name).resolve()] + [(Path(d) / name).resolve() for d in hint_dirs]
    seen, cand_u = set(), []
    for c in cand:
        if str(c) not in seen:
            seen.add(str(c))
            cand_u.append(c)
    for c in cand_u:
        if c.exists():
            return c
    raise FileNotFoundError(
        "cannot find '%s'.\nLooked in:\n  %s\n"
        "Pass the full path explicitly, e.g.\n"
        "  --subjects C:/path/to/ppmi_subjects.csv --ida C:/path/to/idaSearch.xlsx"
        % (name, "\n  ".join(str(c) for c in cand_u)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-csv", required=True, help="PPMI_roi_features.csv (analysis subset)")
    ap.add_argument("--subjects", default="ppmi_subjects.csv")
    ap.add_argument("--ida", default="idaSearch_9_21_2026.xlsx")
    args = ap.parse_args()

    files = pd.read_csv(args.feat_csv, usecols=["file"])["file"].astype(str)
    patnos_all = sorted(set(re.search(r"(\d+)", f).group(1) for f in files
                            if re.search(r"(\d+)", f)))
    print("imaged subjects (from features): %d" % len(patnos_all))

    hint_dirs = [".", str(Path(__file__).resolve().parent),
                 "M:/MRI/PPMI", "M:/MRI", "M:/MRI/Results_RDCCI",
                 str(Path.home() / "Downloads")]
    args.subjects = str(_find_file(args.subjects, hint_dirs))
    args.ida = str(_find_file(args.ida, hint_dirs))
    print("using:", args.subjects)
    print("using:", args.ida)
    subj_raw = pd.read_csv(args.subjects, encoding="utf-8-sig")
    subj_raw["PATNO"] = subj_raw["PATNO"].astype(str)
    # replicate the pipeline label filter: keep PD / Control only
    labeled = subj_raw[subj_raw["Group"].isin(["PD", "Control"])]
    patnos = [pt for pt in patnos_all if pt in set(labeled["PATNO"])]
    print("analyzed subjects (label-matched, cf. Table~\ref{tab:cohorts}): %d "
          "(%d imaged subjects dropped as label-unmatched)"
          % (len(patnos), len(patnos_all) - len(patnos)))

    subj = labeled.set_index("PATNO").reindex(patnos)
    grp = subj["Group"]
    print("diagnosis match: PD=%d, HC/Control=%d, missing=%d"
          % ((grp == "PD").sum(), grp.isin(["Control", "HC"]).sum(), grp.isna().sum()))

    ida = pd.read_excel(args.ida, skiprows=1)
    ida.columns = ["Subject ID", "Sex", "Age", "Description", "Type", "Manufacturer"]
    ida["Subject ID"] = ida["Subject ID"].astype(str)
    first = ida.drop_duplicates("Subject ID").set_index("Subject ID")
    meta = first.reindex(patnos)
    age = meta["Age"].astype(float)
    female = (meta["Sex"] == "F")

    print("\n%% ---- paste into tab:acq (PPMI column) ----")
    print("Age (mean$\\pm$SD) & %.1f $\\pm$ %.1f &  &  \\\\" % (age.mean(), age.std()))
    print("Female (\\%%) & %.1f &  &  \\\\" % (100 * female.mean()))
    man = subj["Manufacturer"].value_counts()
    print("\n%% manufacturer check (PPMI analyzed subset):")
    for k, v in man.items():
        print("%%   %-12s %4d (%.1f%%)" % (k, v, 100 * v / man.sum()))


if __name__ == "__main__":
    main()