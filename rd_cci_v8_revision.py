# -*- coding: utf-8 -*-
"""
RD-CCI MRI experiments v7.6 (revision) -- CPCI three-tier calibration protocol
=================================================================
Data layout (default M:/MRI, override with --data-root):
  <root>/PPMI/     PPMI_*_T1w_MNI152_normalized.nii.gz + ppmi_subjects.csv
  <root>/NEUROCON/ sub-*/sub-*_T1w_MNI152_normalized.nii.gz + neurocon_patients.tsv
  <root>/TaoWu/    sub-*/sub-*_T1w_MNI152_normalized.nii.gz + taowu_patients.tsv
  <root>/aal_extracted/aal/atlas/AAL.nii + AAL.xml

v7.1: added CPCI-Sim (simulation-calibrated deployable proxy) after the
  3-seed trial showed raw proxy quantiles are ~20x off in scale (cov=1.0,
  width~17) and training-set rho is anti-conservative under shift.
v7.3: split-conformal CPCI-Sim (subjects = exchangeability units;
  g fitted on one disjoint half, subject-level conformal quantile on the
  other); analytic RelInt baseline (half-width = (1-alpha) quantile of
  |1-kappa| x Delta_hat); score-component ablations (CPCI-Sim-delta,
  CPCI-Sim-rc) + Spearman diagnostics; new `stress` command evaluating
  simulation-family misspecification (wide/shifted severity).
v7.6 (revision): two review-driven commands integrated; see cmd_deploy / cmd_balance below:
  `deploy`  - Scenario-B onboarding curve: re-fit the CPCI-Sim calibration layer
              on m target-site subjects (observables only; m<12 falls back to
              fitting g on all m subjects, flagged as pseudo-replication) and
              evaluate on the remaining target subjects; RelInt and the
              deployed-without-recalibration oracle are horizontal references.
              Feeds Fig. deploy (deployment information budget).
  `balance` - balanced sensitivity: internal coverage of CPCI-Sim with the
              internal test fold subsampled to --n-target subjects, --repeats
              times per seed; no model retraining. Feeds Appendix C (tab:balance)
              and the balanced-sensitivity sentence in Sec. 4.2.
  Companion: make_figs_and_tables_v2.py redraws Fig. tiers with subject-cluster
  bootstrap CIs (matching tab:eff; the old pooled-Wilson version double-counts
  external subjects across the 20 splits).
v7.2: fixed CPCI-Sim pairing -- simulated errors are paired with ite_hat on
  the FACTUAL sample (matching the evaluation protocol), not on the shifted
  counterfactual; removes the off-manifold double penalty that made
  intervals ~3x too wide (20-seed trial: width 2.4-3.1 vs oracle 0.9).

v7 changes (paper alignment, three-tier protocol):
  1. Headline DEPLOYABLE method = CPCI-Proxy: conformal calibration on
     OBSERVABLE proxy scores only (no ground-truth ITE at calibration):
         S(X) = r_cycle(X) + lam * delta_effect(X)
         r_cycle      = ||X - recon||_2   (generator fidelity, observable)
         delta_effect = ||x1 - x0||_2     (effect discrepancy; with the
                                           linear/RBF-feature map this is
                                           the RKHS term of the paper)
     This matches the paper's composite nonconformity score (default lam=0.5).
  2. Three-tier evaluation protocol:
       Tier 1  CPCI-Oracle : S = |true_ITE - ite_hat|  (needs injected truth;
                             reported as coverage UPPER BOUND / reference only)
       Tier 2  CPCI-Proxy  : S = r_cycle + lam*delta_effect (deployable)
       Tier 3  CPCI-Scaled : q = rho * q_proxy, rho estimated on the TRAINING
                             injection (honest quantification of the
                             deployable-proxy gap)
  3. Lambda sensitivity grid (0, 0.25, 0.5, 1.0) per seed/cohort
     -> feeds the paper hyperparameter-sensitivity table.
  4. rate sweep now also records the PROXY coverage gap under shift
     (gap_proxy vs rate_kl, with bootstrap CI on the log-log slope)
     -> empirical "rate vs deployable-coverage-gap" law for Sec. 4.2.
  5. Every output file is annotated with the paper table/figure it feeds.

Usage:
  python rd_cci_v8_revision.py extract     # one-time ROI feature extraction -> CSV
  python rd_cci_v8_revision.py build       # check labels / cohort sizes
  python rd_cci_v8_revision.py main        # Exp 1: three-tier internal+external
  python rd_cci_v8_revision.py rate        # Exp 2: lam_rd/latent sweep + proxy gap
  python rd_cci_v8_revision.py all
  python rd_cci_v8_revision.py deploy --target NEUROCON --m-list 10 20 40 80 --repeats 5   # v7.6: new-site onboarding curve (Fig. deploy)
  python rd_cci_v8_revision.py balance --n-target 42 --repeats 20                          # v7.6: balanced sensitivity (Appendix C)

Options:  --data-root  --out-root  --seeds 42 43 ...  --lam-proxy 0.5
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DEFAULT_CONFIG = {
    "data_root": "M:/MRI",
    "out_root": "M:/MRI/Results_RDCCI_v7",
    "alpha": 0.1,
    "lambda_rd": 0.01,
    "latent_dim": 32,
    "gamma_cf": 0.5,
    "hidden_dim": 128,
    "lr": 2e-3,
    "epochs": 500,
    "patience": 60,
    "batch_size": 64,
    "seeds": [42, 43, 44, 45, 46, 47, 48, 49, 50,
              51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61],
    "val_frac": 0.15,
    "test_frac": 0.25,
    "semi_scale_range": (0.5, 1.5),
    "semi_noise": 0.05,
    "gamma_sup": 5.0,
    "gamma_het": 1.0,
    # --- v7: CPCI proxy-score configuration ---
    "lambda_proxy": 0.5,                 # paper composite score weight
    "proxy_lam_grid": (0.0, 0.25, 0.5, 1.0),   # sensitivity grid
    "feat_root": None,    # feature CSV dir override; default = out_root (see --feat-root)
    "n_sim": 16,          # simulated counterfactuals per calibration subject (CPCI-Sim)
}

def _feat_dir(cfg):
    """Directory holding <cohort>_roi_features.csv. Defaults to out_root/features,
    override with --feat-root to reuse features extracted by an older version
    (e.g. M:/MRI/Results_RDCCI) without re-running extract."""
    return Path(cfg.get("feat_root") or cfg["out_root"]) / "features"

PPMI_GROUP_MAP = {"pd": 1, "parkinson": 1, "healthy": 0, "control": 0, "hc": 0}


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _digits(s):
    import re
    m = re.search(r"(\d+)", str(s))
    return m.group(1) if m else ""


def grouped_train_cal_test(X, y, groups, test_frac, cal_frac_of_train, seed):
    from sklearn.model_selection import GroupShuffleSplit
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    tr_idx, te_idx = next(gss1.split(X, y, groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=cal_frac_of_train, random_state=seed)
    sub_tr, sub_cal = next(gss2.split(X[tr_idx], y[tr_idx], groups[tr_idx]))
    return tr_idx[sub_tr], tr_idx[sub_cal], te_idx


# ==================== AAL atlas parsing & feature extraction ====================

def parse_aal_xml(xml_path):
    import xml.etree.ElementTree as ET
    tree = ET.parse(str(xml_path))
    pairs, any_index = [], False
    for label in tree.iter("label"):
        name = (label.text or "").strip()
        idx = label.get("index")
        if idx is None:
            child = label.find("index")
            idx = child.text if child is not None else None
        if name == "":
            nm = label.find("name")
            name = (nm.text or "").strip() if nm is not None else ""
        if idx is not None:
            try:
                pairs.append((int(idx), name))
                any_index = True
                continue
            except (TypeError, ValueError):
                pass
        try:
            pairs.append((int(name), name))
            any_index = True
        except (TypeError, ValueError):
            continue
    if not any_index:
        raw = [(e.text or "").strip() for e in tree.iter("label")]
        pairs = [(i + 1, n) for i, n in enumerate(raw) if n]
    pairs.sort()
    return pairs


def build_label_map(atlas_img_path, xml_path):
    import nibabel as nib
    atlas = nib.load(str(atlas_img_path))
    adata = atlas.get_fdata().astype(np.int32)
    vals = sorted(v for v in np.unique(adata) if v > 0)
    pairs = parse_aal_xml(xml_path)
    if not pairs:
        raise ValueError("no regions parsed from AAL.xml")
    print(f"[atlas] xml regions: {len(pairs)}, index range {pairs[0][0]}..{pairs[-1][0]}")
    names = [n for _, n in pairs]
    if len(vals) == len(names):
        label_map = {v: names[i] for i, v in enumerate(vals)}
    else:
        idx2name = dict(pairs)
        label_map = {v: idx2name.get(v, "ROI%d" % v) for v in vals}
    return atlas, label_map


def extract_roi_features(img, adata, label_map, use_std=True):
    data = img.get_fdata().astype(np.float32)
    if data.shape != adata.shape:
        raise ValueError("shape mismatch: img %s vs atlas %s" % (data.shape, adata.shape))
    feats, names = [], []
    for v, name in label_map.items():
        mask = adata == v
        if mask.sum() == 0:
            continue
        vals = data[mask]
        feats.append(vals.mean())
        names.append(name)
        if use_std:
            feats.append(vals.std())
            names.append(name + "__std")
    return np.array(feats, dtype=np.float32), names


def make_resampler(atlas_img):
    import nibabel as nib
    cache = {}

    def resample_to(img):
        sh = tuple(img.shape)
        if sh not in cache:
            try:
                from nibabel.processing import resample_to_img
                r = resample_to_img(atlas_img, img, interpolation="nearest")
                cache[sh] = r.get_fdata().astype(np.int32)
            except Exception:
                from scipy.ndimage import zoom
                f = [s / a for s, a in zip(img.shape, atlas_img.shape)]
                cache[sh] = zoom(atlas_img.get_fdata().astype(np.float32), f,
                                 order=0).round().astype(np.int32)
            vs = img.header.get_zooms()
            print("[atlas] resampled to %s (voxel ~%.2fmm)" % (sh, vs[0]))
        return cache[sh]
    return resample_to


def cmd_extract(cfg):
    from tqdm import tqdm
    import nibabel as nib
    root = Path(cfg["data_root"])
    out = Path(cfg["out_root"]) / "features"
    out.mkdir(parents=True, exist_ok=True)
    atlas, label_map = build_label_map(
        root / "aal_extracted" / "aal" / "atlas" / "AAL.nii",
        root / "aal_extracted" / "aal" / "atlas" / "AAL.xml")
    print("[extract] atlas labels: %d" % len(label_map))
    resample = make_resampler(atlas)
    cohorts = {"PPMI": root / "PPMI", "NEUROCON": root / "NEUROCON", "TaoWu": root / "TaoWu"}
    for cname, cdir in cohorts.items():
        files = sorted(cdir.glob("*_T1w_MNI152_normalized.nii.gz"))
        if not files:
            print("[extract] %s: no images in %s" % (cname, cdir))
            continue
        rows, names_ref, bad = [], None, []
        for f in tqdm(files, desc=cname):
            try:
                img = nib.load(str(f))
                feats, names = extract_roi_features(img, resample(img), label_map)
                if names_ref is None:
                    names_ref = names
                elif names != names_ref:
                    raise ValueError("ROI name mismatch")
                rows.append([f.name] + feats.tolist())
            except Exception as e:
                bad.append((f.name, str(e)[:80]))
        if not rows:
            print("[extract] %s: ALL %d files FAILED - CSV NOT written; "
                  "fix the error above and re-run." % (cname, len(files)))
            continue
        df = pd.DataFrame(rows, columns=["file"] + (names_ref or []))
        df.to_csv(out / ("%s_roi_features.csv" % cname), index=False)
        print("[extract] %s: %d ok, %d failed -> %s" % (cname, len(df), len(bad), out))
        for b in bad[:5]:
            print("   failed:", b)


# ==================== cohort labels & loading ====================

_FEATURE_MASK_CACHE = {}
_LAST_UNMATCHED = []


def _expected_feature_count(cfg):
    df = pd.read_csv(_feat_dir(cfg) / "PPMI_roi_features.csv", nrows=1)
    return len([c for c in df.columns if c != "file"])


def get_feature_mask(cfg):
    key = cfg["out_root"]
    if key in _FEATURE_MASK_CACHE:
        return _FEATURE_MASK_CACHE[key]
    cache_file = Path(cfg["out_root"]) / "feature_filter.json"
    keep = None
    if cache_file.exists():
        keep = np.array(json.loads(cache_file.read_text(encoding="utf-8"))["keep"], dtype=bool)
        if len(keep) != _expected_feature_count(cfg):
            print("[filter] cache mismatch (%d vs %d), recomputing"
                  % (len(keep), _expected_feature_count(cfg)))
            keep = None
    if keep is None:
        zero_max, cols_ref = None, None
        for cname in ["PPMI", "NEUROCON", "TaoWu"]:
            fcsv = _feat_dir(cfg) / ("%s_roi_features.csv" % cname)
            df = pd.read_csv(fcsv)
            if len(df) == 0:
                raise ValueError(
                    "%s has 0 rows - features missing/empty. Re-run "
                    "`extract`, or pass --feat-root pointing at a valid "
                    "features directory." % fcsv)
            cols = [c for c in df.columns if c != "file"]
            X = df[cols].to_numpy(dtype=np.float32)
            zf = (X == 0).mean(axis=0)
            zero_max = zf if zero_max is None else np.maximum(zero_max, zf)
            cols_ref = cols
        keep = zero_max <= 0.05
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(
            {"keep": keep.tolist(), "n_total": int(len(keep)), "n_keep": int(keep.sum())}),
            encoding="utf-8")
        pd.DataFrame({"feature": cols_ref, "keep": keep}).to_csv(
            Path(cfg["out_root"]) / "feature_filter_detail.csv", index=False)
        print("[filter] ROI features kept %d/%d (dropped %d with >5%% zeros in any cohort)"
              % (int(keep.sum()), len(keep), int((~keep).sum())))
    _FEATURE_MASK_CACHE[key] = keep
    return keep


def load_ppmi_labels(cfg):
    root = Path(cfg["data_root"])
    subj = pd.read_csv(root / "PPMI" / "ppmi_subjects.csv")
    id_col = next((c for c in subj.columns if c.strip().lower() in ("subject", "subject_id", "patno")), None)
    grp_col = next((c for c in subj.columns if "group" in c.lower() or "research" in c.lower()), None)
    if id_col is None or grp_col is None:
        raise ValueError("id/group column not found in ppmi_subjects.csv: %s" % list(subj.columns))
    lab, seen = {}, set()
    for _, r in subj.iterrows():
        g = str(r[grp_col]).lower()
        if "prodromal" in g or "swedd" in g:
            continue
        y = -1
        for k, v in PPMI_GROUP_MAP.items():
            if k in g:
                y = v
                break
        key = _digits(r[id_col])
        if y >= 0 and key and key not in seen:
            lab[key] = y
            seen.add(key)
    return lab


def _norm_prefix(name):
    n = str(name).lower()
    if "patient" in n or n.startswith("pd"):
        return 1
    if "control" in n or n.startswith("hc") or "normal" in n:
        return 0
    return -1


def load_cohort(cfg, cname):
    root = Path(cfg["data_root"])
    feat_csv = _feat_dir(cfg) / ("%s_roi_features.csv" % cname)
    df = pd.read_csv(feat_csv)
    files = df["file"].astype(str).tolist()
    feat_cols = [c for c in df.columns if c != "file"]
    X = df[feat_cols].to_numpy(dtype=np.float32)
    mask = get_feature_mask(cfg)
    X = X[:, mask]
    global _LAST_UNMATCHED
    if cname == "PPMI":
        lab = load_ppmi_labels(cfg)
        y = np.array([lab.get(_digits(f), -1) for f in files])
        _LAST_UNMATCHED = [f for f, yy in zip(files, y) if yy < 0]
    else:
        y = np.array([_norm_prefix(f) for f in files])
        tsv = root / cname / ("%s_patients.tsv" % cname.lower())
        if tsv.exists():
            meta = pd.read_csv(tsv, sep="\t")
            name_col = meta.columns[0]
            stat_col = next((c for c in meta.columns if c.lower() == "status"), None)
            hy_col = next((c for c in meta.columns
                           if c.lower().replace("&", "").replace("_", "") in ("hy", "hoehnyahr")), None)
            for i, f in enumerate(files):
                subj_id = f.replace("sub-", "").split("_T1w")[0]
                hit = meta[meta[name_col].astype(str).str.contains(subj_id, na=False)]
                if len(hit) == 0:
                    continue
                r = hit.iloc[0]
                if stat_col is not None:
                    sv = _norm_prefix(r[stat_col])
                    if sv >= 0:
                        y[i] = sv
                if hy_col is not None:
                    try:
                        if float(r[hy_col]) >= 1:
                            y[i] = max(y[i], 1)
                    except (TypeError, ValueError):
                        pass
    keep = y >= 0
    return X[keep], y[keep], np.array(files)[keep]


def cmd_build(cfg):
    for cname in ["PPMI", "NEUROCON", "TaoWu"]:
        try:
            feat_csv = _feat_dir(cfg) / ("%s_roi_features.csv" % cname)
            n_files = len(pd.read_csv(feat_csv)) if feat_csv.exists() else 0
            X, y, files = load_cohort(cfg, cname)
            unmatched = n_files - len(X)
            msg = "[build] %s: n=%d (PD=%d, HC=%d), dim=%d" % (
                cname, len(X), int(y.sum()), int((y == 0).sum()), X.shape[1])
            if unmatched:
                msg += ", %d files unmatched!" % unmatched
            print(msg)
            if cname == "PPMI" and _LAST_UNMATCHED:
                print("   unmatched:", _LAST_UNMATCHED[:5])
        except FileNotFoundError as e:
            print("[build] %s: features missing, run extract first (%s)" % (cname, e))


# ==================== Twin-VAE counterfactual generator ====================

import torch
import torch.nn as nn


class TwinVAE(nn.Module):
    def __init__(self, d_in, latent=32, hidden=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d_in + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.lsig = nn.Linear(latent, 1)
        self.dec = nn.Sequential(
            nn.Linear(latent + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, d_in))

    def encode(self, x, t):
        h = self.enc(torch.cat([x, t], 1))
        return self.mu(h), self.logvar(h)

    def decode(self, z, t):
        return self.dec(torch.cat([z, t], 1))

    def forward(self, x, t):
        mu, logvar = self.encode(x, t)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.decode(z, t)
        cf = self.decode(z, 1 - t)
        return recon, cf, mu, logvar


def train_group_classifier(X, y, seed=42, epochs=200):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed + 999)
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    yt = torch.tensor(y, dtype=torch.float32, device=dev).unsqueeze(1)
    clf = nn.Sequential(nn.Linear(X.shape[1], 64), nn.ReLU(), nn.Linear(64, 1)).to(dev)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        perm = np.random.permutation(len(Xt))
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            opt.zero_grad()
            loss = bce(clf(Xt[b]), yt[b])
            loss.backward()
            opt.step()
    clf.eval()
    for t in clf.parameters():
        t.requires_grad_(False)
    return clf


def train_twin_vae(X, y, cfg, latent=None, lam_rd=None, seed=42, verbose=False, ite_true=None):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    latent = latent or cfg["latent_dim"]
    lam_rd = lam_rd if lam_rd is not None else cfg["lambda_rd"]
    gamma_cf = cfg.get("gamma_cf", 0.5)
    gamma_sup = cfg.get("gamma_sup", 1.0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr = torch.tensor(X, dtype=torch.float32, device=dev)
    ytr = torch.tensor(y, dtype=torch.float32, device=dev).unsqueeze(1)
    It = torch.tensor(ite_true, dtype=torch.float32, device=dev) if ite_true is not None else None
    model = TwinVAE(X.shape[1], latent, cfg["hidden_dim"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    clf = train_group_classifier(X, y, seed=seed)
    n = len(Xtr)
    n_val = max(5, int(n * cfg["val_frac"]))
    idx = np.random.permutation(n)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    best, best_state, bad = np.inf, None, 0
    for ep in range(cfg["epochs"]):
        model.train()
        perm = np.random.permutation(tr_idx)
        for i in range(0, len(perm), cfg["batch_size"]):
            b = perm[i:i + cfg["batch_size"]]
            opt.zero_grad()
            recon, cf, mu, logvar = model(Xtr[b], ytr[b])
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            cf_loss = bce(clf(cf), 1 - ytr[b])
            loss = mse(Xtr[b], model.decode(mu, ytr[b])) + lam_rd * kl + gamma_cf * cf_loss
            if It is not None:
                diff = model.decode(mu, torch.zeros_like(ytr[b])) - model.decode(mu, torch.ones_like(ytr[b]))
                ite_pred = diff.norm(dim=1)
                loss = loss + gamma_sup * mse(ite_pred, It[b])
                sig_b = torch.nn.functional.softplus(model.lsig(mu)) + 0.05
                loss = loss + cfg.get("gamma_het", 1.0) * torch.mean(
                    ((ite_pred - It[b]) / sig_b) ** 2 + 2.0 * torch.log(sig_b))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = tr_idx if ep == 0 else val_idx
            recon, _, mu, logvar = model(Xtr[v], ytr[v])
            vl = mse(Xtr[v], recon).item()
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: t.cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    if best_state is not None:
        model.load_state_dict({k: t.to(dev) for k, t in best_state.items()})
    model.eval()
    if verbose:
        print("    [vae] latent=%d lam_rd=%g epochs=%d val_recon=%.4f" % (latent, lam_rd, ep + 1, best))
    return model


@torch.no_grad()
def vae_outputs(model, X, y):
    dev = next(model.parameters()).device
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    yt = torch.tensor(y, dtype=torch.float32, device=dev).unsqueeze(1)
    mu, logvar = model.encode(Xt, yt)
    recon = model.decode(mu, yt).cpu().numpy()
    x0 = model.decode(mu, torch.zeros_like(yt)).cpu().numpy()
    x1 = model.decode(mu, torch.ones_like(yt)).cpu().numpy()
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)).cpu().numpy()
    sig = (torch.nn.functional.softplus(model.lsig(mu)) + 0.05).squeeze(1).cpu().numpy()
    return recon, x0, x1, kl, sig


# ==================== CPCI proxy-score construction (v7) ====================

def proxy_components(model, X, y):
    """Observable components of the composite counterfactual score.

    r_cycle(X)      = ||X - recon||_2   -- generator fidelity on the FACTUAL
                                         side (observable; plays the role of
                                         the paper's cycle-consistency term)
    delta_effect(X) = ||x1 - x0||_2     -- estimated effect magnitude (the
                                         paper's RKHS effect discrepancy;
                                         Euclidean == RKHS with linear
                                         feature map)
    """
    recon, x0, x1, kl, sig = vae_outputs(model, X, y)
    r_cycle = np.linalg.norm(X - recon, axis=1)
    delta_effect = np.linalg.norm(x1 - x0, axis=1)
    return r_cycle, delta_effect, sig


def proxy_quantiles(model, Xcal, ycal, Xtr, ytr, ite_true_tr, cfg, lam_grid):
    """Calibrate proxy-score quantiles for a grid of lambda (deployable:
    uses ONLY observable quantities on the calibration and training sets).
    Also returns the scaled-proxy quantiles q_sp = rho * q_proxy where rho is
    the error/proxy scale estimated on the TRAINING injection (optimistic but
    honest quantification of the deployable gap)."""
    lv = _quantile_ci_level(len(ycal), cfg["alpha"])
    r_cal, d_cal, _ = proxy_components(model, Xcal, ycal)
    r_tr, d_tr, _ = proxy_components(model, Xtr, ytr)
    err_tr = np.abs(ite_true_tr - d_tr)      # training injection truth
    q_lam, q_sp, rho_lam = {}, {}, {}
    for lam in lam_grid:
        S_cal = r_cal + lam * d_cal
        S_tr = r_tr + lam * d_tr
        q_lam[lam] = float(np.quantile(S_cal, lv))
        rho = float(np.quantile(err_tr, lv) / max(1e-8, np.quantile(S_tr, lv)))
        rho_lam[lam] = rho
        q_sp[lam] = rho * q_lam[lam]
    return q_lam, q_sp, rho_lam


def sim_proxy_calibration(model, Xcal, ycal, d_vec, cfg, seed, lam, n_sim=None,
                          S_mode="score", fit_frac=0.5, range_override=None):
    """CPCI-Sim (v7.3, split-conformal): deployable calibration of the
    proxy-score -> ITE-error map on SELF-GENERATED simulated effects.
    No real counterfactual truth is used.

    Theory fix for the "pseudo-replication" objection: SUBJECTS are the
    exchangeability units, and the calibration subjects are split into two
    DISJOINT halves. g is fitted only on half A's simulated pairs; on half B
    each subject contributes ONE score -- the within-subject (1-alpha)
    quantile of simulated residuals e - g(S) -- and the subject-level
    conformal quantile of these scores at level
    ceil((n_B+1)(1-alpha))/n_B yields q_sim. This restores the standard
    split-conformal argument: the score of each held-out subject is
    exchangeable with the test subject's, given g fixed from the other
    half. (v7.2 pairing of errors with ite_hat on the factual sample is
    retained.)

    S_mode: "score" -> S = r_cycle + lam*delta (default, paper method);
            "delta"  -> S = delta only;  "rc" -> S = r_cycle only
            (the two latter are score-component ablations).
    """
    from sklearn.isotonic import IsotonicRegression
    n_sim = n_sim or cfg.get("n_sim", 16)
    rng = np.random.RandomState(seed + 7777)
    n = len(ycal)
    lo_s, hi_s = (range_override if range_override is not None
                  else cfg["semi_scale_range"])
    r_cal, d_cal, _ = proxy_components(model, Xcal, ycal)
    if S_mode == "delta":
        S_cal = d_cal.copy()
    elif S_mode == "rc":
        S_cal = r_cal.copy()
    else:
        S_cal = r_cal + lam * d_cal
    idx = np.tile(np.arange(n), n_sim)
    scales = rng.uniform(lo_s, hi_s, size=len(idx))
    err = d_cal[idx] * np.abs(1.0 - scales)            # |d_cal - tau|
    perm = rng.permutation(n)
    n_fit = max(10, int(round(n * fit_frac)))
    i_fit, i_conf = perm[:n_fit], perm[n_fit:]
    mask_fit = np.isin(idx, i_fit)
    g = IsotonicRegression(out_of_bounds="clip").fit(S_cal[idx[mask_fit]],
                                                     err[mask_fit])
    lv_m = min(0.99, np.ceil((n_sim + 1) * (1 - cfg["alpha"])) / n_sim)
    lv = _quantile_ci_level(len(i_conf), cfg["alpha"])
    u = np.array([np.quantile(err[idx == i] - g.predict(S_cal[[i]])[0], lv_m)
                  for i in i_conf])
    q_sim = float(np.quantile(u, lv))
    return g, q_sim


# ==================== semi-synthetic ground truth ====================

_EFFECT_BETA = {}


def effect_beta(dim, seed=777):
    """Fixed random direction of effect modification: individual effects
    depend on baseline features x. Globally fixed -> shared across cohorts."""
    key = (dim, seed)
    if key not in _EFFECT_BETA:
        r = np.random.RandomState(seed)
        b = r.normal(0, 1, dim)
        _EFFECT_BETA[key] = b / (np.linalg.norm(b) + 1e-8)
    return _EFFECT_BETA[key]


def make_semisynthetic(X, y, d_vec, rng, cfg):
    beta = effect_beta(X.shape[1])
    s = 1.0 + 0.5 * np.tanh(X @ beta)
    delta = s[:, None] * d_vec[None, :]
    noise = rng.normal(0, cfg["semi_noise"], size=X.shape)
    sign = np.where(y == 1, -1.0, 1.0)[:, None]
    y_cf_true = X + sign * delta + noise
    true_ite = np.linalg.norm(delta, axis=1)
    return X, y, true_ite, y_cf_true


# ==================== conformal ====================

def conformal_intervals(Xtr, ytr, Xcal, ycal, true_cal, cfg, seed=42, model=None,
                        latent=None, lam_rd=None, ite_true_tr=None):
    if model is None:
        model = train_twin_vae(Xtr, ytr, cfg, latent=latent, lam_rd=lam_rd,
                               seed=seed, ite_true=ite_true_tr)
    recon, x0, x1, kl_cal, sig_cal = vae_outputs(model, Xcal, ycal)
    ite_hat_cal = np.linalg.norm(x1 - x0, axis=1)
    S = np.abs(true_cal - ite_hat_cal)
    n = len(S)
    q = np.quantile(S, min(0.99, np.ceil((n + 1) * (1 - cfg["alpha"])) / n))
    dist = float(np.mean((Xcal - recon) ** 2))
    return model, q, float(np.mean(kl_cal)), dist, ite_hat_cal


def evaluate_on(model, q, X, y, cfg):
    recon, x0, x1, _, sig = vae_outputs(model, X, y)
    ite_hat = np.linalg.norm(x1 - x0, axis=1)
    lo = np.maximum(0, ite_hat - q)
    hi = ite_hat + q
    return lo, hi, ite_hat, recon, sig


# ==================== Experiment 1 (three-tier protocol) ====================

def _interval_eval(ite_hat, q, true_ite, sigma=None):
    half = q * sigma if sigma is not None else q
    lo = np.maximum(0.0, ite_hat - half)
    hi = ite_hat + half
    m = (lo <= true_ite) & (true_ite <= hi)
    return float(m.mean()), float((hi - lo).mean()), float(np.abs(ite_hat - true_ite).mean()), int(m.sum())


def weighted_quantile(values, weights, q):
    idx = np.argsort(values)
    v, w = values[idx], weights[idx]
    cw = np.cumsum(w) - 0.5 * w
    cw = cw / w.sum()
    return float(np.interp(q, cw, v))


def _quantile_ci_level(n, alpha):
    return min(0.99, np.ceil((n + 1) * (1 - alpha)) / n)


def cmd_main(cfg):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPRegressor
    from scipy.stats import spearmanr
    Xp, yp, files_p = load_cohort(cfg, "PPMI")
    groups_p = np.array([_digits(f) for f in files_p])
    ext = {}
    for c in ["NEUROCON", "TaoWu"]:
        X, y, _ = load_cohort(cfg, c)
        ext[c] = (X, y)
    lam_grid = list(cfg.get("proxy_lam_grid", (0.0, 0.25, 0.5, 1.0)))
    lam_default = float(cfg["lambda_proxy"])
    rows, diag, int_rows, eff_rows = [], [], [], []
    INT_SAVE = {"CPCI-Sim", "RelInt", "CPCI-Sim-delta", "CPCI-Sim-rc",
                "CPCI-Oracle", "CPCI-Scaled", "MeanEffect", "Direct-tau"}
    t0 = time.time()
    for seed in cfg["seeds"]:
        rng = np.random.RandomState(seed)
        tr, cal, te = grouped_train_cal_test(Xp, yp, groups_p, cfg["test_frac"], 0.25, seed)
        Xtr, ytr, Xcal, ycal, Xte, yte = Xp[tr], yp[tr], Xp[cal], yp[cal], Xp[te], yp[te]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Z = lambda A: (A - mu) / sd
        Xtr_s, Xcal_s, Xte_s = Z(Xtr), Z(Xcal), Z(Xte)
        d_vec = Xtr_s[ytr == 1].mean(0) - Xtr_s[ytr == 0].mean(0)
        _, _, ite_true_tr, _ = make_semisynthetic(Xtr_s, ytr, d_vec, rng, cfg)
        _, _, ite_true_cal, _ = make_semisynthetic(Xcal_s, ycal, d_vec, rng, cfg)
        eff_rows.extend(dict(seed=seed, j=j, d=float(v)) for j, v in enumerate(d_vec))
        model, q, rate_kl, dist, ite_hat_cal = conformal_intervals(
            Xtr_s, ytr, Xcal_s, ycal, ite_true_cal, cfg, seed=seed, ite_true_tr=ite_true_tr)

        # ---- Tier 1 oracle (needs injected truth; coverage upper bound) ----
        recon_c, x0_c, x1_c, _, sig_cal = vae_outputs(model, Xcal_s, ycal)
        ite_hat_cal = np.linalg.norm(x1_c - x0_c, axis=1)
        lv = _quantile_ci_level(len(ite_true_cal), cfg["alpha"])
        S_h = np.abs(ite_true_cal - ite_hat_cal) / sig_cal
        q_h = np.quantile(S_h, lv)

        # ---- Tier 2/3 proxy calibration (deployable, no ground truth) ----
        q_lam, q_sp, rho_lam = proxy_quantiles(
            model, Xcal_s, ycal, Xtr_s, ytr, ite_true_tr, cfg, lam_grid)

        # ---- Tier 2b: simulation-calibrated deployable proxy (CPCI-Sim) ----
        sim_models = {}
        for lam in lam_grid:
            g_l, q_sim_l = sim_proxy_calibration(
                model, Xcal_s, ycal, d_vec, cfg, seed, lam)
            sim_models[lam] = (g_l, q_sim_l)

        # v7.5: headline CPCI-Sim uses the FIDELITY score (r_cycle only);
        # the composite score is retained as an ablation (reviewer-driven
        # simplification: calibration determines reliability, difficulty
        # modeling affects adaptivity).
        g_rc, q_rc = sim_proxy_calibration(
            model, Xcal_s, ycal, d_vec, cfg, seed, lam_default, S_mode="rc")
        g_delta, q_delta = sim_proxy_calibration(
            model, Xcal_s, ycal, d_vec, cfg, seed, lam_default, S_mode="delta")

        # ---- baselines (unchanged) ----
        d_norm = float(np.linalg.norm(d_vec))
        q_me = np.quantile(np.abs(ite_true_cal - d_norm), lv)
        reg = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=800, random_state=seed)
        reg.fit(Xtr_s, ite_true_tr)
        tau_hat_cal = reg.predict(Xcal_s)
        q_dt = np.quantile(np.abs(ite_true_cal - tau_hat_cal), lv)

        evals = [("PPMI-internal", Xte, yte, 0, False),
                 ("NEUROCON", ext["NEUROCON"][0], ext["NEUROCON"][1], 1000, True),
                 ("TaoWu", ext["TaoWu"][0], ext["TaoWu"][1], 1500, True)]
        for cname, Xe_raw, ye, off, is_ext in evals:
            Xe_s = Z(Xe_raw)
            rng_e = np.random.RandomState(seed + off)
            _, _, true_e, _ = make_semisynthetic(Xe_s, ye, d_vec, rng_e, cfg)
            _, _, ite_hat, _, sig_e = evaluate_on(model, q, Xe_s, ye, cfg)
            r_e, d_e, _ = proxy_components(model, Xe_s, ye)
            n_e = len(Xe_s)
            methods = [("CPCI-Oracle", ite_hat, q, None),
                       ("RD-CCI-h", ite_hat, q_h, sig_e),
                       ("Point", ite_hat, 0.0, None),
                       ("MeanEffect", np.full(n_e, d_norm), q_me, None),
                       ("Direct-tau", reg.predict(Xe_s), q_dt, None)]
            for lam in lam_grid:
                tag = "" if abs(lam - lam_default) < 1e-9 else "-l%g" % lam
                methods.append(("CPCI-Proxy%s" % tag, ite_hat, q_lam[lam], None))
                methods.append(("CPCI-Scaled%s" % tag, ite_hat, q_sp[lam], None))
            for meth, ih, qq, ss in methods:
                cov, wid, mae, cov_n = _interval_eval(ih, qq, true_e, sigma=ss)
                rows.append(dict(seed=seed, cohort=cname, method=meth, n=n_e,
                                 coverage=cov, width=wid, mae=mae, covered=cov_n,
                                 rate_kl=rate_kl))
                if meth in INT_SAVE:
                    half = qq * ss if ss is not None else qq
                    lo_i = np.maximum(0.0, ih - half)
                    hi_i = ih + half
                    for j in range(n_e):
                        int_rows.append(dict(seed=seed, cohort=cname, method=meth,
                                             sid=j, lo=round(float(lo_i[j]), 4),
                                             hi=round(float(hi_i[j]), 4),
                                             true=round(float(true_e[j]), 4),
                                             err=round(float(abs(ih[j] - true_e[j])), 4)))
            # headline CPCI-Sim: fidelity score
            half_m = np.maximum(0.0, g_rc.predict(r_e) + q_rc)
            lo_m = np.maximum(0.0, ite_hat - half_m)
            hi_m = ite_hat + half_m
            m = (lo_m <= true_e) & (true_e <= hi_m)
            rows.append(dict(seed=seed, cohort=cname, method="CPCI-Sim", n=n_e,
                             coverage=float(m.mean()),
                             width=float((hi_m - lo_m).mean()),
                             mae=float(np.abs(ite_hat - true_e).mean()),
                             covered=int(m.sum()), rate_kl=rate_kl))
            for j in range(n_e):
                int_rows.append(dict(seed=seed, cohort=cname, method="CPCI-Sim",
                                     sid=j, lo=round(float(lo_m[j]), 4),
                                     hi=round(float(hi_m[j]), 4),
                                     true=round(float(true_e[j]), 4),
                                     err=round(float(abs(ite_hat[j] - true_e[j])), 4)))
            # composite-score ablation (formerly the headline)
            S_c = r_e + lam_default * d_e
            half_c = np.maximum(0.0, sim_models[lam_default][0].predict(S_c)
                                + sim_models[lam_default][1])
            lo_c = np.maximum(0.0, ite_hat - half_c)
            hi_c = ite_hat + half_c
            m = (lo_c <= true_e) & (true_e <= hi_c)
            rows.append(dict(seed=seed, cohort=cname,
                             method="CPCI-Sim (composite)", n=n_e,
                             coverage=float(m.mean()),
                             width=float((hi_c - lo_c).mean()),
                             mae=float(np.abs(ite_hat - true_e).mean()),
                             covered=int(m.sum()), rate_kl=rate_kl))
            for j in range(n_e):
                int_rows.append(dict(seed=seed, cohort=cname,
                                     method="CPCI-Sim (composite)", sid=j,
                                     lo=round(float(lo_c[j]), 4),
                                     hi=round(float(hi_c[j]), 4),
                                     true=round(float(true_e[j]), 4),
                                     err=round(float(abs(ite_hat[j] - true_e[j])), 4)))
            # lambda grid (composite weighting), non-default values only
            for lam in lam_grid:
                if abs(lam - lam_default) < 1e-9:
                    continue
                tag = "-l%g" % lam
                g_l, q_sim_l = sim_models[lam]
                S_e = r_e + lam * d_e
                half_e = np.maximum(0.0, g_l.predict(S_e) + q_sim_l)
                lo_s = np.maximum(0.0, ite_hat - half_e)
                hi_s = ite_hat + half_e
                m = (lo_s <= true_e) & (true_e <= hi_s)
                rows.append(dict(seed=seed, cohort=cname,
                                 method="CPCI-Sim%s" % tag, n=n_e,
                                 coverage=float(m.mean()),
                                 width=float((hi_s - lo_s).mean()),
                                 mae=float(np.abs(ite_hat - true_e).mean()),
                                 covered=int(m.sum()), rate_kl=rate_kl))
            # --- v7.3: score-component ablations, analytic RelInt, diagnostics ---
            for gmode, (g_a, q_a) in [("delta", (g_delta, q_delta)),
                                      ("rc", (g_rc, q_rc))]:
                Sa = d_e if gmode == "delta" else r_e
                half_a = np.maximum(0.0, g_a.predict(Sa) + q_a)
                lo_a = np.maximum(0.0, ite_hat - half_a)
                hi_a = ite_hat + half_a
                m = (lo_a <= true_e) & (true_e <= hi_a)
                rows.append(dict(seed=seed, cohort=cname,
                                 method="CPCI-Sim-%s" % gmode, n=n_e,
                                 coverage=float(m.mean()),
                                 width=float((hi_a - lo_a).mean()),
                                 mae=float(np.abs(ite_hat - true_e).mean()),
                                 covered=int(m.sum()), rate_kl=rate_kl))
                for j in range(n_e):
                    int_rows.append(dict(seed=seed, cohort=cname,
                                         method="CPCI-Sim-%s" % gmode, sid=j,
                                         lo=round(float(lo_a[j]), 4),
                                         hi=round(float(hi_a[j]), 4),
                                         true=round(float(true_e[j]), 4),
                                         err=round(float(abs(ite_hat[j] - true_e[j])), 4)))
            # analytic relative interval: half-width = z*ite_hat, z = (1-alpha)
            # quantile of |1-kappa| for kappa ~ U(semi_scale_range)
            z_rel = 0.5 * (1.0 - cfg["alpha"]) * (cfg["semi_scale_range"][1]
                     - cfg["semi_scale_range"][0])
            half_r = z_rel * ite_hat
            lo_r = np.maximum(0.0, ite_hat - half_r)
            hi_r = ite_hat + half_r
            m = (lo_r <= true_e) & (true_e <= hi_r)
            rows.append(dict(seed=seed, cohort=cname, method="RelInt", n=n_e,
                             coverage=float(m.mean()),
                             width=float((hi_r - lo_r).mean()),
                             mae=float(np.abs(ite_hat - true_e).mean()),
                             covered=int(m.sum()), rate_kl=rate_kl))
            for j in range(n_e):
                int_rows.append(dict(seed=seed, cohort=cname, method="RelInt", sid=j,
                                     lo=round(float(lo_r[j]), 4),
                                     hi=round(float(hi_r[j]), 4),
                                     true=round(float(true_e[j]), 4),
                                     err=round(float(abs(ite_hat[j] - true_e[j])), 4)))
            # score diagnostics: can components rank unobservable errors?
            abs_err = np.abs(true_e - ite_hat)
            diag.append(dict(seed=seed, cohort=cname,
                             sp_score=spearmanr(r_e + lam_default * d_e, abs_err).correlation,
                             sp_delta=spearmanr(d_e, abs_err).correlation,
                             sp_rc=spearmanr(r_e, abs_err).correlation))
            if is_ext:
                dom = LogisticRegression(max_iter=1000)
                dom.fit(np.vstack([Xcal_s, Xe_s]),
                        np.concatenate([np.zeros(len(Xcal_s)), np.ones(n_e)]))
                p1 = dom.predict_proba(Xcal_s)[:, 1]
                w = np.clip(p1 / np.maximum(1e-6, 1 - p1), 0.1, 10.0)
                S_cal = np.abs(ite_true_cal - ite_hat_cal)
                q_w = weighted_quantile(S_cal, w, 1 - cfg["alpha"])
                cov, wid, mae, cov_n = _interval_eval(ite_hat, q_w, true_e)
                rows.append(dict(seed=seed, cohort=cname, method="Weighted-CP",
                                 n=n_e, coverage=cov, width=wid, mae=mae, covered=cov_n,
                                 rate_kl=rate_kl))
                for j in range(n_e):
                    int_rows.append(dict(seed=seed, cohort=cname, method="Weighted-CP",
                                         sid=j, lo=round(float(max(0.0, ite_hat[j] - q_w)), 4),
                                         hi=round(float(ite_hat[j] + q_w), 4),
                                         true=round(float(true_e[j]), 4),
                                         err=round(float(abs(ite_hat[j] - true_e[j])), 4)))
                perm = np.random.RandomState(seed + 3000).permutation(n_e)
                n_or = max(5, int(0.3 * n_e))
                or_i, ev_i = perm[:n_or], perm[n_or:]
                q_or = np.quantile(np.abs(true_e[or_i] - ite_hat[or_i]),
                                   _quantile_ci_level(n_or, cfg["alpha"]))
                cov, wid, mae, cov_n = _interval_eval(ite_hat[ev_i], q_or, true_e[ev_i])
                rows.append(dict(seed=seed, cohort=cname, method="Oracle",
                                 n=len(ev_i), coverage=cov, width=wid, mae=mae, covered=cov_n,
                                 rate_kl=rate_kl))
                for j in ev_i:
                    int_rows.append(dict(seed=seed, cohort=cname, method="Oracle",
                                         sid=j, lo=round(float(max(0.0, ite_hat[j] - q_or)), 4),
                                         hi=round(float(ite_hat[j] + q_or), 4),
                                         true=round(float(true_e[j]), 4),
                                         err=round(float(abs(ite_hat[j] - true_e[j])), 4)))
        print("[main] seed %d done (%ds)" % (seed, time.time() - t0))
    df = pd.DataFrame(rows)
    ORACLE_METHODS = {"CPCI-Oracle", "RD-CCI-h", "MeanEffect", "Direct-tau",
                      "Weighted-CP", "Oracle"}
    df["tier"] = df["method"].map(
        lambda m: "oracle" if m in ORACLE_METHODS
        else ("none" if m == "Point" else "deployable"))
    out = Path(cfg["out_root"]); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "three_tier_results_per_seed.csv", index=False)
    pd.DataFrame(int_rows).to_csv(out / "intervals_per_subject.csv", index=False)
    pd.DataFrame(eff_rows).to_csv(out / "effect_direction.csv", index=False)
    print("[main] saved intervals_per_subject.csv (%d rows) and effect_direction.csv"
          % len(int_rows))
    pd.DataFrame(diag).to_csv(out / "score_diagnostics.csv", index=False)
    dsp = pd.DataFrame(diag).groupby("cohort")[["sp_score", "sp_delta", "sp_rc"]].mean()
    print("\n[diag] Spearman(|ITE error|, score components), mean over seeds:")
    print(dsp.round(4).to_string())
    print("    [feeds paper: does the composite score beat its components?]")

    print("\n===== summary, Wilson 95%% CI on pooled subjects =====")
    print("    [feeds paper Table: multi-method coverage, MRI cohorts]")
    rows_s = []
    stable = df["rate_kl"] >= 5.0
    for (cname, meth), g in df.groupby(["cohort", "method"]):
        k, n = int(g["covered"].sum()), int(g["n"].sum())
        lo_w, hi_w = wilson_ci(k, n)
        gs = g[stable.loc[g.index]]
        rows_s.append(dict(cohort=cname, method=meth, coverage=g["coverage"].mean(),
                           coverage_ci="[%.3f,%.3f]" % (lo_w, hi_w),
                           cov_stable=gs["coverage"].mean() if len(gs) else np.nan,
                           n_stable=int(len(gs)),
                           width=g["width"].mean(), mae=g["mae"].mean(), seeds=len(g)))
        print("%-14s %-18s cov=%.3f (CI %.3f-%.3f; stable %.3f n=%d) width=%.3f mae=%.3f"
              % (cname, meth, g["coverage"].mean(), lo_w, hi_w,
                 gs["coverage"].mean() if len(gs) else float("nan"), len(gs),
                 g["width"].mean(), g["mae"].mean()))
    sdf = pd.DataFrame(rows_s)
    sdf.to_csv(out / "three_tier_results_summary.csv", index=False)

    print("\n===== DEPLOYABLE summary: proxy vs oracle upper bound =====")
    print("    [feeds paper Table: Oracle vs Proxy vs Scaled coverage + gap to nominal]")
    nom = 1 - cfg["alpha"]
    dep_rows = []
    for cname in ["PPMI-internal", "NEUROCON", "TaoWu"]:
        sub = sdf[sdf["cohort"] == cname]
        line = "%-14s" % cname
        for meth in ["CPCI-Oracle", "CPCI-Proxy", "CPCI-Scaled", "CPCI-Sim"]:
            r = sub[sub["method"] == meth]
            if len(r):
                cov = r["coverage"].iloc[0]
                dep_rows.append(dict(cohort=cname, method=meth, coverage=cov,
                                     gap_to_nominal=cov - nom, width=r["width"].iloc[0]))
                line += "  %s cov=%.3f (gap %+.3f)" % (meth, cov, cov - nom)
        print(line)
    pd.DataFrame(dep_rows).to_csv(out / "deployable_summary.csv", index=False)

    print("\n===== lambda sensitivity of the COMPOSITE score (default lam=%g) =====" % lam_default)
    print("    [feeds paper Table: hyperparameter sensitivity]")
    lam_rows = []
    sub = sdf[sdf["method"].str.startswith("CPCI-Proxy")]
    for (cname, meth), g in sub.groupby(["cohort", "method"]):
        lam_rows.append(dict(cohort=cname, method=meth, coverage=g["coverage"].mean(),
                             width=g["width"].mean()))
        print("%-14s %-18s cov=%.3f width=%.3f" % (cname, meth, g["coverage"].mean(), g["width"].mean()))
    pd.DataFrame(lam_rows).to_csv(out / "lambda_sensitivity.csv", index=False)
    print("\nsaved to %s" % out)


# ==================== Experiment 2 (rate sweep + proxy gap) ====================

def bootstrap_slope(per_seed, xcol, ycol, n_boot=2000, seed=0):
    rng = np.random.RandomState(seed)
    vals = np.sort(per_seed["val"].unique())
    xs = per_seed.groupby("val")[xcol].mean().to_numpy()
    slopes = []
    for _ in range(n_boot):
        ys = []
        for v in vals:
            gg = per_seed[per_seed["val"] == v][ycol].to_numpy()
            ys.append(gg[rng.randint(0, len(gg), len(gg))].mean())
        if min(ys) > 1e-6:
            slopes.append(np.polyfit(np.log(xs), np.log(np.array(ys)), 1)[0])
    return np.percentile(slopes, [2.5, 50, 97.5])


def cmd_rate(cfg):
    Xp, yp, files_p = load_cohort(cfg, "PPMI")
    groups_p = np.array([_digits(f) for f in files_p])
    Xne, yne, _ = load_cohort(cfg, "NEUROCON")
    Xtw, ytw, _ = load_cohort(cfg, "TaoWu")
    lam = float(cfg["lambda_proxy"])
    rows = []
    grids = ([("lam_rd", v) for v in [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1]]
             + [("latent", v) for v in [8, 16, 32, 64, 96]])
    for mode, val in grids:
        for seed in cfg["seeds"]:
            rng = np.random.RandomState(seed)
            tr, cal, te = grouped_train_cal_test(Xp, yp, groups_p, cfg["test_frac"], 0.25, seed)
            Xtr, ytr, Xcal, ycal, Xte, yte = Xp[tr], yp[tr], Xp[cal], yp[cal], Xp[te], yp[te]
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
            Z = lambda A: (A - mu) / sd
            Xtr_s, Xcal_s, Xte_s = Z(Xtr), Z(Xcal), Z(Xte)
            d_vec = Xtr_s[ytr == 1].mean(0) - Xtr_s[ytr == 0].mean(0)
            kw = dict(latent=val) if mode == "latent" else dict(lam_rd=val)
            _, _, ite_true_tr, _ = make_semisynthetic(Xtr_s, ytr, d_vec, rng, cfg)
            _, _, ite_true_cal, _ = make_semisynthetic(Xcal_s, ycal, d_vec, rng, cfg)
            model, q, rate_kl, dist, _ = conformal_intervals(
                Xtr_s, ytr, Xcal_s, ycal, ite_true_cal, cfg, seed=seed,
                ite_true_tr=ite_true_tr, **kw)
            # deployable proxy quantile on calibration set
            q_lam, q_sp, rho = proxy_quantiles(
                model, Xcal_s, ycal, Xtr_s, ytr, ite_true_tr, cfg, [lam])
            q_proxy = q_lam[lam]
            g_sim, q_sim = sim_proxy_calibration(
                model, Xcal_s, ycal, d_vec, cfg, seed, lam, n_sim=8)
            rec = dict(sweep=mode, val=val, seed=seed, rate_kl=rate_kl, distortion=dist)
            _, _, ite_true, _ = make_semisynthetic(Xte_s, yte, d_vec, rng, cfg)
            lo, hi, ite_hat, _, _ = evaluate_on(model, q, Xte_s, yte, cfg)
            rec["width_internal"] = float((hi - lo).mean())
            rec["cov_internal"] = float(((lo <= ite_true) & (ite_true <= hi)).mean())
            Zne, Ztw = Z(Xne), Z(Xtw)
            Xe_all = np.vstack([Zne, Ztw])
            ye_all = np.concatenate([yne, ytw])
            for tag, Xe, ye, off in [("NEUROCON", Zne, yne, 1000),
                                     ("TaoWu", Ztw, ytw, 1500),
                                     ("External", Xe_all, ye_all, 2000)]:
                rng_e = np.random.RandomState(seed + off)
                _, _, ite_true_e, _ = make_semisynthetic(Xe, ye, d_vec, rng_e, cfg)
                lo, hi, _, _, _ = evaluate_on(model, q, Xe, ye, cfg)
                cov_e = float(((lo <= ite_true_e) & (ite_true_e <= hi)).mean())
                rec["cov_%s" % tag] = cov_e
                rec["gap_%s" % tag] = abs(cov_e - (1 - cfg["alpha"]))
                # deployable proxy coverage under shift (v7)
                r_e, d_e, _ = proxy_components(model, Xe, ye)
                lo_p = np.maximum(0.0, d_e - q_proxy)
                hi_p = d_e + q_proxy
                cov_p = float(((lo_p <= ite_true_e) & (ite_true_e <= hi_p)).mean())
                rec["cov_proxy_%s" % tag] = cov_p
                rec["gap_proxy_%s" % tag] = abs(cov_p - (1 - cfg["alpha"]))
                S_e = r_e + lam * d_e
                half_e = np.maximum(0.0, g_sim.predict(S_e) + q_sim)
                lo_s = np.maximum(0.0, d_e - half_e)
                hi_s = d_e + half_e
                cov_s = float(((lo_s <= ite_true_e) & (ite_true_e <= hi_s)).mean())
                rec["cov_sim_%s" % tag] = cov_s
                rec["gap_sim_%s" % tag] = abs(cov_s - (1 - cfg["alpha"]))
            rows.append(rec)
            print("[rate] %s=%g seed=%d rate=%.2f nats w=%.3f gap_NE=%.3f gap_TW=%.3f gap_Ext=%.3f | SIM gap_Ext=%.3f"
                  % (mode, val, seed, rate_kl, rec["width_internal"],
                     rec["gap_NEUROCON"], rec["gap_TaoWu"], rec["gap_External"],
                     rec["gap_sim_External"]))
    df = pd.DataFrame(rows)
    out = Path(cfg["out_root"]); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "rate_curve.csv", index=False)
    df["collapsed"] = df["rate_kl"] < 5.0
    g = df.groupby(["sweep", "val"]).agg(
        rate_kl=("rate_kl", "mean"), distortion=("distortion", "mean"),
        width=("width_internal", "mean"), cov_int=("cov_internal", "mean"),
        gap_NE=("gap_NEUROCON", "mean"), gap_TW=("gap_TaoWu", "mean"),
        gap_Ext=("gap_External", "mean"),
        gap_proxy_NE=("gap_proxy_NEUROCON", "mean"),
        gap_proxy_TW=("gap_proxy_TaoWu", "mean"),
        gap_proxy_Ext=("gap_proxy_External", "mean"),
        gap_sim_NE=("gap_sim_NEUROCON", "mean"),
        gap_sim_TW=("gap_sim_TaoWu", "mean"),
        gap_sim_Ext=("gap_sim_External", "mean"),
        n_collapsed=("collapsed", "sum"),
        gap_Ext_stable=("gap_External", lambda s: s[df.loc[s.index, "collapsed"] == False].mean())).reset_index()
    g.to_csv(out / "rate_curve_summary.csv", index=False)
    print("\n[rate] summary:"); print(g.round(4).to_string(index=False))

    fits = {}
    print("\n[rate] bootstrap log-log slopes  [feeds paper Sec. 4.2 empirical rate law]")
    for sweep in ["lam_rd", "latent"]:
        dsw = df[df["sweep"] == sweep]
        dstab = dsw[~dsw["collapsed"]]
        print("\n[rate] --- sweep=%s ---" % sweep)
        for cname in ["NEUROCON", "TaoWu", "External"]:
            for prefix, label in [("gap_", "oracle"), ("gap_sim_", "SIM")]:
                ycol = "%s%s" % (prefix, cname)
                lo_, med_, hi_ = bootstrap_slope(dsw, "rate_kl", ycol)
                fits[(sweep, cname, label)] = med_
                print("  [%s] %s: gap ~ rate^%.3f (95%% CI [%.3f, %.3f])" % (label, cname, med_, lo_, hi_))
                if label == "oracle" and len(dstab) >= 3 and (dstab[ycol] > 1e-4).sum() >= 3:
                    lo_s, med_s, hi_s = bootstrap_slope(dstab, "rate_kl", ycol)
                    print("  [oracle-stable] %s: gap ~ rate^%.3f (95%% CI [%.3f, %.3f])"
                          % (cname, med_s, lo_s, hi_s))
    pd.DataFrame([dict(sweep=s, cohort=c, protocol=p, slope=m)
                  for (s, c, p), m in fits.items()]).to_csv(
        out / "rate_slopes.csv", index=False)
    try:
        import matplotlib.pyplot as plt
        gl = g[g["sweep"] == "lam_rd"]
        gz = g[g["sweep"] == "latent"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        axes[0].loglog(gl["rate_kl"], gl["distortion"], "s-", color="tab:red")
        axes[0].set_xlabel("Rate (nats)")
        axes[0].set_ylabel("Distortion (recon MSE)")
        axes[0].set_title("(A) Rate-Distortion Frontier")
        axes[0].grid(True, alpha=0.3)
        styles = [("NEUROCON", "gap_NE", "o", "tab:blue"),
                  ("TaoWu", "gap_TW", "^", "tab:green"),
                  ("External", "gap_Ext", "D", "tab:red")]
        dfr = df[df["sweep"] == "lam_rd"]
        for cname, gcol, mk, col in styles:
            axes[1].scatter(dfr["rate_kl"], dfr[gcol], marker=mk, alpha=0.2, color=col, s=16)
            axes[1].loglog(gl["rate_kl"], gl[gcol], mk + "-", color=col, label=cname + " (oracle)")
            axes[1].scatter(dfr["rate_kl"], dfr["gap_proxy_" + cname], marker=mk, alpha=0.2,
                            color=col, facecolors="none", s=16)
            s = fits[("lam_rd", cname, "oracle")]
            xs = np.array([gl["rate_kl"].min(), gl["rate_kl"].max()])
            b = np.log(gl[gcol].mean()) - s * np.log(gl["rate_kl"].mean())
            axes[1].loglog(xs, np.exp(b) * xs ** s, "--", alpha=0.5, color=col)
        axes[1].set_xlabel("Rate (nats)")
        axes[1].set_ylabel("Coverage gap under shift")
        axes[1].set_title("(B) Rate-Coverage Gap (solid=oracle, open=proxy)")
        axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
        dfz = df[df["sweep"] == "latent"]
        for cname, gcol, mk, col in styles:
            axes[2].scatter(dfz["distortion"], dfz[gcol], marker=mk, alpha=0.2, color=col, s=16)
            axes[2].plot(gz["distortion"], gz[gcol], mk + "-", color=col, label=cname)
        axes[2].set_xlabel("Distortion (recon MSE)")
        axes[2].set_ylabel("Coverage gap under shift")
        axes[2].set_title("(C) Bottleneck Sweep (latent dim)")
        axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "rate_coverage_curve.png", dpi=300)
        print("\n[rate] figure saved: %s  [feeds paper Fig: rate-coverage curve]" % (out / "rate_coverage_curve.png"))
    except Exception as e:
        print("plot skipped:", e)



# ==================== Experiment 3: simulation-family stress test ====================

def cmd_stress(cfg):
    """Simulation-family misspecification stress test (v7.3).

    The calibration family kappa ~ U(0.5,1.5) is fixed; the INJECTED
    evaluation severity s(x) = center + amp*tanh(beta^T x) is deliberately
    moved away from it: wider (0.2,1.8), shifted-low, shifted-high. One
    model is trained per seed (matched family); CPCI-Sim / RelInt /
    CPCI-Oracle are then re-evaluated on the same cohorts under each
    stressed severity. Answers the review question: how much of CPCI-Sim's
    coverage comes from the assumed severity family itself?"""
    Xp, yp, files_p = load_cohort(cfg, "PPMI")
    groups_p = np.array([_digits(f) for f in files_p])
    ext = {}
    for c in ["NEUROCON", "TaoWu"]:
        X, y, _ = load_cohort(cfg, c)
        ext[c] = (X, y)
    lam = float(cfg["lambda_proxy"])
    z_rel = 0.5 * (1.0 - cfg["alpha"]) * (cfg["semi_scale_range"][1]
             - cfg["semi_scale_range"][0])
    configs = [("matched", 1.0, 0.5), ("wide", 1.0, 0.8),
               ("shifted-low", 0.8, 0.6), ("shifted-high", 1.2, 0.4)]
    rows = []
    for seed in cfg["seeds"]:
        rng = np.random.RandomState(seed)
        tr, cal, te = grouped_train_cal_test(Xp, yp, groups_p, cfg["test_frac"], 0.25, seed)
        Xtr, ytr, Xcal, ycal, Xte, yte = Xp[tr], yp[tr], Xp[cal], yp[cal], Xp[te], yp[te]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Z = lambda A: (A - mu) / sd
        Xtr_s, Xcal_s, Xte_s = Z(Xtr), Z(Xcal), Z(Xte)
        d_vec = Xtr_s[ytr == 1].mean(0) - Xtr_s[ytr == 0].mean(0)
        _, _, ite_true_tr, _ = make_semisynthetic(Xtr_s, ytr, d_vec, rng, cfg)
        _, _, ite_true_cal, _ = make_semisynthetic(Xcal_s, ycal, d_vec, rng, cfg)
        model, q, rate_kl, dist, _ = conformal_intervals(
            Xtr_s, ytr, Xcal_s, ycal, ite_true_cal, cfg, seed=seed, ite_true_tr=ite_true_tr)
        g_sim, q_sim = sim_proxy_calibration(model, Xcal_s, ycal, d_vec, cfg, seed, lam)
        evals = [("PPMI-internal", Xte, yte, 0),
                 ("NEUROCON", ext["NEUROCON"][0], ext["NEUROCON"][1], 1000),
                 ("TaoWu", ext["TaoWu"][0], ext["TaoWu"][1], 1500)]
        for cname, Xe_raw, ye, off in evals:
            Xe_s = Z(Xe_raw)
            r_e, d_e, _ = proxy_components(model, Xe_s, ye)
            ite_hat = d_e
            n_e = len(Xe_s)
            for tag_c, ctr, amp in configs:
                rng_e = np.random.RandomState(seed + off)
                beta = effect_beta(Xe_s.shape[1])
                s = ctr + amp * np.tanh(Xe_s @ beta)
                true_e = s * np.linalg.norm(d_vec)
                for meth, half in [("CPCI-Sim", np.maximum(0.0, g_sim.predict(r_e + lam * d_e) + q_sim)),
                                   ("RelInt", z_rel * ite_hat),
                                   ("CPCI-Oracle", np.full(n_e, q))]:
                    half_v = half if hasattr(half, "__len__") else np.full(n_e, half)
                    lo = np.maximum(0.0, ite_hat - half_v)
                    hi = ite_hat + half_v
                    m = (lo <= true_e) & (true_e <= hi)
                    rows.append(dict(seed=seed, cohort=cname, stress=tag_c,
                                     method=meth, coverage=float(m.mean()),
                                     width=float((hi - lo).mean()),
                                     covered=int(m.sum()), n=n_e,
                                     rate_kl=rate_kl))
        print("[stress] seed %d done (rate=%.1f)" % (seed, rate_kl))
    df = pd.DataFrame(rows)
    out = Path(cfg["out_root"]); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "stress_results.csv", index=False)
    g = df.groupby(["stress", "cohort", "method"]).agg(
        coverage=("coverage", "mean"), width=("width", "mean"),
        seeds=("seed", "count")).reset_index()
    g.to_csv(out / "stress_summary.csv", index=False)
    print("\n[stress] summary  [feeds paper: simulation misspecification table]")
    print(g.round(4).to_string(index=False))



# ==================== Experiment 4: domain-shift characterization ====================

def cmd_shift(cfg):
    """Quantify the domain shift between the internal cohort (PPMI) and each
    external cohort on the kept AAL features: 5-fold cross-validated
    domain-classifier AUC, squared mean-feature shift, and group-balance
    comparison. Feeds the paper's shift-characterization paragraph."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    Xp, yp, _ = load_cohort(cfg, "PPMI")
    out = Path(cfg["out_root"]); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in ["NEUROCON", "TaoWu"]:
        Xe, ye, _ = load_cohort(cfg, c)
        Xs = np.vstack([Xp, Xe]).astype(np.float64)
        ys = np.concatenate([np.zeros(len(Xp)), np.ones(len(Xe))])
        pipe = Pipeline([("sc", StandardScaler()),
                         ("lr", LogisticRegression(max_iter=2000))])
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        pr = cross_val_predict(pipe, Xs, ys, cv=skf, method="predict_proba")[:, 1]
        auc = float(roc_auc_score(ys, pr))
        rows.append(dict(pair="PPMI-%s" % c, n_source=len(Xp), n_target=len(Xe),
                         domain_auc=round(auc, 4),
                         mean_feature_shift=float(((Xp.mean(0) - Xe.mean(0)) ** 2).sum()),
                         pd_frac_source=round(float(yp.mean()), 4),
                         pd_frac_target=round(float(ye.mean()), 4)))
    df = pd.DataFrame(rows)
    df.to_csv(out / "shift_stats.csv", index=False)
    print(df.round(4).to_string(index=False))
    print("    [feeds paper Sec 4.1: shift characterization]")


# ==================== Experiment 5: effect-direction mismatch sweep ====================

def cmd_stress2(cfg):
    """Direction-mismatch stress test: evaluation effects are injected along
    d_rot(theta) = cos(theta)*d_hat + sin(theta)*d_perp while calibration
    still assumes effects aligned with the model's estimate. theta sweeps
    0..90 deg. Answers: does CPCI-Sim survive wrong EFFECT DIRECTION, not
    just wrong severity scale?"""
    Xp, yp, files_p = load_cohort(cfg, "PPMI")
    groups_p = np.array([_digits(f) for f in files_p])
    ext = {}
    for c in ["NEUROCON", "TaoWu"]:
        X, y, _ = load_cohort(cfg, c)
        ext[c] = (X, y)
    lam = float(cfg["lambda_proxy"])
    z_rel = 0.5 * (1.0 - cfg["alpha"]) * (cfg["semi_scale_range"][1]
             - cfg["semi_scale_range"][0])
    thetas = [0, 15, 30, 45, 60, 75, 90]
    rows = []
    for seed in cfg["seeds"]:
        rng = np.random.RandomState(seed)
        tr, cal, te = grouped_train_cal_test(Xp, yp, groups_p, cfg["test_frac"], 0.25, seed)
        Xtr, ytr, Xcal, ycal, Xte, yte = Xp[tr], yp[tr], Xp[cal], yp[cal], Xp[te], yp[te]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Z = lambda A: (A - mu) / sd
        Xtr_s, Xcal_s, Xte_s = Z(Xtr), Z(Xcal), Z(Xte)
        d_vec = Xtr_s[ytr == 1].mean(0) - Xtr_s[ytr == 0].mean(0)
        _, _, ite_true_tr, _ = make_semisynthetic(Xtr_s, ytr, d_vec, rng, cfg)
        _, _, ite_true_cal, _ = make_semisynthetic(Xcal_s, ycal, d_vec, rng, cfg)
        model, q, rate_kl, dist, _ = conformal_intervals(
            Xtr_s, ytr, Xcal_s, ycal, ite_true_cal, cfg, seed=seed, ite_true_tr=ite_true_tr)
        g_sim, q_sim = sim_proxy_calibration(model, Xcal_s, ycal, d_vec, cfg, seed, lam)
        # fixed orthonormal perturbation direction
        beta = effect_beta(d_vec.shape[0], seed=999)
        d_perp = beta - (beta @ d_vec) * d_vec / max(1e-12, float(d_vec @ d_vec))
        d_perp = d_perp / (np.linalg.norm(d_perp) + 1e-12)
        beta_sev = effect_beta(d_vec.shape[0])   # severity direction (as in eval)
        evals = [("PPMI-internal", Xte, yte, 0),
                 ("NEUROCON", ext["NEUROCON"][0], ext["NEUROCON"][1], 1000),
                 ("TaoWu", ext["TaoWu"][0], ext["TaoWu"][1], 1500)]
        for cname, Xe_raw, ye, off in evals:
            Xe_s = Z(Xe_raw)
            r_e, d_e, _ = proxy_components(model, Xe_s, ye)
            ite_hat = d_e
            n_e = len(Xe_s)
            half_sim = np.maximum(0.0, g_sim.predict(r_e + lam * d_e) + q_sim)
            s_sev = 1.0 + 0.5 * np.tanh(Xe_s @ beta_sev)
            for th in thetas:
                a, b = np.cos(np.radians(th)), np.sin(np.radians(th))
                d_rot = a * d_vec + b * d_perp
                true_e = s_sev * np.linalg.norm(d_rot)
                for meth, half in [("CPCI-Sim", half_sim),
                                   ("RelInt", z_rel * ite_hat),
                                   ("CPCI-Oracle", np.full(n_e, q))]:
                    lo = np.maximum(0.0, ite_hat - half)
                    hi = ite_hat + half
                    m = (lo <= true_e) & (true_e <= hi)
                    rows.append(dict(seed=seed, cohort=cname, theta=th, method=meth,
                                     coverage=float(m.mean()),
                                     width=float((hi - lo).mean()),
                                     covered=int(m.sum()), n=n_e, rate_kl=rate_kl))
        print("[stress2] seed %d done (rate=%.1f)" % (seed, rate_kl))
    df = pd.DataFrame(rows)
    out = Path(cfg["out_root"]); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "stress2_results.csv", index=False)
    g = df.groupby(["theta", "cohort", "method"]).agg(
        coverage=("coverage", "mean"), width=("width", "mean"),
        seeds=("seed", "count")).reset_index()
    g.to_csv(out / "stress2_summary.csv", index=False)
    print("\n[stress2] summary  [feeds paper direction-mismatch heatmap]")
    print(g.round(4).to_string(index=False))


# ==================== Experiment 6: score/error CDF data (for the CDF figure) ====================

def cmd_cdf(cfg, n_seeds=5):
    """Save score/error distributions underlying the calibration story:
    source-GT calibration errors (oracle scores), simulated calibration
    errors, and target-GT errors (evaluation only), for the first few
    seeds. Used ONLY to draw the F_sim vs F_target CDF comparison."""
    seeds = list(cfg["seeds"])[:n_seeds]
    Xp, yp, files_p = load_cohort(cfg, "PPMI")
    groups_p = np.array([_digits(f) for f in files_p])
    ext = {}
    for c in ["NEUROCON", "TaoWu"]:
        X, y, _ = load_cohort(cfg, c)
        ext[c] = (X, y)
    rows = []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        tr, cal, te = grouped_train_cal_test(Xp, yp, groups_p, cfg["test_frac"], 0.25, seed)
        Xtr, ytr, Xcal, ycal = Xp[tr], yp[tr], Xp[cal], yp[cal]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Z = lambda A: (A - mu) / sd
        Xtr_s, Xcal_s = Z(Xtr), Z(Xcal)
        d_vec = Xtr_s[ytr == 1].mean(0) - Xtr_s[ytr == 0].mean(0)
        _, _, ite_true_tr, _ = make_semisynthetic(Xtr_s, ytr, d_vec, rng, cfg)
        _, _, ite_true_cal, _ = make_semisynthetic(Xcal_s, ycal, d_vec, rng, cfg)
        model, q, rate_kl, dist, _ = conformal_intervals(
            Xtr_s, ytr, Xcal_s, ycal, ite_true_cal, cfg, seed=seed, ite_true_tr=ite_true_tr)
        _, x0_c, x1_c, _, _ = vae_outputs(model, Xcal_s, ycal)
        gt_cal = np.abs(ite_true_cal - np.linalg.norm(x1_c - x0_c, axis=1))
        for v in gt_cal:
            rows.append(dict(seed=seed, kind="source-GT", value=float(v)))
        r_cal, d_cal, _ = proxy_components(model, Xcal_s, ycal)
        rng_s = np.random.RandomState(seed + 7777)
        sim_err = d_cal.repeat(16) * np.abs(1.0 - rng_s.uniform(
            cfg["semi_scale_range"][0], cfg["semi_scale_range"][1], size=len(d_cal) * 16))
        for v in sim_err[::4]:  # subsample for file size
            rows.append(dict(seed=seed, kind="simulated", value=float(v)))
        for cname, (Xe, ye), off in [("NEUROCON", ext["NEUROCON"], 1000),
                                     ("TaoWu", ext["TaoWu"], 1500)]:
            Xe_s = Z(Xe)
            rng_e = np.random.RandomState(seed + off)
            _, _, true_e, _ = make_semisynthetic(Xe_s, ye, d_vec, rng_e, cfg)
            r_e, d_e, _ = proxy_components(model, Xe_s, ye)
            for v in np.abs(true_e - d_e):
                rows.append(dict(seed=seed, kind="target-GT (%s)" % cname, value=float(v)))
        print("[cdf] seed %d done" % seed)
    out = Path(cfg["out_root"]); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "score_cdfs.csv", index=False)
    print("[cdf] saved score_cdfs.csv (%d rows)" % len(rows))



# ==================== Experiment 7: sensitivity to calibration hyperparameters ====================

def cmd_sens(cfg):
    """Sensitivity of CPCI-Sim (fidelity score) to: (i) the severity range
    (a,b) of the simulation family; (ii) simulations per subject M;
    (iii) the split fraction f. The generator is trained ONCE per seed;
    only the calibration layer varies. Feeds the paper's a-priori
    justification of (0.5, 1.5) and the robustness-to-knob table."""
    Xp, yp, files_p = load_cohort(cfg, "PPMI")
    groups_p = np.array([_digits(f) for f in files_p])
    ext = {}
    for c in ["NEUROCON", "TaoWu"]:
        X, y, _ = load_cohort(cfg, c)
        ext[c] = (X, y)
    lam = float(cfg["lambda_proxy"])
    z_rel = 0.5 * (1.0 - cfg["alpha"]) * (cfg["semi_scale_range"][1]
             - cfg["semi_scale_range"][0])
    grid = ([("range", "%.1f-%.1f" % ab, dict(range_override=ab))
             for ab in [(0.7, 1.3), (0.6, 1.4), (0.5, 1.5), (0.4, 1.6)]]
            + [("M", str(m), dict(n_sim=m)) for m in [8, 16, 32]]
            + [("f", str(f), dict(fit_frac=f)) for f in [0.4, 0.6]])
    alpha = cfg["alpha"]
    rows = []
    for seed in cfg["seeds"]:
        rng = np.random.RandomState(seed)
        tr, cal, te = grouped_train_cal_test(Xp, yp, groups_p, cfg["test_frac"], 0.25, seed)
        Xtr, ytr, Xcal, ycal, Xte, yte = Xp[tr], yp[tr], Xp[cal], yp[cal], Xp[te], yp[te]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Z = lambda A: (A - mu) / sd
        Xtr_s, Xcal_s, Xte_s = Z(Xtr), Z(Xcal), Z(Xte)
        d_vec = Xtr_s[ytr == 1].mean(0) - Xtr_s[ytr == 0].mean(0)
        _, _, ite_true_tr, _ = make_semisynthetic(Xtr_s, ytr, d_vec, rng, cfg)
        _, _, ite_true_cal, _ = make_semisynthetic(Xcal_s, ycal, d_vec, rng, cfg)
        model, q, rate_kl, dist, _ = conformal_intervals(
            Xtr_s, ytr, Xcal_s, ycal, ite_true_cal, cfg, seed=seed, ite_true_tr=ite_true_tr)
        evals = [("PPMI-internal", Xte, yte, 0),
                 ("NEUROCON", ext["NEUROCON"][0], ext["NEUROCON"][1], 1000),
                 ("TaoWu", ext["TaoWu"][0], ext["TaoWu"][1], 1500)]
        for knob, val, kw in grid:
            g_k, q_k = sim_proxy_calibration(model, Xcal_s, ycal, d_vec, cfg,
                                             seed, lam, S_mode="rc", **kw)
            for cname, Xe_raw, ye, off in evals:
                Xe_s = Z(Xe_raw)
                r_e, d_e, _ = proxy_components(model, Xe_s, ye)
                ite_hat = d_e
                half = np.maximum(0.0, g_k.predict(r_e) + q_k)
                lo = np.maximum(0.0, ite_hat - half)
                hi = ite_hat + half
                rng_e = np.random.RandomState(seed + off)
                _, _, true_e, _ = make_semisynthetic(Xe_s, ye, d_vec, rng_e, cfg)
                m = (lo <= true_e) & (true_e <= hi)
                width = hi - lo
                is_score = float(np.mean(width
                            + (true_e < lo) * (2.0 / alpha) * (lo - true_e)
                            + (true_e > hi) * (2.0 / alpha) * (true_e - hi)))
                rows.append(dict(seed=seed, knob=knob, value=val, cohort=cname,
                                 coverage=float(m.mean()),
                                 width=float(width.mean()), is_score=is_score))
        print("[sens] seed %d done" % seed)
    df = pd.DataFrame(rows)
    out = Path(cfg["out_root"]); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "sens_results.csv", index=False)
    g = df.groupby(["knob", "value", "cohort"]).agg(
        coverage=("coverage", "mean"), width=("width", "mean"),
        is_score=("is_score", "mean"), seeds=("seed", "count")).reset_index()
    g.to_csv(out / "sens_summary.csv", index=False)
    print("\n[sens] summary  [feeds paper hyperparameter-sensitivity table]")
    print(g.round(4).to_string(index=False))



# ==================== Experiment 8: fully independent effect mechanism ====================

def cmd_stress3(cfg):
    """Strongest circularity break: the evaluation effect direction is
    drawn INDEPENDENTLY of the group contrast d_hat (random unit
    direction, same norm), with severity driven by a second independent
    random direction. Deployed intervals are unchanged. If CPCI-Sim
    retains meaningful coverage here, the shared-generation concern is
    answered; if not, the result bounds the method honestly."""
    Xp, yp, files_p = load_cohort(cfg, "PPMI")
    groups_p = np.array([_digits(f) for f in files_p])
    ext = {}
    for c in ["NEUROCON", "TaoWu"]:
        X, y, _ = load_cohort(cfg, c)
        ext[c] = (X, y)
    lam = float(cfg["lambda_proxy"])
    z_rel = 0.5 * (1.0 - cfg["alpha"]) * (cfg["semi_scale_range"][1]
             - cfg["semi_scale_range"][0])
    rows = []
    for seed in cfg["seeds"]:
        rng = np.random.RandomState(seed)
        tr, cal, te = grouped_train_cal_test(Xp, yp, groups_p, cfg["test_frac"], 0.25, seed)
        Xtr, ytr, Xcal, ycal, Xte, yte = Xp[tr], yp[tr], Xp[cal], yp[cal], Xp[te], yp[te]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Z = lambda A: (A - mu) / sd
        Xtr_s, Xcal_s, Xte_s = Z(Xtr), Z(Xcal), Z(Xte)
        d_vec = Xtr_s[ytr == 1].mean(0) - Xtr_s[ytr == 0].mean(0)
        _, _, ite_true_tr, _ = make_semisynthetic(Xtr_s, ytr, d_vec, rng, cfg)
        _, _, ite_true_cal, _ = make_semisynthetic(Xcal_s, ycal, d_vec, rng, cfg)
        model, q, rate_kl, dist, _ = conformal_intervals(
            Xtr_s, ytr, Xcal_s, ycal, ite_true_cal, cfg, seed=seed, ite_true_tr=ite_true_tr)
        g_sim, q_sim = sim_proxy_calibration(model, Xcal_s, ycal, d_vec, cfg, seed, lam, S_mode="rc")
        # independent direction + independent severity direction
        d0 = effect_beta(d_vec.shape[0], seed=seed + 31337)
        d_ind = d0 / (np.linalg.norm(d0) + 1e-12) * np.linalg.norm(d_vec)
        b0 = effect_beta(d_vec.shape[0], seed=seed + 2718)
        evals = [("PPMI-internal", Xte, yte, 0),
                 ("NEUROCON", ext["NEUROCON"][0], ext["NEUROCON"][1], 1000),
                 ("TaoWu", ext["TaoWu"][0], ext["TaoWu"][1], 1500)]
        for cname, Xe_raw, ye, off in evals:
            Xe_s = Z(Xe_raw)
            r_e, d_e, _ = proxy_components(model, Xe_s, ye)
            ite_hat = d_e
            n_e = len(Xe_s)
            half_sim = np.maximum(0.0, g_sim.predict(r_e) + q_sim)
            s_sev = 1.0 + 0.5 * np.tanh(Xe_s @ b0)
            true_e = s_sev * np.linalg.norm(d_ind)
            for meth, half in [("CPCI-Sim", half_sim),
                               ("RelInt", z_rel * ite_hat),
                               ("CPCI-Oracle", np.full(n_e, q))]:
                lo = np.maximum(0.0, ite_hat - half)
                hi = ite_hat + half
                m = (lo <= true_e) & (true_e <= hi)
                rows.append(dict(seed=seed, cohort=cname, method=meth,
                                 coverage=float(m.mean()),
                                 width=float((hi - lo).mean()),
                                 covered=int(m.sum()), n=n_e, rate_kl=rate_kl,
                                 mean_true=float(true_e.mean()),
                                 mean_ite_hat=float(ite_hat.mean())))
        print("[stress3] seed %d done (rate=%.1f)" % (seed, rate_kl))
    df = pd.DataFrame(rows)
    out = Path(cfg["out_root"]); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "stress3_results.csv", index=False)
    g = df.groupby(["cohort", "method"]).agg(
        coverage=("coverage", "mean"), width=("width", "mean"),
        mean_true=("mean_true", "mean"), mean_ite_hat=("mean_ite_hat", "mean"),
        seeds=("seed", "count")).reset_index()
    g.to_csv(out / "stress3_summary.csv", index=False)
    print("\n[stress3] summary  [independent effect mechanism]")
    print(g.round(4).to_string(index=False))


# ==================== Revision experiments (v7.6) ====================

def _train_generator_on_ppmi(cfg, seed):
    """Train the twin-VAE on the PPMI training fold of split `seed`
    (identical to the main experiment) and return (model, d_vec, mu, sd)."""
    Xp, yp, files_p = load_cohort(cfg, "PPMI")
    groups_p = np.array([_digits(f) for f in files_p])
    rng = np.random.RandomState(seed)
    tr, _, _ = grouped_train_cal_test(Xp, yp, groups_p, cfg["test_frac"],
                                      0.25, seed)
    Xtr, ytr = Xp[tr], yp[tr]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr_s = (Xtr - mu) / sd
    d_vec = Xtr_s[ytr == 1].mean(0) - Xtr_s[ytr == 0].mean(0)
    _, _, ite_true_tr, _ = make_semisynthetic(Xtr_s, ytr, d_vec, rng, cfg)
    model = train_twin_vae(Xtr_s, ytr, cfg, seed=seed, ite_true=ite_true_tr)
    return model, d_vec, mu, sd


def _cov_width(ite_hat, half, true_e):
    lo = np.maximum(0.0, ite_hat - half)
    hi = ite_hat + half
    return float(((lo <= true_e) & (true_e <= hi)).mean()), float((hi - lo).mean())


def _small_m_calibration(model, Xc, yc, cfg, seed):
    """m < 12 fallback for the deploy curve: no two-half subject split; g is
    fitted on all m calibration subjects (a pseudo-replication we flag in
    the paper; small-m cells are excluded from the headline claim)."""
    from sklearn.isotonic import IsotonicRegression
    rng = np.random.RandomState(seed + 7777)
    r_c, d_c, _ = proxy_components(model, Xc, yc)
    n = len(yc)
    n_sim = cfg.get("n_sim", 16)
    idx = np.tile(np.arange(n), n_sim)
    scales = rng.uniform(cfg["semi_scale_range"][0], cfg["semi_scale_range"][1],
                         size=len(idx))
    err = d_c[idx] * np.abs(1.0 - scales)
    g = IsotonicRegression(out_of_bounds="clip").fit(r_c[idx], err)
    lv_m = min(0.99, np.ceil((n_sim + 1) * (1 - cfg["alpha"])) / n_sim)
    u = np.array([np.quantile(err[idx == i] - g.predict(r_c[[i]])[0], lv_m)
                  for i in range(n)])
    lv = min(0.99, np.ceil((n + 1) * (1 - cfg["alpha"])) / n)
    return g, float(np.quantile(u, lv))


def cmd_deploy(cfg, args):
    """New-site onboarding curve (Scenario B of the deployment protocol).

    A generator trained on PPMI (per split) is deployed at an external
    cohort. The CPCI-Sim calibration layer (isotonic map g + subject-level
    conformal quantile, fidelity score) is re-fitted on m target-site
    subjects using ONLY observable quantities (r_cycle, Delta_hat); the
    generator is never retrained. Coverage is evaluated on the remaining
    target subjects, over `repeats` subsamples per seed. RelInt (no target
    calibration) and CPCI-Oracle deployed WITHOUT target recalibration are
    horizontal references.

    Output: deploy_curve_<target>.csv  -> Fig. deploy (paper Sec. 5).
    """
    target = args.target
    lam = float(cfg["lambda_proxy"])
    z_rel = 0.5 * (1.0 - cfg["alpha"]) * (cfg["semi_scale_range"][1]
             - cfg["semi_scale_range"][0])
    Xe_raw, ye, _ = load_cohort(cfg, target)
    n_t = len(ye)
    # clamp the calibration-size grid to the cohort: at least 10 subjects
    # must remain for evaluation (m=80/120 are only meaningful for PPMI)
    m_list = [m for m in args.m_list if 5 <= m <= n_t - 10]
    dropped = sorted(set(args.m_list) - set(m_list))
    if dropped:
        print("[deploy] %s has n=%d; dropped m in %s (need >=10 held-out "
              "subjects)" % (target, n_t, dropped))
    if not m_list:
        raise ValueError("no valid m left for %s (n=%d)" % (target, n_t))
    rows = []
    for seed in cfg["seeds"]:
        model, d_vec, mu, sd = _train_generator_on_ppmi(cfg, seed)
        Xe_s = (Xe_raw - mu) / sd
        rng_e = np.random.RandomState(seed + 1000)
        _, _, true_e, _ = make_semisynthetic(Xe_s, ye, d_vec, rng_e, cfg)
        r_e, d_e, _ = proxy_components(model, Xe_s, ye)
        ite_hat = d_e
        Xp, yp, files_p = load_cohort(cfg, "PPMI")
        groups_p = np.array([_digits(f) for f in files_p])
        rng = np.random.RandomState(seed)
        tr, cal, _ = grouped_train_cal_test(Xp, yp, groups_p,
                                            cfg["test_frac"], 0.25, seed)
        mu2, sd2 = Xp[tr].mean(0), Xp[tr].std(0) + 1e-8
        Xcal_s = (Xp[cal] - mu2) / sd2
        _, _, ite_true_cal, _ = make_semisynthetic(Xcal_s, yp[cal], d_vec,
                                                   rng, cfg)
        _, q_oracle, _, _, _ = conformal_intervals(
            (Xp[tr] - mu2) / sd2, yp[tr], Xcal_s, yp[cal], ite_true_cal,
            cfg, seed=seed, model=model)
        for rep in range(args.repeats):
            rs = np.random.RandomState(1000 * seed + rep)
            perm = rs.permutation(len(ye))
            for m in m_list:
                cal_i, test_i = perm[:m], perm[m:]
                if m >= 12:
                    g, q = sim_proxy_calibration(model, Xe_s[cal_i],
                                                 ye[cal_i], d_vec, cfg,
                                                 seed + rep, lam,
                                                 S_mode="rc")
                else:
                    g, q = _small_m_calibration(model, Xe_s[cal_i],
                                                ye[cal_i], cfg, seed + rep)
                half = np.maximum(0.0, g.predict(r_e[test_i]) + q)
                cov, wid = _cov_width(ite_hat[test_i], half, true_e[test_i])
                rows.append(dict(seed=seed, rep=rep, m=m, method="CPCI-Sim",
                                 n_cal=m, n_test=len(test_i),
                                 coverage=cov, width=wid))
                half_r = z_rel * ite_hat[test_i]
                cov, wid = _cov_width(ite_hat[test_i], half_r, true_e[test_i])
                rows.append(dict(seed=seed, rep=rep, m=m, method="RelInt",
                                 n_cal=0, n_test=len(test_i),
                                 coverage=cov, width=wid))
                half_o = np.full(len(test_i), q_oracle)
                cov, wid = _cov_width(ite_hat[test_i], half_o, true_e[test_i])
                rows.append(dict(seed=seed, rep=rep, m=m,
                                 method="CPCI-Oracle (no recal)",
                                 n_cal=0, n_test=len(test_i),
                                 coverage=cov, width=wid))
        print("[deploy] seed %d done" % seed)
    df = pd.DataFrame(rows)
    out = Path(cfg["out_root"]); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / ("deploy_curve_%s.csv" % target), index=False)
    g = df.groupby(["m", "method"]).agg(
        coverage=("coverage", "mean"), coverage_sd=("coverage", "std"),
        width=("width", "mean"), k=("seed", "count")).reset_index()
    print(g.round(4).to_string(index=False))
    print("saved -> %s" % (out / ("deploy_curve_%s.csv" % target)))
    print("    [feeds paper Fig. deploy (Scenario B information budget)]")


def cmd_balance(cfg, args):
    """Balanced sensitivity (review point: the 754-vs-42/40 imbalance).

    For each split, the internal TEST fold of the headline CPCI-Sim
    (fidelity score) is subsampled to --n-target subjects, --repeats times;
    coverage is recomputed from the stored intervals. No model is
    retrained - only the evaluation subset changes.

    Output: balance_internal.csv -> Sec. 4.2 sentence + Appendix C table.
    """
    lam = float(cfg["lambda_proxy"])
    rows = []
    for seed in cfg["seeds"]:
        model, d_vec, mu, sd = _train_generator_on_ppmi(cfg, seed)
        Xp, yp, files_p = load_cohort(cfg, "PPMI")
        groups_p = np.array([_digits(f) for f in files_p])
        rng = np.random.RandomState(seed)
        tr, cal, te = grouped_train_cal_test(Xp, yp, groups_p,
                                             cfg["test_frac"], 0.25, seed)
        Xte, yte = Xp[te], yp[te]
        Xte_s = (Xte - mu) / sd
        Xcal_s = (Xp[cal] - mu) / sd
        g_rc, q_rc = sim_proxy_calibration(model, Xcal_s, yp[cal], d_vec,
                                           cfg, seed, lam, S_mode="rc")
        _, _, true_full, _ = make_semisynthetic(Xte_s, yte, d_vec,
                                                np.random.RandomState(seed), cfg)
        r_e, d_e, _ = proxy_components(model, Xte_s, yte)
        ite_hat = d_e
        half = np.maximum(0.0, g_rc.predict(r_e) + q_rc)
        cov_full, _ = _cov_width(ite_hat, half, true_full)
        for rep in range(args.repeats):
            rs = np.random.RandomState(3000 * seed + rep)
            sub = rs.choice(len(yte), min(args.n_target, len(yte)),
                            replace=False)
            cov_b, wid_b = _cov_width(ite_hat[sub], half[sub], true_full[sub])
            rows.append(dict(seed=seed, rep=rep, n_sub=args.n_target,
                             coverage_balanced=cov_b, width_balanced=wid_b,
                             coverage_full=cov_full))
        print("[balance] seed %d done (full-fold cov %.4f)" % (seed, cov_full))
    df = pd.DataFrame(rows)
    out = Path(cfg["out_root"]); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "balance_internal.csv", index=False)
    print("full-fold coverage:   %.4f" % df["coverage_full"].mean())
    print("balanced (n=%d):      %.4f +- %.4f  [min %.3f, max %.3f]"
          % (args.n_target, df["coverage_balanced"].mean(),
             df["coverage_balanced"].std(), df["coverage_balanced"].min(),
             df["coverage_balanced"].max()))
    print("saved -> %s" % (out / "balance_internal.csv"))
    print("    [feeds paper Sec. 4.2 sentence + Appendix C tab:balance]")


# ==================== entry ====================# ==================== entry ====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "build", "main", "rate", "stress", "stress2", "stress3", "shift", "cdf", "sens", "deploy", "balance", "all"])
    ap.add_argument("--data-root", default=DEFAULT_CONFIG["data_root"])
    ap.add_argument("--out-root", default=DEFAULT_CONFIG["out_root"])
    ap.add_argument("--latent", type=int, default=None)
    ap.add_argument("--lam-rd", type=float, default=None)
    ap.add_argument("--lam-proxy", type=float, default=None,
                    help="weight lambda of the effect term in the proxy score (paper default 0.5)")
    ap.add_argument("--feat-root", default=None,
                    help="dir holding features/ from a previous extract run "
                         "(e.g. M:/MRI/Results_RDCCI); avoids re-running extract")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--target", default="NEUROCON",
                    help="deploy: external cohort treated as the new site")
    ap.add_argument("--m-list", type=int, nargs="+",
                    default=[5, 10, 20, 40, 80, 120],
                    help="deploy: target-site calibration-size grid")
    ap.add_argument("--n-target", type=int, default=42,
                    help="balance: subsampled internal test-fold size")
    ap.add_argument("--repeats", type=int, default=5,
                    help="deploy/balance: subsampling repetitions per seed")
    args = ap.parse_args()
    cfg = dict(DEFAULT_CONFIG)
    cfg["data_root"] = args.data_root
    cfg["out_root"] = args.out_root
    if args.latent:
        cfg["latent_dim"] = args.latent
    if args.lam_rd is not None:
        cfg["lambda_rd"] = args.lam_rd
    if args.lam_proxy is not None:
        cfg["lambda_proxy"] = args.lam_proxy
    if args.seeds:
        cfg["seeds"] = args.seeds
    if args.feat_root:
        cfg["feat_root"] = args.feat_root
    if args.cmd == "extract":
        cmd_extract(cfg)
    elif args.cmd == "build":
        cmd_build(cfg)
    elif args.cmd == "main":
        cmd_main(cfg)
    elif args.cmd == "rate":
        cmd_rate(cfg)
    elif args.cmd == "stress":
        cmd_stress(cfg)
    elif args.cmd == "stress2":
        cmd_stress2(cfg)
    elif args.cmd == "stress3":
        cmd_stress3(cfg)
    elif args.cmd == "shift":
        cmd_shift(cfg)
    elif args.cmd == "cdf":
        cmd_cdf(cfg)
    elif args.cmd == "sens":
        cmd_sens(cfg)
    elif args.cmd == "deploy":
        cmd_deploy(cfg, args)
    elif args.cmd == "balance":
        cmd_balance(cfg, args)
    elif args.cmd == "all":
        cmd_extract(cfg); cmd_main(cfg); cmd_rate(cfg)


if __name__ == "__main__":
    main()