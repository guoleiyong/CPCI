# -*- coding: utf-8 -*-
"""
RD-CCI MRI 实验：双队列外部验证 + Rate-Coverage 曲线（Theorem 1 实证）
=================================================================
数据布局（默认 M:/MRI，可用 --data-root 修改）：
  <root>/PPMI/     PPMI_*_T1w_MNI152_normalized.nii.gz + ppmi_subjects.csv + PPMI-UPDRS.csv
  <root>/NEUROCON/ sub-control*/sub-patient*_T1w_MNI152_normalized.nii.gz + neurocon_patients.tsv
  <root>/TaoWu/    sub-control*/sub-patient*_T1w_MNI152_normalized.nii.gz + taowu_patients.tsv
  <root>/aal_extracted/aal/atlas/AAL.nii + AAL.xml

用法：
  python rd_cci_mri_experiments.py extract          # 一次性特征提取 -> CSV
  python rd_cci_mri_experiments.py build            # 检查标签/汇总队列（不训练）
  python rd_cci_mri_experiments.py main             # 实验一：PPMI 内部 + NEUROCON/TaoWu 外部验证
  python rd_cci_mri_experiments.py rate             # 实验二：rate(KL)-coverage gap 曲线（Theorem 1）
  python rd_cci_mri_experiments.py all              # 全部

依赖：numpy pandas nibabel torch scikit-learn matplotlib tqdm（可选: neuroCombat）
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ==================== 配置 ====================

DEFAULT_CONFIG = {
    "data_root": "M:/MRI",
    "out_root": "M:/MRI/Results_RDCCI",
    "alpha": 0.1,               # 名义显著性 -> 90% 覆盖
    "lambda_score": 0.5,        # 非一致性分数: S = r_cycle + lambda*delta_effect
    "lambda_rd": 0.01,          # VAE 的 KL(rate) 权重
    "latent_dim": 32,
    "hidden_dim": 128,
    "lr": 1e-3,
    "epochs": 300,
    "patience": 40,
    "batch_size": 64,
    "seeds": [42, 43, 44, 45, 46],
    "val_frac": 0.15,           # PPMI 训练集内划出早停验证集
    "test_frac": 0.25,          # PPMI 留出测试集
    "semi_scale_range": (0.5, 1.5),   # 半合成个体效应缩放范围
    "semi_noise": 0.05,         # 半合成加性噪声（标准化空间）
}

PPMI_GROUP_MAP = {"pd": 1, "parkinson": 1, "healthy": 0, "control": 0, "hc": 0}


def grouped_train_cal_test(X, y, groups, test_frac, cal_frac_of_train, seed):
    """按受试者分组的三路划分，杜绝同一受试者跨集合泄露。
    cal_frac_of_train: 校准集占 (训练+校准) 的比例。"""
    from sklearn.model_selection import GroupShuffleSplit
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    tr_idx, te_idx = next(gss1.split(X, y, groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=cal_frac_of_train, random_state=seed)
    sub_tr, sub_cal = next(gss2.split(X[tr_idx], y[tr_idx], groups[tr_idx]))
    return tr_idx[sub_tr], tr_idx[sub_cal], te_idx


def wilson_ci(k, n, z=1.96):
    """覆盖率的 Wilson 分数置信区间"""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# ==================== 1. AAL 模板解析与特征提取 ====================

def parse_aal_xml(xml_path):
    """解析 AAL.xml -> [(index, name)]，按 index 排序。兼容多种格式：
    - <label index="2001">Precentral_L</label>
    - <label><index>1</index><name>...</name></label>
    - 无 index 的条目跳过；全部无 index 时按文档顺序从 1 编号。"""
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
        # name 本身是纯数字的情况（<label>2001</label>）
        try:
            pairs.append((int(name), name))
            any_index = True
        except (TypeError, ValueError):
            continue
    if not any_index:  # 全部无编号 -> 按文档顺序 1..N
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
        raise ValueError(f"AAL.xml 未解析出任何区域: {xml_path}")
    print(f"[atlas] xml 解析出 {len(pairs)} 个区域, 编号范围 {pairs[0][0]}..{pairs[-1][0]}")
    names = [n for _, n in pairs]
    if len(vals) == len(names):
        label_map = {v: names[i] for i, v in enumerate(vals)}
    else:  # xml index 即体素值（如 2001..9170 的旧编号）
        idx2name = dict(pairs)
        label_map = {v: idx2name.get(v, f"ROI{v}") for v in vals}
    return atlas, label_map


def extract_roi_features(img, adata, label_map, use_std=True):
    """对每个 AAL ROI 提取均值(+标准差)。img 为 nibabel 图像对象，adata 为与其同 shape 的
    重采样 atlas 标签数组。"""
    data = img.get_fdata().astype(np.float32)
    if data.shape != adata.shape:
        raise ValueError(f"shape mismatch: img {data.shape} vs atlas {adata.shape}")
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
    """返回函数: 给定 nibabel 图像 -> 最近邻重采样到该图像网格的 atlas 标签数组。
    同一网格只计算一次。"""
    import nibabel as nib
    cache = {}
    def resample_to(img):
        sh = tuple(img.shape)
        if sh not in cache:
            try:
                from nibabel.processing import resample_to_img
                r = resample_to_img(atlas_img, img, interpolation='nearest')
                cache[sh] = r.get_fdata().astype(np.int32)
            except Exception:
                # 兜底: 仿射为对角缩放时用 scipy 最近邻 zoom
                from scipy.ndimage import zoom
                f = [s / a for s, a in zip(img.shape, atlas_img.shape)]
                cache[sh] = zoom(atlas_img.get_fdata().astype(np.float32), f,
                                 order=0).round().astype(np.int32)
            vs = img.header.get_zooms()
            print(f"[atlas] 重采样到 {sh} (voxel ~{vs[0]:.2f}mm)")
        return cache[sh]
    return resample_to


def cmd_extract(cfg):
    import nibabel as nib
    from tqdm import tqdm
    root = Path(cfg["data_root"])
    out = Path(cfg["out_root"]) / "features"
    out.mkdir(parents=True, exist_ok=True)
    atlas, label_map = build_label_map(
        root / "aal_extracted" / "aal" / "atlas" / "AAL.nii",
        root / "aal_extracted" / "aal" / "atlas" / "AAL.xml")
    print(f"[extract] atlas labels: {len(label_map)}")

    resample = make_resampler(atlas)
    cohorts = {
        "PPMI": root / "PPMI",
        "NEUROCON": root / "NEUROCON",
        "TaoWu": root / "TaoWu",
    }
    for cname, cdir in cohorts.items():
        files = sorted(cdir.glob("*_T1w_MNI152_normalized.nii.gz"))
        if not files:
            print(f"[extract] {cname}: no images found in {cdir}")
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
        df = pd.DataFrame(rows, columns=["file"] + (names_ref or []))
        df.to_csv(out / f"{cname}_roi_features.csv", index=False)
        print(f"[extract] {cname}: {len(df)} ok, {len(bad)} failed -> {out / (cname + '_roi_features.csv')}")
        for b in bad[:5]:
            print("   failed:", b)


# ==================== 2. 队列标签与数据集构建 ====================

def _norm_prefix(name):
    n = str(name).lower()
    if "patient" in n or n.startswith("pd"):
        return 1
    if "control" in n or n.startswith("hc") or "normal" in n:
        return 0
    return -1


def _digits(s):
    """从 'PPMI_172821' / '172821.0' / 172821 中统一提取数字串。"""
    import re
    m = re.search(r"(\d+)", str(s))
    return m.group(1) if m else ""


def load_ppmi_labels(cfg):
    root = Path(cfg["data_root"])
    subj = pd.read_csv(root / "PPMI" / "ppmi_subjects.csv")
    id_col = next((c for c in subj.columns if c.strip().lower() in ("subject", "subject_id", "patno")), None)
    grp_col = next((c for c in subj.columns if "group" in c.lower() or "research" in c.lower()), None)
    if id_col is None or grp_col is None:
        raise ValueError(f"ppmi_subjects.csv 中找不到 id/group 列，现有列: {list(subj.columns)}")
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
        if y >= 0 and key and key not in seen:  # 同一受试者多行(多次访问)只取首次
            lab[key] = y
            seen.add(key)
    return lab


_FEATURE_MASK_CACHE = {}
_LAST_UNMATCHED = []


def get_feature_mask(cfg):
    """跨队列零值过滤: 任一队列零值率>5%的ROI特征剔除(颅骨剥离边缘伪影)。
    结果缓存到 <out_root>/feature_filter.json, 并保存保留特征名列表供论文报告。"""
    key = cfg["out_root"]
    if key in _FEATURE_MASK_CACHE:
        return _FEATURE_MASK_CACHE[key]
    cache_file = Path(cfg["out_root"]) / "feature_filter.json"
    if cache_file.exists():
        keep = np.array(json.loads(cache_file.read_text(encoding="utf-8"))["keep"], dtype=bool)
    else:
        zero_max, cols_ref = None, None
        for cname in ["PPMI", "NEUROCON", "TaoWu"]:
            fcsv = Path(cfg["out_root"]) / "features" / f"{cname}_roi_features.csv"
            df = pd.read_csv(fcsv)
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
        print(f"[filter] ROI 特征: 保留 {int(keep.sum())}/{len(keep)} "
              f"(剔除 {int((~keep).sum())} 个跨队列零值率>5%的特征)")
    _FEATURE_MASK_CACHE[key] = keep
    return keep


def load_cohort(cfg, cname):
    """返回 X (n,d), y (n,), files (n,)"""
    root = Path(cfg["data_root"])
    feat_csv = Path(cfg["out_root"]) / "features" / f"{cname}_roi_features.csv"
    df = pd.read_csv(feat_csv)
    files = df["file"].astype(str).tolist()
    feat_cols = [c for c in df.columns if c != "file"]
    X = df[feat_cols].to_numpy(dtype=np.float32)
    mask = get_feature_mask(cfg)  # 跨队列零值过滤
    X = X[:, mask]

    if cname == "PPMI":
        lab = load_ppmi_labels(cfg)
        y = np.array([lab.get(_digits(f), -1) for f in files])
        global _LAST_UNMATCHED
        _LAST_UNMATCHED = [f for f, yy in zip(files, y) if yy < 0]
    else:
        y = np.array([_norm_prefix(f) for f in files])
        # 与 tsv 交叉校验
        tsv = root / cname / f"{cname.lower()}_patients.tsv"
        if tsv.exists():
            meta = pd.read_csv(tsv, sep="\t")
            name_col = meta.columns[0]
            stat_col = next((c for c in meta.columns if c.lower() in ("status",)), None)
            hy_col = next((c for c in meta.columns if c.lower().replace("&", "").replace("_", "") in ("hy", "hoehnyahr")), None)
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
                # H&Y 校验：患者应 >= 1，对照应为空/0
                if hy_col is not None:
                    try:
                        hy = float(r[hy_col])
                        if hy >= 1:
                            y[i] = max(y[i], 1)
                    except (TypeError, ValueError):
                        pass
    keep = y >= 0
    return X[keep], y[keep], np.array(files)[keep]


def cmd_build(cfg):
    for cname in ["PPMI", "NEUROCON", "TaoWu"]:
        try:
            feat_csv = Path(cfg["out_root"]) / "features" / f"{cname}_roi_features.csv"
            n_files = len(pd.read_csv(feat_csv)) if feat_csv.exists() else 0
            X, y, files = load_cohort(cfg, cname)
            unmatched = n_files - len(X)
            print(f"[build] {cname}: n={len(X)} (PD={int(y.sum())}, HC={int((y==0).sum())}), dim={X.shape[1]} (已过滤)"
                  + (f", {unmatched} 个文件未匹配到标签!" if unmatched else ""))
            if cname == "PPMI" and _LAST_UNMATCHED:
                print("   未匹配:", _LAST_UNMATCHED[:5])
        except FileNotFoundError as e:
            print(f"[build] {cname}: 特征文件缺失，请先运行 extract  ({e})")


# ==================== 3. Twin-VAE 反事实生成器 ====================

import torch
import torch.nn as nn


class TwinVAE(nn.Module):
    """编码器 E(X,T)->z；解码器 D(z,T')->X'。反事实 = D(E(X,T), 1-T)。"""
    def __init__(self, d_in, latent=32, hidden=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d_in + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
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
    """冻结的组分类器: 用于反事实一致性损失(BCE(C(x_cf), 1-T))。"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed + 999)
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    yt = torch.tensor(y, dtype=torch.float32, device=dev).unsqueeze(1)
    clf = nn.Sequential(
        nn.Linear(X.shape[1], 64), nn.ReLU(), nn.Linear(64, 1)).to(dev)
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


def train_twin_vae(X, y, cfg, latent=None, lam_rd=None, seed=42, verbose=False):
    """Twin-VAE + 反事实一致性: 重建MSE + lam_rd*KL + gamma_cf*BCE(C(x_cf), 1-T)。"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    latent = latent or cfg["latent_dim"]
    lam_rd = lam_rd if lam_rd is not None else cfg["lambda_rd"]
    gamma_cf = cfg.get("gamma_cf", 0.5)
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr = torch.tensor(X, dtype=torch.float32, device=dev)
    ytr = torch.tensor(y, dtype=torch.float32, device=dev).unsqueeze(1)
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
            cf_loss = bce(clf(cf), 1 - ytr[b])   # 反事实应被判为目标组
            loss = mse(Xtr[b], recon) + lam_rd * kl + gamma_cf * cf_loss
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
        print(f"    [vae] latent={latent} lam_rd={lam_rd} epochs={ep+1} val_recon={best:.4f}")
    return model


@torch.no_grad()
def vae_outputs(model, X, y):
    dev = next(model.parameters()).device
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    yt = torch.tensor(y, dtype=torch.float32, device=dev).unsqueeze(1)
    mu, logvar = model.encode(Xt, yt)
    recon = model.decode(mu, yt).cpu().numpy()
    x_cf = model.decode(mu, 1 - yt).cpu().numpy()
    x0 = model.decode(mu, torch.zeros_like(yt)).cpu().numpy()
    x1 = model.decode(mu, torch.ones_like(yt)).cpu().numpy()
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)).cpu().numpy()
    return recon, x_cf, x0, x1, kl


# ==================== 4. 半合成反事实真值 ====================

def make_semisynthetic(X, y, d_vec, rng, cfg):
    """
    已知真值的反事实效应：效应方向 d_vec（PPMI 训练集估计），个体缩放 s_i~U(a,b)。
      HC(t=0): Y1 = X + s_i*d + noise, 观测 Y0 = X, 真值 ITE = ||s_i*d||
      PD(t=1): Y0 = X - s_i*d + noise, 观测 Y1 = X, 真值 ITE = ||s_i*d||
    返回 (X_obs, t, true_ITE, Y_cf_true)
    """
    a, b = cfg["semi_scale_range"]
    s = rng.uniform(a, b, size=len(X))
    delta = s[:, None] * d_vec[None, :]
    noise = rng.normal(0, cfg["semi_noise"], size=X.shape)
    sign = np.where(y == 1, -1.0, 1.0)[:, None]
    y_cf_true = X + sign * delta + noise
    true_ite = np.linalg.norm(delta, axis=1)
    return X, y, true_ite, y_cf_true


# ==================== 5. Conformal 流程 ====================

def conformal_intervals(Xtr, ytr, Xcal, ycal, true_cal, cfg, seed=42, model=None,
                        latent=None, lam_rd=None):
    """训练 -> 校准 -> 区间。
    校准分数 = 半合成真值的精确误差 S=|ITE_true-ITE_hat| (split conformal 标准形式;
    域内覆盖≈名义值是数学保证, 区分度体现在区间宽度与外部漂移下的覆盖退化)。
    返回 (model, q, rate=平均KL, distortion=重建MSE, ite_hat_cal)。"""
    if model is None:
        model = train_twin_vae(Xtr, ytr, cfg, latent=latent, lam_rd=lam_rd, seed=seed)
    recon, _, x0, x1, kl_cal = vae_outputs(model, Xcal, ycal)
    ite_hat_cal = np.linalg.norm(x1 - x0, axis=1)
    S = np.abs(true_cal - ite_hat_cal)
    n = len(S)
    q = np.quantile(S, min(0.99, np.ceil((n + 1) * (1 - cfg["alpha"])) / n))
    dist = float(np.mean((Xcal - recon) ** 2))
    return model, q, float(np.mean(kl_cal)), dist, ite_hat_cal


def evaluate_on(model, q, X, y, cfg):
    recon, _, x0, x1, _ = vae_outputs(model, X, y)
    ite_hat = np.linalg.norm(x1 - x0, axis=1)
    lo = np.maximum(0, ite_hat - q)
    hi = ite_hat + q
    return lo, hi, ite_hat, recon


# ==================== 实验一：PPMI 内部 + NEUROCON/TaoWu 外部 ====================

def cmd_main(cfg):
    rng_master = np.random.RandomState(7)
    cohorts = {}
    for c in ["PPMI", "NEUROCON", "TaoWu"]:
        X, y, _ = load_cohort(cfg, c)
        cohorts[c] = (X, y)

    Xp, yp = cohorts["PPMI"]
    Xp_all, yp_all, files_p = load_cohort(cfg, "PPMI")
    groups_p = np.array([_digits(f) for f in files_p])  # 受试者级分组
    results = []
    t0 = time.time()
    for seed in cfg["seeds"]:
        rng = np.random.RandomState(seed)
        tr, cal, te = grouped_train_cal_test(Xp_all, yp_all, groups_p,
                                             cfg["test_frac"], 0.25, seed)
        Xtr, ytr, Xcal, ycal, Xte, yte = Xp_all[tr], yp_all[tr], Xp_all[cal], yp_all[cal], Xp_all[te], yp_all[te]

        # 标准化（仅用训练集统计量，外部队列套用）
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Z = lambda A: (A - mu) / sd
        Xtr_s, Xcal_s, Xte_s = Z(Xtr), Z(Xcal), Z(Xte)

        # 半合成效应方向（仅训练集，防泄露；真实量级，不再放大）
        d_vec = (Xtr_s[ytr == 1].mean(0) - Xtr_s[ytr == 0].mean(0))

        # 校准集半合成真值（精确误差校准, split conformal 标准形式）
        _, _, ite_true_cal, _ = make_semisynthetic(Xcal_s, ycal, d_vec, rng, cfg)
        model, q, rate_kl, dist, _ = conformal_intervals(
            Xtr_s, ytr, Xcal_s, ycal, ite_true_cal, cfg, seed=seed)

        # PPMI 测试集（半合成真值；域内覆盖≈名义值是 conformal 的数学保证, 作 sanity check）
        _, _, ite_true_semi, _ = make_semisynthetic(
            Xte_s, yte, d_vec, rng, cfg)
        lo, hi, ite_hat, _ = evaluate_on(model, q, Xte_s, yte, cfg)
        cov = ((lo <= ite_true_semi) & (ite_true_semi <= hi)).mean()
        results.append(dict(seed=seed, cohort="PPMI-internal", n=len(Xte),
                            coverage=cov, width=float((hi - lo).mean()),
                            mae=float(np.abs(ite_hat - ite_true_semi).mean()),
                            covered=int(((lo <= ite_true_semi) & (ite_true_semi <= hi)).sum()),
                            rate_kl=rate_kl, distortion=dist))

        # 外部队列
        for cname in ["NEUROCON", "TaoWu"]:
            Xe, ye = cohorts[cname]
            Xe_s = Z(Xe)
            rng_e = np.random.RandomState(seed + 1000)
            _, _, ite_true_e, _ = make_semisynthetic(Xe_s, ye, d_vec, rng_e, cfg)
            lo, hi, ite_hat_e, _ = evaluate_on(model, q, Xe_s, ye, cfg)
            cov = ((lo <= ite_true_e) & (ite_true_e <= hi)).mean()
            results.append(dict(seed=seed, cohort=cname, n=len(Xe),
                                coverage=cov, width=float((hi - lo).mean()),
                                mae=float(np.abs(ite_hat_e - ite_true_e).mean()),
                                covered=int(((lo <= ite_true_e) & (ite_true_e <= hi)).sum()),
                                rate_kl=rate_kl, distortion=dist))
        print(f"[main] seed {seed} done ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(results)
    out = Path(cfg["out_root"]); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "main_results_per_seed.csv", index=False)

    print("\n===== 汇总（均值 ± 标准差，覆盖率为所有受试者加权）=====")
    rows = []
    for cname, g in df.groupby("cohort"):
        k, n = g["covered"].sum(), g["n"].sum()
        lo_w, hi_w = wilson_ci(int(k), int(n))
        rows.append(dict(cohort=cname, mean_coverage=g["coverage"].mean(),
                         coverage_ci=f"[{lo_w:.3f},{hi_w:.3f}]",
                         mean_width=g["width"].mean(), mean_mae=g["mae"].mean(),
                         seeds=len(g)))
        print(f"{cname:15s} coverage={g['coverage'].mean():.3f} (CI {lo_w:.3f}-{hi_w:.3f})  "
              f"width={g['width'].mean():.3f}  ITE_MAE={g['mae'].mean():.3f}")
    pd.DataFrame(rows).to_csv(out / "main_results_summary.csv", index=False)
    print(f"\n结果已保存至 {out}")


# ==================== 实验二：Rate(KL)–Coverage gap 曲线 ====================

def cmd_rate(cfg):
    cohorts = {}
    for c in ["PPMI", "NEUROCON", "TaoWu"]:
        X, y, _ = load_cohort(cfg, c)
        cohorts[c] = (X, y)
    Xp, yp = cohorts["PPMI"]
    _, _, files_p = load_cohort(cfg, "PPMI")
    groups_p = np.array([_digits(f) for f in files_p])
    rows = []
    lam_grid = [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1]
    for lam in lam_grid:
        for seed in cfg["seeds"]:
            rng = np.random.RandomState(seed)
            tr, cal, te = grouped_train_cal_test(Xp, yp, groups_p,
                                                 cfg["test_frac"], 0.25, seed)
            Xtr, ytr, Xcal, ycal, Xte, yte = Xp[tr], yp[tr], Xp[cal], yp[cal], Xp[te], yp[te]
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
            Z = lambda A: (A - mu) / sd
            Xtr_s, Xcal_s, Xte_s = Z(Xtr), Z(Xcal), Z(Xte)
            d_vec = (Xtr_s[ytr == 1].mean(0) - Xtr_s[ytr == 0].mean(0))

            _, _, ite_true_cal, _ = make_semisynthetic(Xcal_s, ycal, d_vec, rng, cfg)
            model, q, rate_kl, dist, _ = conformal_intervals(
                Xtr_s, ytr, Xcal_s, ycal, ite_true_cal, cfg, seed=seed, lam_rd=lam)

            rec = dict(lam_rd=lam, seed=seed, rate_kl=rate_kl, distortion=dist)
            # 域内: 宽度(效率) + 覆盖(sanity check)
            _, _, ite_true, _ = make_semisynthetic(Xte_s, yte, d_vec, rng, cfg)
            lo, hi, _, _ = evaluate_on(model, q, Xte_s, yte, cfg)
            rec["width_internal"] = float((hi - lo).mean())
            rec["cov_internal"] = float(((lo <= ite_true) & (ite_true <= hi)).mean())
            # 外部: 覆盖退化 = Theorem 1 的 coverage gap 读数
            for ename in ["NEUROCON", "TaoWu"]:
                Xe, ye = cohorts[ename]
                rng_e = np.random.RandomState(seed + 1000)
                _, _, ite_true_e, _ = make_semisynthetic(Z(Xe), ye, d_vec, rng_e, cfg)
                lo, hi, _, _ = evaluate_on(model, q, Z(Xe), ye, cfg)
                cov_e = float(((lo <= ite_true_e) & (ite_true_e <= hi)).mean())
                rec[f"cov_{ename}"] = cov_e
                rec[f"gap_{ename}"] = abs(cov_e - (1 - cfg["alpha"]))
            rows.append(rec)
            print(f"[rate] lam={lam} seed={seed} rate={rate_kl:.2f} nats  "
                  f"w={rec['width_internal']:.3f}  gap_NE={rec['gap_NEUROCON']:.3f}  gap_TW={rec['gap_TaoWu']:.3f}")

    df = pd.DataFrame(rows)
    out = Path(cfg["out_root"]); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "rate_curve.csv", index=False)

    # 汇总: rate - distortion(效率前沿) + 外部覆盖缺口(Theorem 1 实证)
    g = df.groupby("lam_rd").agg(
        rate_kl=("rate_kl", "mean"), distortion=("distortion", "mean"),
        width=("width_internal", "mean"), cov_int=("cov_internal", "mean"),
        gap_NE=("gap_NEUROCON", "mean"), gap_TW=("gap_TaoWu", "mean")).reset_index()
    g.to_csv(out / "rate_curve_summary.csv", index=False)
    print("\n[rate] 汇总:"); print(g.round(4).to_string(index=False))

    fits = {}
    for cname, gcol in [("NEUROCON", "gap_NE"), ("TaoWu", "gap_TW")]:
        gg = g[g[gcol] > 1e-4]
        if len(gg) >= 3:
            slope, intercept = np.polyfit(np.log(gg["rate_kl"]), np.log(gg[gcol]), 1)
            fits[cname] = (slope, intercept)
            print(f"[rate] {cname}: gap ∝ rate^{slope:.3f}")

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        axes[0].loglog(g["rate_kl"], g["distortion"], "s-", color="tab:red")
        axes[0].set_xlabel("Rate  E[KL(q(z|x)||p(z))] (nats)")
        axes[0].set_ylabel("Distortion (recon MSE)")
        axes[0].set_title("(A) Rate-Distortion Frontier")
        axes[0].grid(True, alpha=0.3)
        for cname, gcol, mk in [("NEUROCON", "gap_NE", "o"), ("TaoWu", "gap_TW", "^")]:
            axes[1].loglog(g["rate_kl"], g[gcol], mk + "-", label=cname)
            if cname in fits:
                s, b = fits[cname]
                xs = np.array([g["rate_kl"].min(), g["rate_kl"].max()])
                axes[1].loglog(xs, np.exp(b) * xs ** s, "--", alpha=0.6,
                               label=f"{cname} fit (exp={s:.2f})")
        axes[1].set_xlabel("Rate (nats)")
        axes[1].set_ylabel("Coverage gap under shift")
        axes[1].set_title("(B) Rate-Coverage Gap (Theorem 1)")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "rate_coverage_curve.png", dpi=300)
        print(f"[rate] 图已保存 {out / 'rate_coverage_curve.png'}")
    except Exception as e:
        print("plot skipped:", e)


# ==================== 入口 ====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "build", "main", "rate", "all"])
    ap.add_argument("--data-root", default=DEFAULT_CONFIG["data_root"])
    ap.add_argument("--out-root", default=DEFAULT_CONFIG["out_root"])
    ap.add_argument("--latent", type=int, default=None)
    ap.add_argument("--lam-rd", type=float, default=None)
    args = ap.parse_args()
    cfg = dict(DEFAULT_CONFIG)
    cfg["data_root"] = args.data_root
    cfg["out_root"] = args.out_root
    if args.latent: cfg["latent_dim"] = args.latent
    if args.lam_rd is not None: cfg["lambda_rd"] = args.lam_rd

    if args.cmd == "extract":
        cmd_extract(cfg)
    elif args.cmd == "build":
        cmd_build(cfg)
    elif args.cmd == "main":
        cmd_main(cfg)
    elif args.cmd == "rate":
        cmd_rate(cfg)
    elif args.cmd == "all":
        cmd_extract(cfg); cmd_main(cfg); cmd_rate(cfg)


if __name__ == "__main__":
    main()