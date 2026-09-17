#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPCI v2 —— 防泄漏加权因果保形预测 (Leakage-Free Weighted Causal Conformal Prediction)

====================================================================
相对 v1 的改进清单（对应论文方法学贡献）
====================================================================
[修复1] evaluate() 不再使用测试集真实ITE设定区间宽度（消除标签泄漏）。
        区间宽度严格由 calibrate() 在校准集上估计；测试标签仅用于计算
        评估指标（覆盖率、宽度、FSC等）。
[修复2] "CycleGAN" 更名为 CCAE (Counterfactual Cycle-consistent
        AutoEncoder)。原实现无判别器、无对抗损失，如实命名。
[新增3] PropensityModel: 倾向得分 IPW 加权校准，处理治疗选择偏差。
[新增4] DensityRatioModel: 协变量偏移密度比加权（外部验证场景，
        Tibshirani et al. 2019 加权保形）。
[新增5] Mondrian 组条件校准：按组分别估计分位数（组条件覆盖对照）。
[新增6] 强基线：NaiveSplitConformal / CQR-S-learner /
        WeightedConformal(Tibshirani)，与原消融分开。
[新增7] 混杂半合成 DGP：治疗分配依赖协变量 e(X)，使 IPW 加权成为
        必要而非装饰。ITE 为非线性函数。
[新增8] FSC(最差组覆盖) / CovGap(组覆盖差距) 公平性指标。
[新增9] 多随机种子 mean±std 报告 + metrics_summary.csv。
[修复10] 可视化全部使用真实评估输出，删除 v1 中 np.random 伪造的
         残差/校准曲线/CI。

注意：本代码使用半合成协议（calibration 阶段使用模拟的 true ITE），
与 Lei & Candès (2021)、Alaa et al. (2023) 的因果保形基准一致，
论文中须如实报告为 semi-synthetic benchmark。
====================================================================
"""

import os
import sys
import json
import argparse
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy import stats as sp_stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics.pairwise import rbf_kernel

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# =====================================================================
# 配置
# =====================================================================
@dataclass
class CPCIv2Config:
    """CPCI v2 配置"""
    # 数据路径（按本机实际情况修改）
    audio_hc_dir: str = "M:/Audio/HC"
    audio_pd_dir: str = "M:/Audio/PD"
    mri_neurocon_dir: str = "M:/MRI/NEUROCON"
    mri_taowu_dir: str = "M:/MRI/TaoWu"
    mri_ppmi_dir: str = "M:/MRI/PPMI"
    output_dir: str = "./CPCI_v2_Results"

    # 保形参数
    alpha: float = 0.1                    # 目标 miscoverage，覆盖率 1-alpha
    lambda_score: float = 0.3             # 反事实分数权重: s = r + lambda_score * S_cf
    lambda_effect: float = 0.3            # S_cf = r_cycle + lambda_effect * delta_rkhs
    eta_uncertainty: float = 0.15         # 区间宽度中集成不确定性项的系数
    rbf_gamma: float = 0.001
    nystrom_landmarks: int = 200
    residual_clip_quantile: float = 0.99  # 校准残差截断（防极端值主导分位数）

    # 倾向得分 / 权重
    use_ipw: bool = True                  # IPW 加权校准（治疗选择偏差）
    use_shift_weight: bool = True         # 协变量偏移密度比加权（外部验证）
    crossfit_k: int = 5                   # 密度比模型的 K 折交叉拟合折数（防 in-sample 过拟合）
    nuisance_model: str = "auto"          # nuisance: "auto"(n>=200用GBM,否则LR) | "logistic" | "gbm"
    ihdp_path: str = "./ihdp_npci_1.csv"  # IHDP 基准（期刊补充的标准 benchmark）
    ipw_clip: Tuple[float, float] = (0.05, 20.0)

    # Mondrian 组条件校准（组 = 诊断标签，预测时可观测的属性）
    use_mondrian: bool = False            # True 时按组分别估计 alpha（消融用）

    # 数据划分
    train_ratio: float = 0.5
    cal_ratio: float = 0.3
    test_ratio: float = 0.2
    n_seeds: int = 10                     # 随机种子数（投稿建议 >= 10）

    # 混杂半合成 DGP
    confounding_strength: float = 1.0     # 倾向分配对协变量的依赖强度
    noise_level_audio: float = 0.10
    noise_level_mri: float = 0.15
    noise_level_external: float = 0.30    # 外部站点 ITE 噪声（模拟站点差异）

    # 反事实生成器
    cf_constraint_factor: float = 2.0

    random_state: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================================
# 评估指标工具（纯函数，输入为预测区间与真实值，不做任何自适应调整）
# =====================================================================
def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """覆盖率的 Wilson score 置信区间"""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def evaluate_intervals(lower: np.ndarray, upper: np.ndarray,
                       true_ites: np.ndarray, groups: np.ndarray,
                       ite_pred: np.ndarray) -> Dict[str, Any]:
    """
    纯评估函数：只计算指标，绝不根据 true_ites 调整区间宽度。

    输入的 lower/upper 必须由 predict_intervals() 提前生成。
    """
    lower = np.asarray(lower); upper = np.asarray(upper)
    true_ites = np.asarray(true_ites); groups = np.asarray(groups)

    covered = (true_ites >= lower) & (true_ites <= upper)
    coverage = float(np.mean(covered))
    ci_low, ci_high = wilson_ci(int(covered.sum()), len(covered))

    width = upper - lower
    widths_mean = float(np.mean(width))
    widths_std = float(np.std(width))

    residuals = np.abs(ite_pred - true_ites)

    # 组条件覆盖
    subgroup_stats: Dict[str, Dict[str, Any]] = {}
    group_coverages = []
    for g in np.unique(groups):
        mask = groups == g
        if mask.sum() == 0:
            continue
        g_cov = float(np.mean(covered[mask]))
        group_coverages.append(g_cov)
        subgroup_stats[f"group_{g}"] = {
            "coverage": g_cov,
            "n": int(mask.sum()),
            "width_mean": float(np.mean(width[mask]))
        }

    fsc = float(np.min(group_coverages)) if group_coverages else coverage
    cov_gap = float(np.max(group_coverages) - np.min(group_coverages)) \
        if len(group_coverages) > 1 else 0.0

    return {
        "coverage": coverage,
        "coverage_ci": (ci_low, ci_high),
        "width_mean": widths_mean,
        "width_std": widths_std,
        "width_median": float(np.median(width)),
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "fsc": fsc,                      # 最差组覆盖 (worst-group coverage)
        "cov_gap": cov_gap,              # 组覆盖差距
        "subgroup_stats": subgroup_stats,
        "n_test": len(true_ites),
        # 以下数组供可视化使用（真实评估输出，非模拟）
        "_arrays": {
            "lower": lower, "upper": upper,
            "widths": width, "covered": covered,
            "residuals": residuals, "ite_pred": ite_pred,
            "true_ites": true_ites, "groups": groups
        }
    }


def coverage_at_nominal(alpha_hat_score: np.ndarray,
                        true_test: np.ndarray,
                        ite_pred: np.ndarray,
                        base_half: np.ndarray,
                        nominals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    真实校准曲线：对一组名义覆盖率水平，用【校准集分数的加权分位数】
    重新构造区间（不使用测试标签），再计算测试集实际覆盖率。

    返回 (nominals, observed_coverages)。标签只用于最后的指标计算。
    """
    n_cal = len(alpha_hat_score)
    # conformal 有限样本修正: 取 ceil((n+1)*gamma)/n 经验分位
    observed = []
    for gamma in nominals:
        level = min(1.0, (np.ceil((n_cal + 1) * gamma)) / n_cal)
        q = float(np.quantile(alpha_hat_score, level))
        half = base_half + q * 0.0 + q  # half width 由校准分数分位数决定
        lower = ite_pred - half
        upper = ite_pred + half
        observed.append(float(np.mean((true_test >= lower) & (true_test <= upper))))
    return np.asarray(nominals), np.asarray(observed)


def weighted_quantile(values: np.ndarray, quantile: float,
                      weights: Optional[np.ndarray] = None) -> float:
    """加权分位数（无权重时退化为普通分位数）"""
    values = np.asarray(values, dtype=float)
    if weights is None:
        return float(np.quantile(values, quantile))
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    v_sorted, w_sorted = values[order], weights[order]
    cw = np.cumsum(w_sorted)
    total = cw[-1]
    if total <= 0:
        return float(np.quantile(values, quantile))
    target = quantile * total
    idx = np.searchsorted(cw, target)
    idx = min(idx, len(v_sorted) - 1)
    return float(v_sorted[idx])


# =====================================================================
# 数据加载（与 v1 相同）
# =====================================================================
class DataLoader:
    """音频 MFCC + MRI ROI 特征加载 + IHDP 基准"""

    def __init__(self, config: CPCIv2Config):
        self.config = config

    def load_ihdp(self, path: str
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """IHDP 基准 (Hill 2011; Dorie et al. 2019 的 NPCI 半合成版本)。

        CSV 无表头，每行: t, x0..x24(25维), y_factual, y_cf, mu0, mu1。
        真实婴儿健康协变量 + 真实(非随机)治疗分配 + 真实观测结局 y_factual；
        真值 ITE = mu1 - mu0 仅用于校准与评估（同一防泄漏协议）。
        """
        df = pd.read_csv(path, header=None)
        if df.shape[1] != 30:
            raise ValueError(f"IHDP CSV 列数应为 30，实际 {df.shape[1]}")
        t = df.iloc[:, 0].values.astype(int)
        X = df.iloc[:, 1:26].values.astype(float)
        y = df.iloc[:, 26].values.astype(float)          # y_factual（真实观测结局）
        true_ite = (df.iloc[:, 29] - df.iloc[:, 28]).values.astype(float)  # mu1 - mu0
        return X, y, t, true_ite

    def load_audio_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        import librosa
        X_list, y_list, ids_list = [], [], []

        for path, label in [(self.config.audio_hc_dir, 0),
                            (self.config.audio_pd_dir, 1)]:
            if os.path.exists(path):
                for file in Path(path).glob("*.wav"):
                    try:
                        y_audio, sr = librosa.load(file, sr=16000)
                        mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
                        X_list.append(mfcc.mean(axis=1))
                        y_list.append(label)
                        ids_list.append(file.stem)
                    except Exception as e:
                        logger.warning(f"加载失败 {file}: {e}")

        if not X_list:
            return np.array([]), np.array([]), np.array([])
        return np.array(X_list), np.array(y_list), np.array(ids_list)

    def load_mri_data(self, include_ppmi: bool = False
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        import nibabel as nib
        X_list, y_list, ids_list, sources_list = [], [], [], []

        for path in [self.config.mri_neurocon_dir, self.config.mri_taowu_dir]:
            if os.path.exists(path):
                for file in Path(path).glob("*.nii*"):
                    try:
                        data = nib.load(file).get_fdata()
                        if data.size == 0:
                            continue
                        feats = self._extract_roi_features(data)
                        if len(feats) == 0:
                            continue
                        X_list.append(feats)
                        y_list.append(0 if 'control' in file.name.lower() else 1)
                        ids_list.append(file.stem.split('_')[0])
                        sources_list.append(0)
                    except Exception as e:
                        if "CRC" not in str(e):
                            logger.warning(f"加载失败 {file}: {e}")

        if include_ppmi and os.path.exists(self.config.mri_ppmi_dir):
            for file in Path(self.config.mri_ppmi_dir).glob("*.nii*"):
                try:
                    data = nib.load(file).get_fdata()
                    if data.size == 0:
                        continue
                    feats = self._extract_roi_features(data)
                    if len(feats) == 0:
                        continue
                    X_list.append(feats)
                    y_list.append(1)
                    ids_list.append(file.stem)
                    sources_list.append(1)
                except Exception as e:
                    if "CRC" not in str(e):
                        logger.warning(f"加载失败 {file}: {e}")

        if not X_list:
            return (np.array([]), np.array([]), np.array([]), np.array([]))
        return (np.array(X_list), np.array(y_list),
                np.array(ids_list), np.array(sources_list))

    @staticmethod
    def _extract_roi_features(data: np.ndarray) -> np.ndarray:
        if data.ndim != 3:
            return np.array([])
        center = tuple(s // 2 for s in data.shape)
        roi_size = 20
        roi = data[
            max(0, center[0] - roi_size):center[0] + roi_size,
            max(0, center[1] - roi_size):center[1] + roi_size,
            max(0, center[2] - roi_size):center[2] + roi_size
        ]
        if roi.size == 0:
            return np.array([])
        return np.array([
            np.mean(roi), np.std(roi), np.median(roi),
            np.percentile(roi, 25), np.percentile(roi, 75),
            np.min(roi), np.max(roi)
        ])


# =====================================================================
# 反事实循环一致自编码器 (CCAE)
# 说明：v1 中称为 "CycleGAN"，但实现无判别器、无对抗损失，
# 实际为 cycle-consistent autoencoder，此处如实命名。
# =====================================================================
class Encoder(nn.Module):
    """编码器 E: X -> Z，用于 cycle-consistency 距离 r_cycle"""

    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CCAE:
    """
    反事实循环一致自编码器 (Counterfactual Cycle-consistent AutoEncoder)。

    训练目标: cycle loss + 0.5 * identity loss（无对抗损失，如实实现）。
    用途: 生成反事实特征 X_cf，其编码空间距离 r_cycle 作为
    非一致性分数中的"反事实一致性"项。
    """

    def __init__(self, feature_dim: int, device: str = "cpu",
                 random_state: int = 42, latent_dim: int = 16):
        self.feature_dim = feature_dim
        self.device = device
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        def build():
            return nn.Sequential(
                nn.Linear(feature_dim, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, feature_dim), nn.Tanh()
            ).to(device)

        self.generator_AB = build()
        self.generator_BA = build()
        self.encoder = Encoder(feature_dim, latent_dim).to(device)
        self.is_trained = False

    def fit(self, X: np.ndarray, treatment: np.ndarray,
            epochs: int = 100, lr: float = 1e-3) -> None:
        if self.is_trained or len(X) < 10:
            return
        idx_A = np.where(treatment == 0)[0]
        idx_B = np.where(treatment == 1)[0]
        if len(idx_A) < 5 or len(idx_B) < 5:
            logger.warning("CCAE: 某治疗组样本过少，跳过反事实预训练")
            return

        logger.info(f"预训练CCAE ({epochs} epochs)...")
        X_t = torch.FloatTensor(X).to(self.device)
        X_A, X_B = X_t[idx_A], X_t[idx_B]

        params = (list(self.generator_AB.parameters()) +
                  list(self.generator_BA.parameters()) +
                  list(self.encoder.parameters()))
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)

        batch = min(32, len(idx_A), len(idx_B))
        best_loss, patience_counter = float("inf"), 0

        for epoch in range(epochs):
            iA = torch.randperm(len(idx_A))[:batch]
            iB = torch.randperm(len(idx_B))[:batch]
            real_A, real_B = X_A[iA], X_B[iB]

            optimizer.zero_grad()
            fake_B = self.generator_AB(real_A)
            fake_A = self.generator_BA(real_B)
            rec_A = self.generator_BA(fake_B)
            rec_B = self.generator_AB(fake_A)

            cycle_loss = (torch.mean(torch.abs(rec_A - real_A)) +
                          torch.mean(torch.abs(rec_B - real_B)))
            identity_loss = (torch.mean(torch.abs(self.generator_BA(real_A) - real_A)) +
                             torch.mean(torch.abs(self.generator_AB(real_B) - real_B)))
            # 编码空间 cycle loss（使 encoder 参与训练）
            enc_rec = self.encoder(rec_A)
            enc_real = self.encoder(real_A)
            enc_loss = torch.mean((enc_rec - enc_real) ** 2)

            loss = cycle_loss + 0.5 * identity_loss + 0.1 * enc_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            if loss.item() < best_loss - 1e-6:
                best_loss, patience_counter = loss.item(), 0
            else:
                patience_counter += 1
            if patience_counter >= 20:
                logger.info(f"CCAE早停于epoch {epoch}, 最佳损失:{best_loss:.4f}")
                break

        self.is_trained = True
        logger.info(f"CCAE预训练完成，最终损失:{best_loss:.4f}")

    @torch.no_grad()
    def generate_counterfactuals(self, X: np.ndarray, treatment: np.ndarray,
                                 constraint_factor: float = 2.0) -> np.ndarray:
        """生成带软约束的反事实特征"""
        X_t = torch.FloatTensor(X).to(self.device)
        feature_std = np.std(X, axis=0) + 1e-8
        threshold = constraint_factor * feature_std
        outs = []

        for i, t in enumerate(treatment):
            x = X_t[i:i + 1]
            cf = self.generator_AB(x) if t == 0 else self.generator_BA(x)
            cf_np = cf.cpu().numpy()
            diff = cf_np - X[i:i + 1]
            mask = np.abs(diff) > threshold
            cf_np = np.where(mask, X[i:i + 1] + np.sign(diff) * threshold, cf_np)
            cf_np = cf_np + np.random.randn(*cf_np.shape) * 0.01 * feature_std
            outs.append(cf_np)
        return np.vstack(outs)

    @torch.no_grad()
    def compute_cycle_consistency(self, X: np.ndarray, X_cf: np.ndarray) -> np.ndarray:
        z_x = self.encoder(torch.FloatTensor(X).to(self.device)).cpu().numpy()
        z_cf = self.encoder(torch.FloatTensor(X_cf).to(self.device)).cpu().numpy()
        r = np.linalg.norm(z_cf - z_x, axis=1) / np.sqrt(self.encoder.latent_dim)
        return np.tanh(r)


# =====================================================================
# Nyström RBF 核（RKHS 距离）
# =====================================================================
class NystromRBFKernel:
    def __init__(self, gamma: float = 0.001, m: int = 200, random_state: int = 42):
        self.gamma, self.m, self.random_state = gamma, m, random_state
        self.landmarks = None
        self.K_mm_inv_sqrt = None
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> "NystromRBFKernel":
        rng = np.random.RandomState(self.random_state)
        n = len(X)
        m = min(self.m, max(n // 2, 10))
        self.landmarks = X[rng.choice(n, m, replace=False)]
        K_mm = rbf_kernel(self.landmarks, self.landmarks, gamma=self.gamma)
        K_mm += np.eye(m) * 1e-6
        try:
            eigvals, eigvecs = np.linalg.eigh(K_mm)
            eigvals = np.maximum(eigvals, 1e-10)
            self.K_mm_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        except np.linalg.LinAlgError:
            self.K_mm_inv_sqrt = np.linalg.inv(K_mm + np.eye(m) * 0.01)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        K_nm = rbf_kernel(X, self.landmarks, gamma=self.gamma)
        return K_nm @ self.K_mm_inv_sqrt

    def rkhs_distance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(self.transform(X1) - self.transform(X2), axis=1)
        return d / np.sqrt(self.transform(X1).shape[1])


# =====================================================================
# 倾向得分模型（IPW 权重来源）
# =====================================================================
def _make_classifier(model_type: str, n_samples: int) -> Any:
    """按配置构造 nuisance 分类器。

    "auto": 样本量较大(n>=200)时用正则化梯度提升(GBM)，否则用逻辑回归——
    小样本(如内部 MRI 训练折)下 GBM 不稳定，正则化 LR 更稳；
    大样本(如 IHDP/PPMI)下 GBM 偏差更小。配合交叉拟合防过拟合。
    """
    mt = model_type
    if mt == "auto":
        mt = "gbm" if n_samples >= 200 else "logistic"
    if mt == "gbm":
        return GradientBoostingClassifier(
            n_estimators=100, max_depth=2, learning_rate=0.05,
            min_samples_leaf=20, subsample=0.8, random_state=42)
    return LogisticRegression(max_iter=1000)


class PropensityModel:
    """
    倾向得分 e(X) = P(T=1|X)。仅使用训练集拟合，校准/测试阶段只调用 predict，
    因此对校准点天然样本外（无需再交叉拟合）。
    IPW 权重: w_i = T_i/e_i + (1-T_i)/(1-e_i)，截断防极端。
    """

    def __init__(self, clip: Tuple[float, float] = (0.05, 20.0),
                 model_type: str = "logistic"):
        self.clip = clip
        self.model_type = model_type
        self.model: Optional[Any] = None
        self.constant_ps: Optional[float] = None  # 拟合失败时的退化值

    def fit(self, X: np.ndarray, treatment: np.ndarray) -> "PropensityModel":
        if len(np.unique(treatment)) < 2:
            self.constant_ps = float(np.mean(treatment))
            return self
        try:
            self.model = _make_classifier(self.model_type, len(X))
            self.model.fit(X, treatment)
        except Exception as e:
            logger.warning(f"倾向得分模型拟合失败，退化为常数: {e}")
            self.constant_ps = float(np.mean(treatment))
        return self

    def ipw_weights(self, X: np.ndarray, treatment: np.ndarray) -> np.ndarray:
        if self.model is None:
            e = np.full(len(X), self.constant_ps if self.constant_ps else 0.5)
        else:
            e = np.clip(self.model.predict_proba(X)[:, 1], *self.clip)
        w = treatment / e + (1 - treatment) / (1 - e)
        return np.clip(w, *self.clip)


# =====================================================================
# 协变量偏移密度比模型（外部验证加权，Tibshirani et al. 2019）
# =====================================================================
class DensityRatioModel:
    """
    用逻辑回归区分 校准集(0) vs 测试集(1)，
    得似然比 w(x) = P(test|x)/P(cal|x) ∝ p_test(x)/p_cal(x)。
    只使用特征 X，不使用任何标签。
    """

    def __init__(self, model_type: str = "logistic"):
        self.model_type = model_type
        self.model: Optional[Any] = None
        self.ok = False

    def fit(self, X_cal: np.ndarray, X_test: np.ndarray) -> "DensityRatioModel":
        n_cal, n_test = len(X_cal), len(X_test)
        if n_cal < 10 or n_test < 10:
            return self
        X = np.vstack([X_cal, X_test])
        z = np.concatenate([np.zeros(n_cal), np.ones(n_test)])
        try:
            self.model = _make_classifier(self.model_type, n_cal + n_test)
            self.model.fit(X, z)
            self.ok = True
        except Exception as e:
            logger.warning(f"密度比模型拟合失败: {e}")
        return self

    def ratio(self, X: np.ndarray) -> np.ndarray:
        if not self.ok:
            return np.ones(len(X))
        p_test = self.model.predict_proba(X)[:, 1]
        return np.clip(p_test / np.maximum(1 - p_test, 1e-6), 0.05, 20.0)

    def fit_crossfit(self, X_cal: np.ndarray, X_test: np.ndarray,
                     k: int = 5) -> "DensityRatioModel":
        """K 折交叉拟合版本（期刊审稿意见：防止校准点被 in-sample 打分）。

        校准侧的密度比必须避免过拟合偏差：把校准集分成 k 折，
        每一折的分类器在【其余校准折 + 全部测试点】上训练，
        对该折给出样本外（OOF）密度比；同时保留全量模型。
        样本太少时自动退回普通 fit()。
        """
        n_cal, n_test = len(X_cal), len(X_test)
        if n_cal < 20 or n_test < 10:
            return self.fit(X_cal, X_test)
        X = np.vstack([X_cal, X_test])
        z = np.concatenate([np.zeros(n_cal), np.ones(n_test)])
        try:
            self.model = _make_classifier(self.model_type, n_cal + n_test)
            self.model.fit(X, z)
            self.ok = True
        except Exception as e:
            logger.warning(f"密度比模型拟合失败: {e}")
            return self
        k = max(2, min(k, n_cal // 10))
        idx = np.arange(n_cal)
        oof = np.zeros(n_cal)
        for val_idx in np.array_split(idx, k):
            tr_idx = np.setdiff1d(idx, val_idx)
            Xtr = np.vstack([X_cal[tr_idx], X_test])
            ztr = np.concatenate([np.zeros(len(tr_idx)), np.ones(n_test)])
            try:
                m = _make_classifier(self.model_type, len(Xtr))
                m.fit(Xtr, ztr)
                p = m.predict_proba(X_cal[val_idx])[:, 1]
                oof[val_idx] = np.clip(p / np.maximum(1 - p, 1e-6),
                                       0.05, 20.0)
            except Exception:
                oof[val_idx] = self.ratio(X_cal[val_idx])
        self._cal_ratios = oof
        return self

    def ratio_cal(self, n_cal: int) -> np.ndarray:
        """校准点的（交叉拟合）密度比；未做交叉拟合时退回全量模型。"""
        if hasattr(self, "_cal_ratios") and len(self._cal_ratios) == n_cal:
            return self._cal_ratios
        return np.ones(n_cal)


# =====================================================================
# T-Learner 结果模型（ITE 点估计 + 集成不确定性）
# =====================================================================
class TLearner:
    """双臂随机森林 T-Learner。ITE = mu1(X) - mu0(X)"""

    def __init__(self, random_state: int = 42):
        self.rf0 = RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_split=5,
            min_samples_leaf=2, max_features="sqrt", random_state=random_state)
        self.rf1 = RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_split=5,
            min_samples_leaf=2, max_features="sqrt", random_state=random_state + 1)

    def fit(self, X: np.ndarray, y: np.ndarray, t: np.ndarray) -> "TLearner":
        """某治疗组为空时退化为另一臂模型（保证流程不崩溃，并给出警告）"""
        self._arm0_ok = (t == 0).sum() >= 2
        self._arm1_ok = (t == 1).sum() >= 2
        if self._arm0_ok:
            self.rf0.fit(X[t == 0], y[t == 0])
        if self._arm1_ok:
            self.rf1.fit(X[t == 1], y[t == 1])
        if not self._arm0_ok and not self._arm1_ok:
            raise ValueError("训练集中无有效样本")
        if not self._arm0_ok:
            logger.warning("T-Learner: 对照组为空，mu0 退化为 mu1")
        if not self._arm1_ok:
            logger.warning("T-Learner: 治疗组为空，mu1 退化为 mu0")
        return self

    @staticmethod
    def _mean_std(rf: RandomForestRegressor, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        preds = np.array([tree.predict(X) for tree in rf.estimators_])
        return preds.mean(axis=0), preds.std(axis=0)

    def predict_ite(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._arm0_ok:
            mu0, s0 = self._mean_std(self.rf0, X)
        else:
            mu0, s0 = self._mean_std(self.rf1, X)
        if self._arm1_ok:
            mu1, s1 = self._mean_std(self.rf1, X)
        else:
            mu1, s1 = self._mean_std(self.rf0, X)
        ite = mu1 - mu0
        sigma = np.sqrt(s0 ** 2 + s1 ** 2)
        sigma = np.maximum(sigma, 1e-3)
        return ite, sigma


# =====================================================================
# 区间估计方法统一接口与各方法实现
# 协议: fit() -> calibrate() -> predict_intervals() -> evaluate_intervals()
# 关键约束: predict_intervals() 不接触任何测试标签；
#          calibrate() 只接触校准集（含校准集模拟ITE，半合成协议）。
# =====================================================================
class IntervalMethod:
    """所有区间估计方法的基类"""

    name: str = "base"

    def __init__(self, config: CPCIv2Config):
        self.config = config
        self.cal_scores: Optional[np.ndarray] = None   # 校准集非一致性分数
        self.cal_weights: Optional[np.ndarray] = None  # 校准集权重(IPW*shift)
        self.alpha_hat: float = 0.0                    # 校准分位数

    def fit(self, X_train, y_train, t_train) -> "IntervalMethod":
        raise NotImplementedError

    def calibrate(self, X_cal, y_cal, t_cal, true_ite_cal, groups_cal,
                  X_test=None) -> "IntervalMethod":
        raise NotImplementedError

    def predict_intervals(self, X_test, t_test, groups_test=None):
        """返回 (lower, upper, ite_pred, base_half)；不使用测试标签"""
        raise NotImplementedError


class NaiveSplitConformal(IntervalMethod):
    """基线1: 朴素 split conformal（T-learner 残差 + 普通分位数）"""

    name = "NaiveSplit"

    def __init__(self, config):
        super().__init__(config)
        self.learner = TLearner(config.random_state)

    def fit(self, X_train, y_train, t_train):
        self.learner.fit(X_train, y_train, t_train)
        return self

    def calibrate(self, X_cal, y_cal, t_cal, true_ite_cal, groups_cal, X_test=None):
        ite_cal, _ = self.learner.predict_ite(X_cal)
        r = np.abs(ite_cal - true_ite_cal)
        clip_v = np.quantile(r, self.config.residual_clip_quantile)
        self.cal_scores = np.clip(r, 0, clip_v)
        self.cal_weights = np.ones(len(r))
        n = len(self.cal_scores)
        level = min(1.0, np.ceil((n + 1) * (1 - self.config.alpha)) / n)
        self.alpha_hat = float(np.quantile(self.cal_scores, level))
        return self

    def predict_intervals(self, X_test, t_test, groups_test=None):
        ite_pred, _ = self.learner.predict_ite(X_test)
        half = np.full(len(X_test), self.alpha_hat)
        return ite_pred - half, ite_pred + half, ite_pred, np.zeros(len(X_test))


class XLearnerConformal(IntervalMethod):
    """基线2: Alaa et al. (2023) conformal meta-learner 风格的代表——
    X-learner (Künzel et al. 2019) 点估计 + 朴素 split conformal 区间。

    该方法遵循 conformal meta-learner 的标准做法: 用成熟 meta-learner
    得到 ITE 点估计, 再在校准集上做普通 split conformal。
    它假设治疗分配满足可忽略性(随机化试验), 因此不做任何倾向加权——
    在混杂的观察性数据下其覆盖保证不再成立,
    正好用于验证本文双重加权校准的必要性。
    """

    name = "XConfMeta"

    def __init__(self, config):
        super().__init__(config)
        rs = config.random_state + 20
        self.rf0 = RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_split=5,
            min_samples_leaf=2, max_features="sqrt", random_state=rs)
        self.rf1 = RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_split=5,
            min_samples_leaf=2, max_features="sqrt", random_state=rs + 1)
        self.tau0 = RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_split=5,
            min_samples_leaf=2, max_features="sqrt", random_state=rs + 2)
        self.tau1 = RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_split=5,
            min_samples_leaf=2, max_features="sqrt", random_state=rs + 3)
        self.propensity = LogisticRegression(max_iter=1000)

    def fit(self, X_train, y_train, t_train):
        X0, y0 = X_train[t_train == 0], y_train[t_train == 0]
        X1, y1 = X_train[t_train == 1], y_train[t_train == 1]
        if len(y0) < 2 or len(y1) < 2:
            raise ValueError("X-learner 需要双臂均至少有 2 个训练样本")
        self.rf0.fit(X0, y0)
        self.rf1.fit(X1, y1)
        # 反事实插补: 治疗组用 mu0 估 D1, 对照组用 mu1 估 D0
        d1 = y1 - self.rf0.predict(X1)
        d0 = self.rf1.predict(X0) - y0
        self.tau1.fit(X1, d1)
        self.tau0.fit(X0, d0)
        try:
            self.propensity.fit(X_train, t_train)
            self._prop_ok = True
        except Exception:
            self._prop_ok = False
        return self

    def predict_ite(self, X):
        t0 = self.tau0.predict(X)
        t1 = self.tau1.predict(X)
        if self._prop_ok:
            g = self.propensity.predict_proba(X)[:, 1]
            g = np.clip(g, 0.05, 0.95)
        else:
            g = np.full(len(X), 0.5)
        return g * t0 + (1.0 - g) * t1

    def calibrate(self, X_cal, y_cal, t_cal, true_ite_cal, groups_cal, X_test=None):
        ite_cal = self.predict_ite(X_cal)
        r = np.abs(ite_cal - true_ite_cal)
        clip_v = np.quantile(r, self.config.residual_clip_quantile)
        self.cal_scores = np.clip(r, 0, clip_v)
        self.cal_weights = np.ones(len(r))
        n = len(self.cal_scores)
        level = min(1.0, np.ceil((n + 1) * (1 - self.config.alpha)) / n)
        self.alpha_hat = float(np.quantile(self.cal_scores, level))
        return self

    def predict_intervals(self, X_test, t_test, groups_test=None):
        ite_pred = self.predict_ite(X_test)
        half = np.full(len(X_test), self.alpha_hat)
        return ite_pred - half, ite_pred + half, ite_pred, np.zeros(len(X_test))


class WeightedConformal(IntervalMethod):
    """基线3: Tibshirani et al. (2019) 协变量偏移加权 conformal"""

    name = "WeightedConf"

    def __init__(self, config):
        super().__init__(config)
        self.learner = TLearner(config.random_state + 10)

    def fit(self, X_train, y_train, t_train):
        self.learner.fit(X_train, y_train, t_train)
        return self

    def calibrate(self, X_cal, y_cal, t_cal, true_ite_cal, groups_cal, X_test=None):
        ite_cal, _ = self.learner.predict_ite(X_cal)
        r = np.abs(ite_cal - true_ite_cal)
        clip_v = np.quantile(r, self.config.residual_clip_quantile)
        self.cal_scores = np.clip(r, 0, clip_v)
        w = np.ones(len(r))
        if self.config.use_shift_weight and X_test is not None and len(X_test) >= 10:
            drm = DensityRatioModel(model_type=self.config.nuisance_model).fit_crossfit(X_cal, X_test,
                                                   k=self.config.crossfit_k)
            w = drm.ratio_cal(len(r))
        self.cal_weights = w
        n = len(self.cal_scores)
        level = min(1.0, np.ceil((n + 1) * (1 - self.config.alpha)) / n)
        self.alpha_hat = weighted_quantile(self.cal_scores, level, w)
        return self

    def predict_intervals(self, X_test, t_test, groups_test=None):
        ite_pred, _ = self.learner.predict_ite(X_test)
        half = np.full(len(X_test), self.alpha_hat)
        return ite_pred - half, ite_pred + half, ite_pred, np.zeros(len(X_test))


class MondrianConformal(IntervalMethod):
    """基线3: Mondrian 组条件 conformal（按组分别估计分位数）"""

    name = "Mondrian"

    def __init__(self, config):
        super().__init__(config)
        self.learner = TLearner(config.random_state + 20)
        self.alpha_per_group: Dict[Any, float] = {}
        self.global_alpha: float = 0.0

    def fit(self, X_train, y_train, t_train):
        self.learner.fit(X_train, y_train, t_train)
        return self

    def calibrate(self, X_cal, y_cal, t_cal, true_ite_cal, groups_cal, X_test=None):
        ite_cal, _ = self.learner.predict_ite(X_cal)
        r = np.abs(ite_cal - true_ite_cal)
        clip_v = np.quantile(r, self.config.residual_clip_quantile)
        self.cal_scores = np.clip(r, 0, clip_v)
        self.cal_weights = np.ones(len(r))
        n = len(r)
        level = min(1.0, np.ceil((n + 1) * (1 - self.config.alpha)) / n)
        self.global_alpha = float(np.quantile(self.cal_scores, level))
        self.alpha_hat = self.global_alpha
        for g in np.unique(groups_cal):
            mask = groups_cal == g
            ng = int(mask.sum())
            if ng >= 10:
                lv = min(1.0, np.ceil((ng + 1) * (1 - self.config.alpha)) / ng)
                self.alpha_per_group[g] = float(np.quantile(self.cal_scores[mask], lv))
            else:
                self.alpha_per_group[g] = self.global_alpha
        return self

    def predict_intervals(self, X_test, t_test, groups_test=None):
        ite_pred, _ = self.learner.predict_ite(X_test)
        half = np.array([
            self.alpha_per_group.get(g, self.global_alpha) for g in groups_test
        ])
        return ite_pred - half, ite_pred + half, ite_pred, np.zeros(len(X_test))


class CQRBaseline(IntervalMethod):
    """
    基线4: CQR（Conformalized Quantile Regression, Romano et al. 2019）
    双臂分位数回归 + conformal 校正。
    ITE 区间: [q_lo(x,1)-q_hi(x,0)-alpha, q_hi(x,1)-q_lo(x,0)+alpha]
    """

    name = "CQR"

    @staticmethod
    def _make_quantile_regressor(quantile: float, random_state: int):
        """跨 sklearn 版本兼容的分位数回归器"""
        try:
            return HistGradientBoostingRegressor(
                loss="quantile", quantile=quantile, max_iter=200,
                learning_rate=0.05, max_depth=6,
                random_state=random_state)
        except (TypeError, ValueError):  # 兼容旧版 sklearn
            return GradientBoostingRegressor(
                loss="quantile", quantile=quantile, n_estimators=200,
                learning_rate=0.05, max_depth=6, subsample=0.8,
                random_state=random_state)

    def __init__(self, config):
        super().__init__(config)
        q_lo = self.config.alpha
        q_hi = 1 - self.config.alpha
        self.q = {
            (arm, ql): self._make_quantile_regressor(ql, config.random_state + 30)
            for arm in (0, 1) for ql in (q_lo, q_hi)
        }
        self.q_lo, self.q_hi = q_lo, q_hi

    def fit(self, X_train, y_train, t_train):
        self._fitted_arms = {}
        for arm in (0, 1):
            m = t_train == arm
            if m.sum() < 5:
                # 空臂退化：用另一臂数据拟合，保证 predict 不崩溃
                m = t_train != arm
                logger.warning(f"CQR: 治疗组{arm}样本过少，退化为另一臂模型")
            for ql in (self.q_lo, self.q_hi):
                self.q[(arm, ql)].fit(X_train[m], y_train[m])
            self._fitted_arms[arm] = True
        return self

    def calibrate(self, X_cal, y_cal, t_cal, true_ite_cal, groups_cal, X_test=None):
        r = np.zeros(len(X_cal))
        for arm in (0, 1):
            m = t_cal == arm
            if m.sum() == 0:
                continue
            q_lo_p = self.q[(arm, self.q_lo)].predict(X_cal[m])
            q_hi_p = self.q[(arm, self.q_hi)].predict(X_cal[m])
            r[m] = np.maximum(q_lo_p - y_cal[m], y_cal[m] - q_hi_p)
        clip_v = np.quantile(r, self.config.residual_clip_quantile)
        self.cal_scores = np.clip(r, 0, clip_v)
        self.cal_weights = np.ones(len(r))
        n = len(r)
        level = min(1.0, np.ceil((n + 1) * (1 - self.config.alpha)) / n)
        self.alpha_hat = float(np.quantile(self.cal_scores, level))
        return self

    def predict_intervals(self, X_test, t_test, groups_test=None):
        # ITE 点估计: 双臂上下分位数的差的中点
        ite_pred = (0.5 * (self.q[(1, self.q_lo)].predict(X_test) +
                           self.q[(1, self.q_hi)].predict(X_test)) -
                    0.5 * (self.q[(0, self.q_lo)].predict(X_test) +
                           self.q[(0, self.q_hi)].predict(X_test)))
        lo = (self.q[(1, self.q_lo)].predict(X_test) -
              self.q[(0, self.q_hi)].predict(X_test) - self.alpha_hat)
        hi = (self.q[(1, self.q_hi)].predict(X_test) -
              self.q[(0, self.q_lo)].predict(X_test) + self.alpha_hat)
        return lo, hi, ite_pred, np.zeros(len(X_test))


class CPCI(IntervalMethod):
    """
    本文方法: 反事实感知加权因果保形预测 (Counterfactual-aware Weighted
    Causal Conformal Prediction)

    非一致性分数:  s_i = r_i + lambda_score * S_i,
                  S_i = r_cycle(x_i, x_cf_i) + lambda_effect * delta_RKHS(x_i, x_cf_i)
    校准:         alpha = (加权) 分位数_{1-alpha}(s)，权重 = IPW x 密度比
    区间:         [ite_hat - (alpha + eta * sigma), ite_hat + (alpha + eta * sigma)]
    """

    def __init__(self, config: CPCIv2Config, use_ipw: bool = True,
                 use_score: bool = True, name: str = "CPCI(ours)"):
        super().__init__(config)
        self._use_ipw = use_ipw
        self._use_score = use_score
        self.name = name
        self.learner = TLearner(config.random_state + 40)
        self.propensity = PropensityModel(config.ipw_clip,
                                          model_type=config.nuisance_model)
        self.ccae: Optional[CCAE] = None
        self.kernel: Optional[NystromRBFKernel] = None

    def fit(self, X_train, y_train, t_train):
        self.learner.fit(X_train, y_train, t_train)
        if self._use_ipw:
            self.propensity.fit(X_train, t_train)
        if self._use_score:
            self.ccae = CCAE(X_train.shape[1], self.config.device,
                             self.config.random_state)
            self.ccae.fit(X_train, t_train, epochs=100)
            self.kernel = NystromRBFKernel(
                gamma=self.config.rbf_gamma,
                m=min(self.config.nystrom_landmarks, max(len(X_train) // 2, 20)),
                random_state=self.config.random_state).fit(X_train)
        return self

    def _counterfactual_score(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        """S = r_cycle + lambda_effect * delta_RKHS（编码器/RKHS均在拟合阶段训练）"""
        if self.ccae is None or self.kernel is None:
            return np.zeros(len(X))
        X_cf = self.ccae.generate_counterfactuals(
            X, t, constraint_factor=self.config.cf_constraint_factor)
        r_cycle = self.ccae.compute_cycle_consistency(X, X_cf)
        delta = self.kernel.rkhs_distance(X, X_cf)
        S = r_cycle + self.config.lambda_effect * delta
        return np.clip(S, 0, 5.0)

    def calibrate(self, X_cal, y_cal, t_cal, true_ite_cal, groups_cal, X_test=None):
        ite_cal, _ = self.learner.predict_ite(X_cal)
        r = np.abs(ite_cal - true_ite_cal)
        clip_v = np.quantile(r, self.config.residual_clip_quantile)
        r = np.clip(r, 0, clip_v)

        if self._use_score:
            S = self._counterfactual_score(X_cal, t_cal)
            # 标准化到与残差同尺度
            S = S * (np.median(r) / (np.median(S) + 1e-8))
            s = r + self.config.lambda_score * S
        else:
            s = r
        self.cal_scores = s

        # 权重 = IPW x 密度比（各自可关）
        w = np.ones(len(s))
        if self._use_ipw and self.propensity.model is not None:
            w = w * self.propensity.ipw_weights(X_cal, t_cal)
        if self.config.use_shift_weight and X_test is not None and len(X_test) >= 10:
            drm = DensityRatioModel(model_type=self.config.nuisance_model).fit_crossfit(X_cal, X_test,
                                                   k=self.config.crossfit_k)
            w = w * drm.ratio_cal(len(s))
        self.cal_weights = w

        n = len(s)
        level = min(1.0, np.ceil((n + 1) * (1 - self.config.alpha)) / n)
        self.alpha_hat = weighted_quantile(s, level, w)
        self.alpha_hat = max(self.alpha_hat, 1e-6)
        return self

    def predict_intervals(self, X_test, t_test, groups_test=None):
        ite_pred, sigma = self.learner.predict_ite(X_test)
        base_half = self.config.eta_uncertainty * sigma
        half = self.alpha_hat + base_half
        return ite_pred - half, ite_pred + half, ite_pred, base_half


def build_methods(config: CPCIv2Config,
                  only: Optional[List[str]] = None) -> Dict[str, IntervalMethod]:
    """构建全部对比方法（含本文方法的消融）。only 可指定子集。"""
    methods: Dict[str, IntervalMethod] = {
        "NaiveSplit": NaiveSplitConformal(config),
        "XConfMeta": XLearnerConformal(config),
        "WeightedConf": WeightedConformal(config),
        "Mondrian": MondrianConformal(config),
        "CQR": CQRBaseline(config),
        "CPCI(ours)": CPCI(config, use_ipw=True, use_score=True),
        "CPCI-noIPW": CPCI(config, use_ipw=False, use_score=True,
                           name="CPCI-noIPW"),
        "CPCI-noScore": CPCI(config, use_ipw=True, use_score=False,
                             name="CPCI-noScore"),
    }
    if only is not None:
        methods = {k: v for k, v in methods.items() if k in only}
    return methods


# =====================================================================
# 混杂半合成数据生成 (Semi-Synthetic DGP)
# 与 v1 的关键区别:
#   v1: treatment ~ Bernoulli(0.5) 完全随机，ITE = 线性函数 + 噪声
#       -> 无混杂，任何加权都是装饰；模型可轻易学出线性ITE。
#   v2: treatment ~ Bernoulli(e(X))，e(X) 依赖协变量（混杂）
#       ITE 为非线性函数，IPW 加权成为必要。
# =====================================================================
class SemiSyntheticDGP:
    def __init__(self, config: CPCIv2Config):
        self.config = config

    def generate(self, X: np.ndarray, y: np.ndarray, modality: str,
                 rng: np.random.RandomState,
                 external: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        返回 (treatment, true_ite, y_obs)
        y_obs = y(诊断基线) + T * ITE + 噪声  （半合成结构，与 v1 一致）
        """
        n, p = X.shape
        i1, i2, i3 = 0, min(1, p - 1), min(3, p - 1)
        i4 = min(2, p - 1)

        # 先标准化特征：真实特征（如MFCC）量纲差异大，
        # 不标准化会导致 expit(大负数)≈0 使某治疗组为空
        Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

        # 倾向得分: e(X) 依赖多个协变量 -> 治疗选择偏差（混杂）
        z = Xs[:, i1] + 0.5 * Xs[:, i2] - 0.3 * Xs[:, min(4, p - 1)]
        # 截断倾向得分，避免极端倾向使某组样本过少
        e = np.clip(expit(self.config.confounding_strength * z), 0.2, 0.8)
        treatment = rng.binomial(1, e, n)

        # 保底：两组均需足够样本，否则回退为随机分配（保证流程可运行）
        n1 = int(treatment.sum())
        if min(n1, n - n1) < max(10, int(0.1 * n)):
            treatment = rng.binomial(1, 0.5, n)
            logger.warning(
                f"倾向分配导致治疗组样本过少(T={n1}/{n})，"
                f"本seed回退为随机分配 Bernoulli(0.5)")

        # 非线性 ITE（tanh/交互项/正弦），避免线性 trivially learnable
        ite = (0.6 * np.tanh(Xs[:, i1])
               + 0.4 * Xs[:, i2] * Xs[:, i3]
               + 0.2 * np.sin(2.0 * Xs[:, i4]))

        if modality == "audio":
            noise_level = self.config.noise_level_audio
        else:
            noise_level = (self.config.noise_level_external if external
                           else self.config.noise_level_mri)
        ite = ite + rng.normal(0, noise_level, n)

        y_obs = y + treatment * ite + rng.normal(0, 0.1, n)
        return treatment, ite, y_obs


# =====================================================================
# 实验运行器
# =====================================================================
class ExperimentRunner:
    METRICS = ["coverage", "width_mean", "width_std", "width_median",
               "fsc", "cov_gap", "residual_mean", "residual_std"]

    def __init__(self, config: CPCIv2Config):
        self.config = config
        self.dgp = SemiSyntheticDGP(config)

    @staticmethod
    def _split_indices(n: int, config: CPCIv2Config, rng: np.random.RandomState):
        idx = rng.permutation(n)
        n_tr = int(n * config.train_ratio)
        n_ca = int(n * config.cal_ratio)
        return idx[:n_tr], idx[n_tr:n_tr + n_ca], idx[n_tr + n_ca:]

    def _run_one_seed(self, X: np.ndarray, y: np.ndarray, modality: str,
                      seed: int, methods: Dict[str, IntervalMethod],
                      X_external: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                      precomputed: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
                      ) -> Dict[str, Dict[str, Any]]:
        """
        单个种子下的完整流程。所有方法共享同一数据划分，保证公平对比。
        X_external = (X_ext, y_ext) 若为外部验证；
        precomputed = (t, true_ite, y_obs) 若为 IHDP 等自带真值的基准
        （跳过 DGP 生成；组标签 = 治疗臂，FSC 即最差臂覆盖）。
        """
        config = self.config
        rng = np.random.RandomState(seed)

        if precomputed is not None:
            t_all, ite_all, yobs_all = precomputed
            groups_all = t_all.copy()  # 组 = 治疗臂 -> FSC = 最差臂覆盖
        else:
            t_all, ite_all, yobs_all = self.dgp.generate(X, y, modality, rng,
                                                         external=False)
            # 组标签 = 诊断标签（预测时可观测属性，Mondrian 分层用）
            groups_all = y.copy()

        n = len(X)
        if X_external is not None:
            X_ext, y_ext = X_external
            n_tr = int(n * 0.75)
            idx = rng.permutation(n)
            tr, ca = idx[:n_tr], idx[n_tr:]
        else:
            tr, ca, te = self._split_indices(n, config, rng)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])

        results: Dict[str, Dict[str, Any]] = {}
        viz_seed: Dict[str, Any] = {}

        for m_name, method in methods.items():
            # ---- fit: 仅训练集 ----
            method.fit(X_tr, yobs_all[tr], t_all[tr])

            # ---- calibrate: 仅校准集（半合成协议，使用校准集 ITE）----
            if X_external is not None:
                X_ca = scaler.transform(X[ca])
                X_te = scaler.transform(X_ext)
                method.calibrate(X_ca, yobs_all[ca], t_all[ca], ite_all[ca],
                                 groups_all[ca], X_test=X_te)
            else:
                X_ca = scaler.transform(X[ca])
                X_te = scaler.transform(X[te])
                method.calibrate(X_ca, yobs_all[ca], t_all[ca], ite_all[ca],
                                 groups_all[ca], X_test=X_te)

            # ---- predict: 不使用任何测试标签 ----
            if X_external is not None:
                t_ext, ite_ext, yobs_ext = self.dgp.generate(
                    X_ext, y_ext, modality, rng, external=True)
                g_ext = y_ext.copy()
                lower, upper, ite_pred, base_half = method.predict_intervals(
                    X_te, t_ext, g_ext)
                true_te, g_te, t_te = ite_ext, g_ext, t_ext
            else:
                t_te = t_all[te]
                true_te = ite_all[te]
                g_te = groups_all[te]
                lower, upper, ite_pred, base_half = method.predict_intervals(
                    X_te, t_te, g_te)

            # ---- evaluate: 标签仅用于计算指标 ----
            res = evaluate_intervals(lower, upper, true_te, g_te, ite_pred)
            arrays = res.pop("_arrays")
            results[m_name] = res

            # 可视化数据（真实评估输出）
            viz_seed[m_name] = {
                "widths": arrays["widths"],
                "covered": arrays["covered"],
                "residuals": arrays["residuals"],
                "subgroup": {g: st["coverage"] for g, st in res["subgroup_stats"].items()},
                "cal_scores": np.asarray(method.cal_scores) if method.cal_scores is not None else None,
                "ite_pred_test": ite_pred,
                "base_half": base_half,
                "true_ites_test": true_te,
            }

        return {"metrics": results, "viz": viz_seed}

    def run_setting(self, X: np.ndarray, y: np.ndarray, modality: str,
                    seeds: List[int], setting_name: str,
                    X_external: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                    only: Optional[List[str]] = None,
                    precomputed: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
                    ) -> Dict[str, Any]:
        """多种子运行并聚合 mean±std。only 可只跑指定方法子集。"""
        per_seed_metrics: Dict[str, List[Dict[str, Any]]] = {}
        viz_all: Dict[int, Dict[str, Any]] = {}

        for seed in seeds:
            methods = build_methods(self.config, only=only)
            out = self._run_one_seed(X, y, modality, seed, methods, X_external,
                                     precomputed=precomputed)
            for m_name, m_res in out["metrics"].items():
                per_seed_metrics.setdefault(m_name, []).append(m_res)
            viz_all[seed] = out["viz"]
            logger.info(f"[{setting_name}] seed={seed} 完成")

        # 聚合
        summary: Dict[str, Dict[str, Dict[str, float]]] = {}
        for m_name, records in per_seed_metrics.items():
            summary[m_name] = {"mean": {}, "std": {}}
            for key in self.METRICS:
                vals = [r[key] for r in records]
                summary[m_name]["mean"][key] = float(np.mean(vals))
                summary[m_name]["std"][key] = float(np.std(vals))
            # 子组覆盖（各 seed 组结构一致时取均值）
            sub_keys = records[0]["subgroup_stats"].keys()
            summary[m_name]["subgroup_mean"] = {
                g: float(np.mean([r["subgroup_stats"][g]["coverage"] for r in records]))
                for g in sub_keys
            }

        return {"summary": summary, "viz": viz_all,
                "n_seeds": len(seeds), "setting": setting_name}


# =====================================================================
# 真实数据驱动的可视化（v1 中所有 np.random 伪造图表已删除）
# 输入全部来自 evaluate_intervals 的真实输出与校准集真实分数
# =====================================================================
class ResultVisualizer:
    PALETTE = ["#0F4C81", "#9B2335", "#006B54", "#E15D44", "#55B4B0",
               "#B565A7", "#955251", "#98B4D4"]

    def __init__(self, output_dir: str):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt = plt
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, name):
        for fmt in ("png", "pdf"):
            fig.savefig(self.output_dir / f"{name}.{fmt}",
                        dpi=300, bbox_inches="tight", facecolor="white")
        self.plt.close(fig)
        logger.info(f"图已保存: {name}")

    def plot_coverage_by_method(self, settings_results: Dict[str, Any]):
        """图1: 各设定下覆盖率对比（mean±std over seeds）"""
        plt = self.plt
        settings = list(settings_results.keys())
        methods = list(settings_results[settings[0]]["summary"].keys())
        n_s, n_m = len(settings), len(methods)
        x = np.arange(n_m)
        width = 0.8 / n_s

        fig, ax = plt.subplots(figsize=(max(10, n_m * 1.4), 6))
        for si, setting in enumerate(settings):
            summ = settings_results[setting]["summary"]
            covs = [summ[m]["mean"]["coverage"] * 100 for m in methods]
            stds = [summ[m]["std"]["coverage"] * 100 for m in methods]
            offset = (si - (n_s - 1) / 2) * width
            bars = ax.bar(x + offset, covs, width, yerr=stds, capsize=3,
                          label=setting, color=self.PALETTE[si % 8],
                          edgecolor="black", linewidth=0.6, alpha=0.85,
                          error_kw=dict(lw=1))
            for xi, v in zip(x + offset, covs):
                ax.text(xi, v + 1, f"{v:.1f}", ha="center", va="bottom",
                        fontsize=7, rotation=90)
        target = (1 - 0.1) * 100
        ax.axhline(target, color="red", linestyle="--", linewidth=1.5,
                   label=f"Target {target:.0f}%")
        ax.axhspan(target - 2, target + 2, alpha=0.1, color="green")
        ax.set_ylim(70, 105)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Coverage (%)")
        ax.set_title("Marginal Coverage by Method (mean ± std over seeds)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        self._save(fig, "fig1_coverage_by_method")

    def plot_coverage_width_pareto(self, settings_results: Dict[str, Any]):
        """图2: 覆盖率-宽度帕累托前沿"""
        plt = self.plt
        settings = list(settings_results.keys())
        methods = list(settings_results[settings[0]]["summary"].keys())
        fig, axes = plt.subplots(1, len(settings), figsize=(6 * len(settings), 5.5),
                                 squeeze=False)
        for ax, setting in zip(axes[0], settings):
            summ = settings_results[setting]["summary"]
            for mi, m in enumerate(methods):
                cov = summ[m]["mean"]["coverage"] * 100
                wid = summ[m]["mean"]["width_mean"]
                ax.scatter(wid, cov, s=120, color=self.PALETTE[mi % 8],
                           edgecolor="black", zorder=5)
                ax.annotate(m, (wid, cov), textcoords="offset points",
                            xytext=(6, 4), fontsize=7)
            ax.axhline(90, color="red", linestyle="--", linewidth=1.2)
            ax.set_xlabel("Mean Interval Width")
            ax.set_ylabel("Coverage (%)")
            ax.set_title(f"{setting}")
            ax.grid(alpha=0.3)
        fig.suptitle("Coverage–Width Trade-off (mean over seeds)", fontweight="bold")
        self._save(fig, "fig2_coverage_width_pareto")

    def plot_worst_group_coverage(self, settings_results: Dict[str, Any]):
        """图3: 最差组覆盖 FSC 对比"""
        plt = self.plt
        settings = list(settings_results.keys())
        methods = list(settings_results[settings[0]]["summary"].keys())
        n_s, n_m = len(settings), len(methods)
        x = np.arange(n_m)
        width = 0.8 / n_s
        fig, ax = plt.subplots(figsize=(max(10, n_m * 1.4), 6))
        for si, setting in enumerate(settings):
            summ = settings_results[setting]["summary"]
            fscs = [summ[m]["mean"]["fsc"] * 100 for m in methods]
            stds = [summ[m]["std"]["fsc"] * 100 for m in methods]
            offset = (si - (n_s - 1) / 2) * width
            ax.bar(x + offset, fscs, width, yerr=stds, capsize=3,
                   label=setting, color=self.PALETTE[si % 8],
                   edgecolor="black", linewidth=0.6, alpha=0.85)
        ax.axhline(90, color="red", linestyle="--", linewidth=1.5, label="Target 90%")
        ax.set_ylim(70, 105)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Worst-Group Coverage FSC (%)")
        ax.set_title("Worst-Group (Conditional) Coverage by Method")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        self._save(fig, "fig3_worst_group_coverage")

    def plot_calibration_curve(self, setting_result: Dict[str, Any],
                               setting_name: str):
        """图4: 真实校准曲线——区间由校准集分数分位数构造，覆盖率用测试标签评估"""
        plt = self.plt
        viz = setting_result["viz"]
        nominals = np.linspace(0.5, 0.99, 15)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot([0.5, 1], [0.5, 1], "k--", linewidth=1.5, label="Perfect calibration")
        methods = list(viz[list(viz.keys())[0]].keys())
        for mi, m_name in enumerate(methods):
            scores_all, base_all, ite_all, true_all = [], [], [], []
            for seed, seed_viz in viz.items():
                v = seed_viz.get(m_name)
                if v is None or v["cal_scores"] is None:
                    continue
                scores_all.append(np.asarray(v["cal_scores"]))
                base_all.append(np.asarray(v["base_half"]))
                ite_all.append(np.asarray(v["ite_pred_test"]))
                true_all.append(np.asarray(v["true_ites_test"]))
            if not scores_all:
                continue
            scores = np.concatenate(scores_all)
            base = np.concatenate(base_all)
            ite_p = np.concatenate(ite_all)
            true_t = np.concatenate(true_all)
            n_cal = len(scores)
            obs = []
            for gamma in nominals:
                level = min(1.0, np.ceil((n_cal + 1) * gamma) / n_cal)
                q = float(np.quantile(scores, level))
                half = base + q
                obs.append(float(np.mean((true_t >= ite_p - half) &
                                         (true_t <= ite_p + half))))
            ax.plot(nominals * 100, np.asarray(obs) * 100, "o-",
                    color=self.PALETTE[mi % 8], linewidth=1.8, markersize=4,
                    label=m_name)
        ax.set_xlabel("Nominal Coverage (%)")
        ax.set_ylabel("Observed Coverage (%)")
        ax.set_title(f"Calibration Curve ({setting_name}, pooled over seeds)\n"
                     "intervals built from calibration scores only")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlim(50, 100)
        ax.set_ylim(40, 102)
        self._save(fig, f"fig4_calibration_curve_{setting_name}")

    def plot_width_distribution(self, setting_result: Dict[str, Any],
                                setting_name: str):
        """图5: 区间宽度分布（真实逐样本宽度， pooled over seeds）"""
        plt = self.plt
        viz = setting_result["viz"]
        methods = list(viz[list(viz.keys())[0]].keys())
        data, labels = [], []
        for m_name in methods:
            widths = np.concatenate([
                np.asarray(seed_viz[m_name]["widths"])
                for seed_viz in viz.values()
                if seed_viz.get(m_name) is not None
            ])
            data.append(widths)
            labels.append(m_name)
        fig, ax = plt.subplots(figsize=(max(9, len(methods) * 1.2), 5.5))
        parts = ax.violinplot(data, showmedians=True, showextrema=False)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(self.PALETTE[i % 8])
            pc.set_alpha(0.7)
            pc.set_edgecolor("black")
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(2)
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Interval Width")
        ax.set_title(f"Distribution of Prediction Interval Widths ({setting_name})")
        ax.grid(axis="y", alpha=0.3)
        self._save(fig, f"fig5_width_distribution_{setting_name}")


# =====================================================================
# 汇总输出
# =====================================================================
def save_summary_csv(settings_results: Dict[str, Any], output_dir: str) -> pd.DataFrame:
    """长表: setting, method, metric, mean, std"""
    rows = []
    for setting, res in settings_results.items():
        for m_name, agg in res["summary"].items():
            for key in ExperimentRunner.METRICS:
                rows.append({
                    "setting": setting, "method": m_name, "metric": key,
                    "mean": agg["mean"][key], "std": agg["std"][key]
                })
    df = pd.DataFrame(rows)
    df.to_csv(Path(output_dir) / "metrics_summary.csv", index=False)
    return df


def print_summary_table(settings_results: Dict[str, Any]):
    for setting, res in settings_results.items():
        print(f"\n{'=' * 78}")
        print(f"设定: {setting}  (n_seeds={res['n_seeds']})")
        print(f"{'=' * 78}")
        header = f"{'Method':<16}{'Coverage':>18}{'Width':>14}{'FSC':>18}{'CovGap':>10}"
        print(header)
        print("-" * 78)
        for m_name, agg in res["summary"].items():
            cm, cs = agg["mean"], agg["std"]
            print(f"{m_name:<16}"
                  f"{cm['coverage'] * 100:>8.1f}±{cs['coverage'] * 100:<8.1f}"
                  f"{cm['width_mean']:>14.3f}"
                  f"{cm['fsc'] * 100:>8.1f}±{cs['fsc'] * 100:<8.1f}"
                  f"{cm['cov_gap'] * 100:>9.1f}")


# =====================================================================
# 混杂强度敏感性实验（IPW 贡献的正面证据，对应论文鲁棒性小节）
# =====================================================================
def run_confounding_sensitivity(config: CPCIv2Config,
                                data: Dict[str, Tuple],
                                seeds: List[int],
                                runner: ExperimentRunner) -> pd.DataFrame:
    """
    扫描 confounding_strength ∈ {0, 0.5, 1.0, 2.0}，
    对比 CPCI(ours) 与 CPCI-noIPW 的覆盖率 / FSC。

    Sanity check: γ=0 时治疗完全随机，IPW 权重近似常数，
    两条曲线应基本一致（小样本分位数离散化下可能略有差异）。
    混杂增强时 ours 相对 noIPW 的覆盖优势应扩大——这是 IPW
    贡献的正面证据，对应论文鲁棒性小节。
    """
    gammas = [0.0, 0.5, 1.0, 2.0]
    only = ["CPCI(ours)", "CPCI-noIPW"]
    rows: List[Dict[str, Any]] = []
    plot_data: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    orig_gamma = config.confounding_strength

    try:
        for modality_key in ("audio", "mri"):
            if modality_key not in data:
                continue
            X, y = data[modality_key]
            plot_data[modality_key] = {}
            for g in gammas:
                config.confounding_strength = g
                logger.info(f"[敏感性] {modality_key}, confounding={g}")
                res = runner.run_setting(X, y, modality_key, seeds,
                                         f"{modality_key}_g{g}", only=only)
                for m_name, agg in res["summary"].items():
                    rows.append({
                        "modality": modality_key, "confounding": g,
                        "method": m_name,
                        "coverage_mean": agg["mean"]["coverage"],
                        "coverage_std": agg["std"]["coverage"],
                        "fsc_mean": agg["mean"]["fsc"],
                        "width_mean": agg["mean"]["width_mean"],
                    })
                    d = plot_data[modality_key].setdefault(
                        m_name, {"gamma": [], "cov": [], "cov_std": [], "fsc": []})
                    d["gamma"].append(g)
                    d["cov"].append(agg["mean"]["coverage"] * 100)
                    d["cov_std"].append(agg["std"]["coverage"] * 100)
                    d["fsc"].append(agg["mean"]["fsc"] * 100)
    finally:
        config.confounding_strength = orig_gamma

    df = pd.DataFrame(rows)
    df.to_csv(Path(config.output_dir) / "confounding_sensitivity.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(plot_data), figsize=(6 * len(plot_data), 5),
                             squeeze=False)
    palette = {"CPCI(ours)": "#0F4C81", "CPCI-noIPW": "#E15D44"}
    for ax, (modality_key, mdata) in zip(axes[0], plot_data.items()):
        for m_name, d in mdata.items():
            c = palette.get(m_name, "#333333")
            ax.errorbar(d["gamma"], d["cov"], yerr=d["cov_std"], marker="o",
                        linewidth=2, capsize=4,
                        label=f"{m_name} (coverage)", color=c)
            ax.plot(d["gamma"], d["fsc"], marker="s", linestyle="--",
                    linewidth=1.5, alpha=0.7,
                    label=f"{m_name} (FSC)", color=c)
        ax.axhline(90, color="red", linestyle=":", linewidth=1.2)
        ax.set_xlabel("Confounding strength γ")
        ax.set_ylabel("Coverage / FSC (%)")
        ax.set_title(f"{modality_key}")
        ax.set_xticks(gammas)
        ax.grid(alpha=0.3)
    if len(plot_data) > 0:
        axes[0][0].legend(fontsize=8)
    fig.suptitle("IPW Weighting under Varying Treatment Selection Bias "
                 "(mean ± std over seeds)", fontweight="bold")
    out = Path(config.output_dir) / "fig6_confounding_sensitivity"
    for fmt in ("png", "pdf"):
        fig.savefig(f"{out}.{fmt}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("图已保存: fig6_confounding_sensitivity")
    return df


# =====================================================================
# Demo 数据（真实数据缺失时验证流程正确性）
# =====================================================================
def make_demo_data(config: CPCIv2Config):
    rng = np.random.RandomState(0)

    def make(n_per_class, p, sep=1.0):
        n = n_per_class * 2
        X = np.vstack([
            rng.normal(-sep / 2, 1, (n_per_class, p)),
            rng.normal(sep / 2, 1, (n_per_class, p))
        ])
        y = np.array([0] * n_per_class + [1] * n_per_class)
        idx = rng.permutation(n)
        return X[idx], y[idx]

    X_audio, y_audio = make(80, 13, sep=1.2)
    X_mri, y_mri = make(70, 7, sep=1.0)
    X_ppmi = rng.normal(0.5, 1.1, (200, 7))   # 站点偏移 + 全 PD
    y_ppmi = np.ones(200, dtype=int)
    return {"audio": (X_audio, y_audio), "mri": (X_mri, y_mri),
            "ppmi": (X_ppmi, y_ppmi)}


# =====================================================================
# 主程序
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="CPCI v2 防泄漏加权因果保形预测")
    parser.add_argument("--demo", action="store_true", help="使用合成演示数据")
    parser.add_argument("--seeds", type=int, default=None, help="随机种子数")
    parser.add_argument("--no-sensitivity", action="store_true",
                        help="跳过混杂强度敏感性实验")
    args = parser.parse_args()

    config = CPCIv2Config()
    if args.seeds:
        config.n_seeds = args.seeds
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    seeds = [config.random_state + i for i in range(config.n_seeds)]

    # ---------- 数据加载 ----------
    loader = DataLoader(config)

    # 启动路径自检：逐条打印配置的数据路径是否存在
    if not args.demo:
        logger.info("数据路径自检:")
        for label, p in [("audio_hc", config.audio_hc_dir),
                         ("audio_pd", config.audio_pd_dir),
                         ("mri_neurocon", config.mri_neurocon_dir),
                         ("mri_taowu", config.mri_taowu_dir),
                         ("mri_ppmi", config.mri_ppmi_dir),
                         ("ihdp", config.ihdp_path)]:
            logger.info(f"  {label:<14} -> {p}  [{'存在' if os.path.exists(p) else '不存在'}]")

    data: Dict[str, Tuple] = {}

    if not args.demo:
        X_a, y_a, _ = loader.load_audio_data()
        if len(X_a) > 0:
            data["audio"] = (X_a, y_a)
        X_m, y_m, _, sources = loader.load_mri_data(include_ppmi=True)
        n_ppmi = int(np.sum(sources == 1)) if len(sources) else 0
        if len(X_m) > 0 and np.sum(sources == 0) > 0:
            data["mri"] = (X_m[sources == 0], y_m[sources == 0])
            if n_ppmi > 0:
                data["ppmi"] = (X_m[sources == 1], y_m[sources == 1])

    if not data:
        logger.warning("未找到真实数据，自动切换为 --demo 合成数据模式")
        demo = make_demo_data(config)
        data["audio"] = demo["audio"]
        data["mri"] = demo["mri"]
        data["ppmi"] = demo["ppmi"]

    # ---------- 运行实验 ----------
    runner = ExperimentRunner(config)
    settings_results: Dict[str, Any] = {}

    if "audio" in data:
        X, y = data["audio"]
        logger.info(f"运行 audio 设定 (n={len(X)})")
        settings_results["audio"] = runner.run_setting(X, y, "audio", seeds, "audio")

    if "mri" in data:
        X, y = data["mri"]
        logger.info(f"运行 mri 设定 (n={len(X)})")
        settings_results["mri"] = runner.run_setting(X, y, "mri", seeds, "mri")

    if "ppmi" in data:
        X_int, y_int = data["mri"]
        X_ext, y_ext = data["ppmi"]
        logger.info(f"运行 external(PPMI) 设定 (internal n={len(X_int)}, external n={len(X_ext)})")
        settings_results["external"] = runner.run_setting(
            X_int, y_int, "mri", seeds, "external", X_external=(X_ext, y_ext))

    # IHDP 标准基准（可选）：文件存在则运行，不存在则跳过并提示
    ihdp_file = Path(config.ihdp_path)
    if ihdp_file.exists():
        try:
            X_i, y_i, t_i, ite_i = loader.load_ihdp(str(ihdp_file))
            logger.info(f"运行 ihdp 设定 (n={len(X_i)})")
            settings_results["ihdp"] = runner.run_setting(
                X_i, y_i, "ihdp", seeds, "ihdp", precomputed=(t_i, ite_i, y_i))
        except Exception as e:
            logger.warning(f"IHDP 加载失败，跳过该设定: {e}")
    else:
        logger.warning(f"未找到 IHDP 数据文件（{config.ihdp_path}），跳过 ihdp 设定；"
                       "投期刊版本建议放入该基准数据后重跑。")

    # ---------- 输出 ----------
    print_summary_table(settings_results)
    df = save_summary_csv(settings_results, config.output_dir)

    # JSON（仅指标，不含大数组）
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items() if k != "viz"}
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if isinstance(o, (tuple,)):
            return list(o)
        return o

    with open(Path(config.output_dir) / "results_summary.json", "w",
              encoding="utf-8") as f:
        json.dump(_clean(settings_results), f, indent=2, ensure_ascii=False)

    # ---------- 可视化 ----------
    # IHDP 只进 Table（论文图表保持 audio/mri/external 三设定，
    # 保证与已投稿版本的图一致；如需四设定图可去掉下面的过滤）
    viz_settings = {k: v for k, v in settings_results.items() if k != "ihdp"}
    viz = ResultVisualizer(config.output_dir)
    viz.plot_coverage_by_method(viz_settings)
    viz.plot_coverage_width_pareto(viz_settings)
    viz.plot_worst_group_coverage(viz_settings)
    main_setting = "mri" if "mri" in settings_results else list(settings_results)[0]
    viz.plot_calibration_curve(settings_results[main_setting], main_setting)
    viz.plot_width_distribution(settings_results[main_setting], main_setting)

    # ---------- 混杂强度敏感性实验（默认运行，可用 --no-sensitivity 跳过）----------
    if not args.no_sensitivity:
        run_confounding_sensitivity(config, data, seeds, runner)

    logger.info(f"\n全部结果已保存至: {config.output_dir}")
    logger.info("提示: 实验为半合成协议（校准使用模拟ITE），论文中请如实报告。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
