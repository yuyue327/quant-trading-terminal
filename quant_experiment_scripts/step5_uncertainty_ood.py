#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段5：不确定性量化与分布外检测
1. LLM置信度校准（温度缩放）
2. 蒙特卡洛dropout估计预测不确定性
3. 分布外检测（马氏距离 / 能量分数）
4. 不确定性感知的门控加权
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance
from scipy.spatial.distance import mahalanobis
from scipy.special import softmax
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ========== 配置 ==========
FEATURE_DIR = "data/features"
LLM_DIR = "data/llm_scores"
RESULT_DIR = "data/results"
FIGURE_DIR = "data/figures"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

STOCKS = {
    'A_sh.600036_招商银行': 'bank',
    'A_sz.000001_平安银行': 'bank',
    'A_sz.002142_宁波银行': 'bank',
    'A_sh.600030_中信证券': 'securities',
    'A_sh.601688_华泰证券': 'securities',
    'A_sz.300059_东方财富': 'securities',
    'A_sh.600519_贵州茅台': 'liquor',
    'A_sz.000858_五粮液': 'liquor',
    'A_sz.000568_泸州老窖': 'liquor',
    'A_sz.000333_美的集团': 'consumer',
    'A_sz.000651_格力电器': 'consumer',
    'A_sh.600887_伊利股份': 'consumer',
    'A_sz.300750_宁德时代': 'new_energy',
    'A_sz.002594_比亚迪': 'new_energy',
    'A_sh.601012_隆基绿能': 'new_energy',
    'US_AAPL_AAPL': 'us_tech',
    'US_MSFT_MSFT': 'us_tech',
    'US_NVDA_NVDA': 'us_tech',
}

FEATURE_COLS = [
    'MA5', 'MA10', 'MA20', 'MA60', 'EMA12', 'EMA26',
    'MACD', 'MACD_signal', 'MACD_hist', 'RSI',
    'BB_upper', 'BB_middle', 'BB_lower', 'BB_width', 'BB_pct',
    'ATR', 'volume_ratio', 'pct_change', 'high_low_pct',
    'close_position', 'volatility_5', 'volatility_20'
]


def load_data(stock_name):
    feat_path = os.path.join(FEATURE_DIR, f"{stock_name}.parquet")
    llm_path = os.path.join(LLM_DIR, f"{stock_name}.parquet")
    df_feat = pd.read_parquet(feat_path)
    df_llm = pd.read_parquet(llm_path)
    return df_feat.merge(df_llm, on='date', how='inner')


def temperature_scaling(df, features, target='label', n_bins=10):
    """
    温度缩放校准LLM分数（将LLM分数视为概率）
    实际LLM分数是0-100整数，不是概率，这里将其归一化到[0,1]后做温度缩放
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    # 将LLM分数归一化
    llm_prob = df['llm_score'] / 100.0
    y = df[target].values

    # 使用逻辑回归的温度缩放（简化版：拟合一个缩放系数）
    # 更严谨的做法是用验证集优化温度参数T
    from sklearn.calibration import CalibratedClassifierCV
    # 这里我们直接计算经验校准曲线
    prob_true, prob_pred = calibration_curve(y, llm_prob, n_bins=n_bins)

    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='LLM')
    plt.plot([0, 1], [0, 1], '--', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'calibration_curve.png'), dpi=150)
    plt.close()

    # 计算ECE（Expected Calibration Error）
    ece = np.mean(np.abs(prob_true - prob_pred))
    print(f"  Expected Calibration Error (ECE): {ece:.4f}")
    return ece


def monte_carlo_dropout_uncertainty(df, features, target='label', n_iter=30, dropout_rate=0.1):
    """
    蒙特卡洛dropout估计预测不确定性
    对随机森林使用子采样模拟dropout效果（每次训练时随机丢弃部分特征）
    """
    X = df[features].values
    y = df[target].values
    predictions = []

    for _ in range(n_iter):
        # 随机丢弃20%的特征（模拟dropout）
        mask = np.random.binomial(1, 1 - dropout_rate, size=X.shape[1]).astype(bool)
        X_masked = X[:, mask]
        clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=None, n_jobs=-1)
        # 使用前80%训练，后20%预测（简化）
        split = int(0.8 * len(X))
        clf.fit(X_masked[:split], y[:split])
        pred_prob = clf.predict_proba(X_masked[split:])[:, 1]
        predictions.append(pred_prob)

    predictions = np.array(predictions)
    pred_mean = predictions.mean(axis=0)
    pred_std = predictions.std(axis=0)
    uncertainty = pred_std  # 标准差作为不确定性度量

    # 可视化不确定性分布
    plt.figure(figsize=(10, 4))
    plt.hist(uncertainty, bins=50, alpha=0.7)
    plt.title('Distribution of MC Dropout Uncertainty')
    plt.xlabel('Uncertainty (std of predictions)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'mc_uncertainty_dist.png'), dpi=150)
    plt.close()

    return uncertainty


def mahalanobis_ood(X_train, X_test):
    """马氏距离分布外检测"""
    cov = EmpiricalCovariance().fit(X_train)
    distances = [mahalanobis(x, X_train.mean(axis=0), cov.covariance_) for x in X_test]
    return np.array(distances)


def energy_score_ood(logits, temperature=1.0):
    """能量分数分布外检测（需要分类器输出logits）"""
    # 随机森林可以输出概率，取logits = log(prob)
    probs = np.array(logits)
    logits = np.log(probs + 1e-9)
    energy = temperature * np.log(np.sum(np.exp(logits / temperature), axis=1))
    return energy


def uncertainty_aware_gating(df, features, llm_uncertainty, ood_score, threshold_unc=0.2, threshold_ood=50):
    """
    不确定性感知门控：根据LLM不确定性和分布外分数决定是否使用LLM
    """
    # 归一化
    unc_norm = (llm_uncertainty - llm_uncertainty.min()) / (llm_uncertainty.max() - llm_uncertainty.min() + 1e-9)
    ood_norm = (ood_score - ood_score.min()) / (ood_score.max() - ood_score.min() + 1e-9)

    # 门控分数：不确定性和OOD越高，越不信任LLM
    gate_score = 1 - (unc_norm + ood_norm) / 2
    gate_score = np.clip(gate_score, 0, 1)
    return gate_score


def run_all_uncertainty_analysis():
    print("=" * 60)
    print("阶段5：不确定性量化与分布外检测")
    print("=" * 60)

    # 1. 温度缩放校准（以招商银行为例）
    print("\n=== 1. LLM分数校准 ===")
    stock = 'A_sh.600036_招商银行'
    df = load_data(stock)
    ece = temperature_scaling(df, FEATURE_COLS)

    # 2. 蒙特卡洛dropout不确定性估计
    print("\n=== 2. MC Dropout不确定性估计 ===")
    uncertainty = monte_carlo_dropout_uncertainty(df, FEATURE_COLS)
    df['uncertainty'] = np.nan
    split = int(0.8 * len(df))
    df.iloc[split:, df.columns.get_loc('uncertainty')] = uncertainty

    # 3. 分布外检测
    print("\n=== 3. 分布外检测 ===")
    X = df[FEATURE_COLS].values
    split = int(0.8 * len(X))
    X_train = X[:split]
    X_test = X[split:]
    ood_distances = mahalanobis_ood(X_train, X_test)

    # 4. 不确定性感知门控
    print("\n=== 4. 不确定性感知门控 ===")
    gate_weights = uncertainty_aware_gating(df.iloc[split:], FEATURE_COLS, uncertainty, ood_distances)

    # 5. 可视化
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.hist(uncertainty, bins=30, alpha=0.7)
    plt.title('Uncertainty Distribution (MC Dropout)')

    plt.subplot(2, 2, 2)
    plt.hist(ood_distances, bins=30, alpha=0.7)
    plt.title('Mahalanobis Distance (OOD Score)')

    plt.subplot(2, 2, 3)
    plt.scatter(uncertainty, ood_distances, alpha=0.5)
    plt.xlabel('Uncertainty')
    plt.ylabel('OOD Distance')
    plt.title('Uncertainty vs OOD')

    plt.subplot(2, 2, 4)
    plt.plot(gate_weights, alpha=0.7)
    plt.title('Gate Weights (Higher = More Trust LLM)')
    plt.xlabel('Time step')
    plt.ylabel('Gate weight')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'uncertainty_ood_gate.png'), dpi=150)
    plt.close()

    # 保存结果
    results = pd.DataFrame({
        'uncertainty': uncertainty,
        'ood_distance': ood_distances,
        'gate_weight': gate_weights
    })
    results.to_csv(os.path.join(RESULT_DIR, "uncertainty_ood_results.csv"), index=False)

    print("\n阶段5完成！结果保存在 data/results/ 和 data/figures/")


if __name__ == "__main__":
    run_all_uncertainty_analysis()