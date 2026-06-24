#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段8：鲁棒性与泛化终极验证
1. 对抗攻击测试（对技术指标添加微小扰动）
2. 时间分布偏移测试（跨时间验证）
3. 零样本跨市场（A股训练→美股预测）
4. 消融实验矩阵
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
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

def rolling_evaluation_f1(df, features, window, use_llm=True):
    """滚动窗口评估，返回平均F1"""
    f1_list = []
    for i in range(window, len(df) - 1):
        train_idx = list(range(i-window, i))
        test_idx = [i+1]
        X_train = df.iloc[train_idx][features].values
        y_train = df.iloc[train_idx]['label'].values
        X_test = df.iloc[test_idx][features].values
        y_test = df.iloc[test_idx]['label'].values
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1_list.append(f1_score(y_test, y_pred, zero_division=0))
    return np.mean(f1_list)

def adversarial_attack(df, features, window, epsilon=0.01):
    """对抗攻击：对技术指标添加微小高斯噪声"""
    df_attacked = df.copy()
    for col in features:
        if col != 'llm_score':
            noise = np.random.normal(0, epsilon * df[col].std(), size=len(df))
            df_attacked[col] = df[col] + noise
    f1_original = rolling_evaluation_f1(df, features + ['llm_score'], window)
    f1_attacked = rolling_evaluation_f1(df_attacked, features + ['llm_score'], window)
    return f1_original, f1_attacked, (f1_original - f1_attacked) / f1_original

def time_shift_validation(stock_name, split_date='2023-12-31'):
    """时间分布偏移测试：用前一段训练，后一段测试"""
    df = load_data(stock_name)
    train_df = df[df['date'] <= split_date]
    test_df = df[df['date'] > split_date]

    features = FEATURE_COLS + ['llm_score']
    window = 60

    # 在训练集上训练最终模型
    X_train = train_df[features].values
    y_train = train_df['label'].values
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # 在测试集上滚动评估
    preds = []
    for i in range(window, len(test_df)):
        X_test = test_df.iloc[i][features].values.reshape(1, -1)
        y_test = test_df.iloc[i]['label']
        pred = clf.predict(X_test)[0]
        preds.append(pred)
    y_true = test_df['label'].iloc[window:].values
    return f1_score(y_true, preds, zero_division=0)

def zero_shot_cross_market(train_stocks, test_stocks):
    """零样本跨市场：在A股上训练，直接在美股上测试（不微调）"""
    # 合并训练数据
    X_train_list, y_train_list = [], []
    for s in train_stocks:
        df = load_data(s)
        features = FEATURE_COLS + ['llm_score']
        X_train_list.append(df[features].values)
        y_train_list.append(df['label'].values)
    X_train = np.vstack(X_train_list)
    y_train = np.hstack(y_train_list)

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    results = []
    for s in test_stocks:
        df = load_data(s)
        X_test = df[features].values
        y_test = df['label'].values
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        results.append({'stock': s, 'f1_zero_shot': f1})
    return pd.DataFrame(results)

def ablation_matrix():
    """消融实验：移除不同特征组"""
    stock = 'A_sh.600036_招商银行'
    df = load_data(stock)
    window = 60

    groups = {
        'full': FEATURE_COLS + ['llm_score'],
        'no_llm': FEATURE_COLS,
        'no_momentum': [c for c in FEATURE_COLS if c not in ['MACD', 'MACD_signal', 'MACD_hist', 'RSI']] + ['llm_score'],
        'no_volatility': [c for c in FEATURE_COLS if c not in ['volatility_5', 'volatility_20', 'ATR']] + ['llm_score'],
        'no_volume': [c for c in FEATURE_COLS if c not in ['volume_ratio']] + ['llm_score'],
        'no_bb': [c for c in FEATURE_COLS if not c.startswith('BB_')] + ['llm_score'],
    }

    results = []
    for name, feats in groups.items():
        f1 = rolling_evaluation_f1(df, feats, window)
        results.append({'group': name, 'f1': f1})
        print(f"  {name}: F1={f1:.4f}")

    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(RESULT_DIR, "ablation_matrix.csv"), index=False)

    # 可视化
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_res, x='group', y='f1')
    plt.title('Ablation Study: Feature Group Importance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'ablation_matrix.png'), dpi=150)
    plt.close()
    return df_res

def run_all_robustness():
    print("="*60)
    print("阶段8：鲁棒性与泛化终极验证")
    print("="*60)

    # 1. 对抗攻击测试
    print("\n=== 1. 对抗攻击测试 ===")
    stock = 'A_sh.600036_招商银行'
    df = load_data(stock)
    f1_orig, f1_adv, drop_rate = adversarial_attack(df, FEATURE_COLS, window=60, epsilon=0.01)
    print(f"  原始F1: {f1_orig:.4f}, 攻击后F1: {f1_adv:.4f}, 下降率: {drop_rate*100:.2f}%")

    # 2. 时间偏移测试
    print("\n=== 2. 时间分布偏移测试 ===")
    f1_shift = time_shift_validation(stock, split_date='2023-12-31')
    print(f"  跨时间验证F1: {f1_shift:.4f}")

    # 3. 零样本跨市场
    print("\n=== 3. 零样本跨市场（A股→美股） ===")
    train_stocks = [s for s in STOCKS.keys() if not s.startswith('US_')]
    test_stocks = [s for s in STOCKS.keys() if s.startswith('US_')]
    df_cross = zero_shot_cross_market(train_stocks, test_stocks)
    print(df_cross)
    df_cross.to_csv(os.path.join(RESULT_DIR, "zero_shot_cross_market.csv"), index=False)

    # 4. 消融实验矩阵
    print("\n=== 4. 消融实验矩阵 ===")
    ablation_matrix()

    # 汇总鲁棒性报告
    summary = pd.DataFrame({
        'test': ['Adversarial Attack', 'Time Shift', 'Zero-shot Cross-market (mean)'],
        'result': [f"{drop_rate*100:.2f}% drop", f"{f1_shift:.4f}", f"{df_cross['f1_zero_shot'].mean():.4f}"]
    })
    summary.to_csv(os.path.join(RESULT_DIR, "robustness_summary.csv"), index=False)
    print("\n鲁棒性汇总:")
    print(summary)

    print("\n阶段8完成！")

if __name__ == "__main__":
    run_all_robustness()