#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤17：模型解释与特征重要性分析
- 随机森林特征重要性（基于训练窗口）
- SHAP 值分析（使用部分样本）
- 市场状态划分（牛市/熊市/震荡），评估模型在各状态下的F1
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from step4_causal_attribution import load_data, FEATURE_COLS

RESULT_DIR = "data/results"
FIGURE_DIR = "data/figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

# 尝试导入 shap，如果没有安装则提示
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("警告：shap 未安装，将跳过 SHAP 分析。如需使用请运行 pip install shap")


def calculate_rf_feature_importance(df, features, window_size=2000):
    """使用最后一个训练窗口计算随机森林特征重要性"""
    # 取最后 window_size 个样本作为训练集（避免滚动，仅作示意）
    train_df = df.iloc[-window_size:]
    X = train_df[features].values
    y = train_df['label'].values
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    return importances, indices


def market_regime(df, window=60):
    """
    根据滚动窗口的收益率划分市场状态：
    - 牛市：窗口收益率 > 2%
    - 熊市：窗口收益率 < -2%
    - 震荡：其余
    """
    returns = df['close'].pct_change().fillna(0)
    regime = pd.Series('震荡', index=df.index)
    for i in range(window, len(df)):
        win_ret = returns.iloc[i - window:i].mean() * 252  # 年化
        if win_ret > 0.02:
            regime.iloc[i] = '牛市'
        elif win_ret < -0.02:
            regime.iloc[i] = '熊市'
    return regime


def evaluate_by_regime(df, features, regime_series, seq_len=20, window=60):
    """简化评估：使用滚动窗口的RF（非序列）预测，按市场状态计算F1"""
    preds = []
    true = []
    regimes = []
    total_points = len(df) - (window + seq_len + 1)
    for idx, i in enumerate(range(window + seq_len, len(df) - 1)):
        train_end = i - seq_len
        train_start = train_end - window
        train_df = df.iloc[train_start:train_end]
        X_train = train_df[features].values
        y_train = train_df['label'].values
        X_test = df[features].iloc[i].values.reshape(1, -1)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)[0]
        preds.append(pred)
        true.append(df['label'].iloc[i + 1])
        regimes.append(regime_series.iloc[i + 1])
    # 计算各状态的F1
    results = {}
    for regime in ['牛市', '熊市', '震荡']:
        mask = np.array(regimes) == regime
        if mask.sum() > 0:
            f1 = f1_score(np.array(true)[mask], np.array(preds)[mask], zero_division=0)
            results[regime] = f1
        else:
            results[regime] = np.nan
    return results


def main():
    print("=" * 60)
    print("步骤17：模型解释与特征重要性分析")
    print("=" * 60)

    stock = "A_sh.600036_招商银行"
    df = load_data(stock)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)

    features = FEATURE_COLS + ['llm_score']
    feature_names = features  # 可直接使用

    # 1. 随机森林特征重要性
    print("\n计算随机森林特征重要性（基于最后2000个样本）...")
    importances, indices = calculate_rf_feature_importance(df, features, window_size=2000)
    top_n = 15
    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), importances[indices[:top_n]][::-1], align='center')
    plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]][::-1])
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Feature Importance (Top 15)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'feature_importance.png'), dpi=300)
    print(f"特征重要性图保存至：{FIGURE_DIR}/feature_importance.png")

    # 2. SHAP 分析（可选）
    if SHAP_AVAILABLE:
        print("\n进行 SHAP 分析（可能需要几分钟）...")
        # 取一部分数据作为背景和解释集
        background_df = df.iloc[-500:]
        X_background = background_df[features].values
        rf_explainer = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_explainer.fit(X_background, background_df['label'].values)
        # 使用 KernelExplainer 或 TreeExplainer
        explainer = shap.TreeExplainer(rf_explainer)
        X_sample = df.iloc[-200:][features].values
        shap_values = explainer.shap_values(X_sample)[1]  # 类别1的SHAP值
        # 汇总图
        plt.figure()
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, 'shap_summary.png'), dpi=300, bbox_inches='tight')
        print(f"SHAP 汇总图保存至：{FIGURE_DIR}/shap_summary.png")
        # 条形图
        plt.figure()
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, 'shap_bar.png'), dpi=300, bbox_inches='tight')
        print(f"SHAP 条形图保存至：{FIGURE_DIR}/shap_bar.png")
    else:
        print("\n跳过 SHAP 分析（未安装 shap）")

    # 3. 市场状态划分与各状态F1评估
    print("\n划分市场状态（牛市/熊市/震荡）...")
    regime_series = market_regime(df, window=60)
    print("各状态样本数：")
    print(regime_series.value_counts())

    print("\n评估模型在不同市场状态下的F1（基于RF滚动窗口）...")
    regime_f1 = evaluate_by_regime(df, features, regime_series, seq_len=20, window=60)
    regime_df = pd.DataFrame([regime_f1]).T
    regime_df.columns = ['F1 Score']
    regime_df.to_csv(os.path.join(RESULT_DIR, "regime_performance.csv"))
    print(regime_df)

    # 绘制柱状图
    plt.figure(figsize=(6, 4))
    regimes_plot = ['牛市', '熊市', '震荡']
    f1_vals = [regime_f1[r] for r in regimes_plot]
    colors = ['green', 'red', 'gray']
    bars = plt.bar(regimes_plot, f1_vals, color=colors)
    plt.ylabel('F1 Score')
    plt.title('Model Performance by Market Regime')
    plt.ylim(0, 0.6)
    for bar, val in zip(bars, f1_vals):
        if not np.isnan(val):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{val:.3f}', ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'regime_f1.png'), dpi=300)
    print(f"市场状态F1图保存至：{FIGURE_DIR}/regime_f1.png")

    print("\n步骤17完成！")


if __name__ == "__main__":
    main()