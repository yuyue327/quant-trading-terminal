#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step35_rf_with_real_llm.py
使用 step34 生成的真实 LLM 情感特征（llm_score）重新训练随机森林并回测
与旧特征（模拟 llm_score）进行对比
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# 配置
DATA_DIR = "data/features"
FEATURE_SEL_DIR = "data/feature_selection"
RESULT_DIR = "data/results"
FIGURE_DIR = "data/figures"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def load_individual_features(stock_name):
    json_path = os.path.join(FEATURE_SEL_DIR, "selected_features_per_stock.json")
    with open(json_path, 'r') as f:
        all_selected = json.load(f)
    if stock_name not in all_selected:
        raise KeyError(f"股票 {stock_name} 未找到")
    features = all_selected[stock_name]
    features = list(dict.fromkeys(features))
    return features


def load_stock_data_with_real_llm(stock_name, features):
    """加载包含真实 llm_score 的特征文件（step34 输出）"""
    file_path = os.path.join(DATA_DIR, f"{stock_name}_with_news.parquet")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"请先运行 step34 生成 {file_path}")
    df = pd.read_parquet(file_path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    for col in features:
        if col not in df.columns:
            df[col] = 0.0
    if 'label' not in df.columns:
        raise ValueError(f"{stock_name} 缺少 label 列")
    return df


def rolling_rf_predict(df, features, window=360, stride=20, calibrate=True):
    dates = []
    probs = []
    total_windows = len(range(window, len(df) - 1, stride))
    for test_idx in tqdm(range(window, len(df) - 1, stride), desc="滚动窗口"):
        train_start = test_idx - window
        train_df = df.iloc[train_start:test_idx]
        test_df = df.iloc[test_idx:test_idx + 1]
        X_train = train_df[features].values
        y_train = train_df['label'].values
        X_test = test_df[features].values
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        if calibrate:
            calibrated_rf = CalibratedClassifierCV(rf, method='sigmoid', cv=3)
            calibrated_rf.fit(X_train_scaled, y_train)
            prob = calibrated_rf.predict_proba(X_test_scaled)[0, 1]
        else:
            rf.fit(X_train_scaled, y_train)
            prob = rf.predict_proba(X_test_scaled)[0, 1]
        dates.append(df.index[test_idx])
        probs.append(prob)
    return pd.Series(probs, index=dates, name='prob')


def backtest_long_short(probs, df, long_thresh=0.55, short_thresh=0.45, transaction_cost=0.001):
    common_idx = probs.index.intersection(df.index)
    probs = probs.loc[common_idx]
    close = df.loc[common_idx, 'close']
    direction = np.zeros_like(probs)
    direction[probs > long_thresh] = 1
    direction[probs < short_thresh] = -1
    position_size = np.abs(probs - 0.5) * 2
    position = direction * position_size
    position = pd.Series(position, index=probs.index).shift(1).fillna(0)
    returns = close.pct_change().fillna(0)
    strategy_returns = position * returns
    trade_costs = position.diff().abs() * transaction_cost
    net_returns = strategy_returns - trade_costs
    nav = (1 + net_returns).cumprod() * 1e6
    bench_nav = (1 + returns).cumprod() * 1e6
    total_ret = nav.iloc[-1] / 1e6 - 1
    bench_ret = bench_nav.iloc[-1] / 1e6 - 1
    trading_days = len(nav)
    annual_ret = (1 + total_ret) ** (252 / trading_days) - 1 if total_ret > -1 else np.nan
    bench_annual = (1 + bench_ret) ** (252 / trading_days) - 1 if bench_ret > -1 else np.nan
    excess_ret = net_returns - 0.03 / 252
    sharpe = np.sqrt(252) * excess_ret.mean() / excess_ret.std() if excess_ret.std() != 0 else np.nan
    max_dd = (nav / nav.cummax() - 1).min()
    win_rate = (net_returns[net_returns != 0] > 0).mean() if (net_returns != 0).any() else 0
    trade_count = (position.diff().abs() > 0).sum()
    metrics = {
        '总收益率': f"{total_ret:.2%}",
        '基准收益率': f"{bench_ret:.2%}",
        '年化收益率': f"{annual_ret:.2%}",
        '基准年化': f"{bench_annual:.2%}",
        '夏普比率': f"{sharpe:.2f}",
        '最大回撤': f"{max_dd:.2%}",
        '胜率': f"{win_rate:.2%}",
        '交易次数': int(trade_count)
    }
    return nav, bench_nav, net_returns, metrics, position


def main():
    print("=" * 60)
    print("step35：使用真实 LLM 情感特征重新训练随机森林并回测")
    print("=" * 60)
    stock = "A_sh.600036_招商银行"
    print(f"加载股票 {stock} 的个性化特征...")
    features = load_individual_features(stock)
    print(f"特征数量: {len(features)}")
    # 确保 llm_score 在特征列表中
    if 'llm_score' not in features:
        features.append('llm_score')
        print("添加 llm_score 到特征列表")

    # 加载带真实 LLM 分数的数据
    df = load_stock_data_with_real_llm(stock, features)
    if 'high' not in df.columns:
        df['high'] = df['close'] * 1.02
    if 'low' not in df.columns:
        df['low'] = df['close'] * 0.98

    # 滚动预测
    cache_path = os.path.join(RESULT_DIR, "rf_probs_real_llm.csv")
    if os.path.exists(cache_path):
        print("加载缓存的预测概率...")
        probs = pd.read_csv(cache_path, index_col=0, parse_dates=True)['prob']
    else:
        print("开始滚动窗口随机森林预测（使用真实 LLM 特征）...")
        probs = rolling_rf_predict(df, features, window=360, stride=20, calibrate=True)
        probs.to_csv(cache_path, header=True)
        print(f"预测概率已保存至 {cache_path}")

    # 多空回测
    print("执行多空策略回测...")
    nav, bench_nav, net_returns, metrics, pos = backtest_long_short(
        probs, df, long_thresh=0.55, short_thresh=0.45, transaction_cost=0.001
    )

    print("\n=== 策略绩效指标（真实 LLM 情感） ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # 与旧结果对比（从 step32 的缓存加载）
    old_metrics_path = os.path.join(RESULT_DIR, "rf_strategy_metrics.csv")
    if os.path.exists(old_metrics_path):
        old_metrics = pd.read_csv(old_metrics_path).iloc[0]
        print("\n=== 性能对比（真实 LLM vs 模拟 LLM） ===")
        print(f"指标          模拟LLM      真实LLM     变化")
        print(
            f"夏普比率:      {old_metrics['夏普比率']}       {metrics['夏普比率']}       {float(metrics['夏普比率']) - float(old_metrics['夏普比率']):+.2f}")
        print(f"年化收益率:    {old_metrics['年化收益率']}    {metrics['年化收益率']}    ")
        print(f"最大回撤:      {old_metrics['最大回撤']}    {metrics['最大回撤']}    ")
        print(f"胜率:          {old_metrics['胜率']}    {metrics['胜率']}    ")

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(nav.index, nav, label='策略净值 (真实LLM)', linewidth=2)
    plt.plot(bench_nav.index, bench_nav, label='买入持有', linewidth=1.5, linestyle='--')
    plt.title('随机森林多空策略净值曲线（真实 LLM 情感）')
    plt.xlabel('日期')
    plt.ylabel('净值 (元)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(FIGURE_DIR, 'rf_strategy_nav_real_llm.png'), dpi=300)
    plt.close()

    # 保存指标
    pd.DataFrame([metrics]).to_csv(os.path.join(RESULT_DIR, "rf_strategy_metrics_real_llm.csv"), index=False)
    print("\nstep35 完成。")


if __name__ == "__main__":
    main()