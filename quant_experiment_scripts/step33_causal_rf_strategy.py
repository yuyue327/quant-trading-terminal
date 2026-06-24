#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step33_causal_rf_strategy.py
使用因果特征子集（基于互信息+树重要性筛选的top-k）重新训练随机森林策略
并与全特征对比，生成论文所需对比表格和图表
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


def load_individual_features(stock_name, top_k=None):
    """加载个性化特征，可选只保留前 top_k 个（按重要性排序）"""
    json_path = os.path.join(FEATURE_SEL_DIR, "selected_features_per_stock.json")
    with open(json_path, 'r') as f:
        all_selected = json.load(f)
    features = all_selected[stock_name]
    # 按重要性排序（已在 step27 中按综合得分排序，直接取前 top_k）
    if top_k is not None and top_k < len(features):
        features = features[:top_k]
    return features


def load_stock_data(stock_name, features):
    file_path = os.path.join(DATA_DIR, f"{stock_name}.parquet")
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
    """滚动窗口随机森林预测"""
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
    print("step33：因果特征子集 vs 全特征对比实验")
    print("=" * 60)
    stock = "A_sh.600036_招商银行"

    # 全特征（15个）
    features_full = load_individual_features(stock)
    # 因果特征子集（取前 8 个最重要的，模拟因果筛选；可根据 step27 的综合得分取）
    features_causal = load_individual_features(stock, top_k=8)
    print(f"全特征数量: {len(features_full)}")
    print(f"因果子集数量: {len(features_causal)}")
    print(f"因果子集: {features_causal}")

    df = load_stock_data(stock, features_full)
    if 'high' not in df.columns:
        df['high'] = df['close'] * 1.02
    if 'low' not in df.columns:
        df['low'] = df['close'] * 0.98

    # 全特征预测（已有缓存则直接加载）
    cache_full = os.path.join(RESULT_DIR, "rf_probs_full.csv")
    if os.path.exists(cache_full):
        probs_full = pd.read_csv(cache_full, index_col=0, parse_dates=True)['prob']
    else:
        print("训练全特征模型...")
        probs_full = rolling_rf_predict(df, features_full, window=360, stride=20, calibrate=True)
        probs_full.to_csv(cache_full, header=True)

    # 因果子集预测
    cache_causal = os.path.join(RESULT_DIR, "rf_probs_causal.csv")
    if os.path.exists(cache_causal):
        probs_causal = pd.read_csv(cache_causal, index_col=0, parse_dates=True)['prob']
    else:
        print("训练因果子集模型...")
        probs_causal = rolling_rf_predict(df, features_causal, window=360, stride=20, calibrate=True)
        probs_causal.to_csv(cache_causal, header=True)

    # 回测全特征
    print("\n回测全特征策略...")
    nav_full, _, _, metrics_full, _ = backtest_long_short(probs_full, df, long_thresh=0.55, short_thresh=0.45)
    # 回测因果子集
    print("回测因果子集策略...")
    nav_causal, _, _, metrics_causal, _ = backtest_long_short(probs_causal, df, long_thresh=0.55, short_thresh=0.45)

    # 输出对比
    print("\n=== 绩效对比 ===")
    print("指标\t\t全特征\t\t因果子集")
    print("-" * 50)
    for key in ['夏普比率', '年化收益率', '最大回撤', '胜率', '交易次数']:
        print(f"{key}:\t\t{metrics_full[key]}\t\t{metrics_causal[key]}")

    # 保存对比结果
    compare_df = pd.DataFrame({
        'Features': ['Full (15)', 'Causal (8)'],
        'Sharpe': [float(metrics_full['夏普比率']), float(metrics_causal['夏普比率'])],
        'Annual_Ret': [float(metrics_full['年化收益率'].strip('%')), float(metrics_causal['年化收益率'].strip('%'))],
        'Max_DD': [float(metrics_full['最大回撤'].strip('%')), float(metrics_causal['最大回撤'].strip('%'))],
        'Win_Rate': [float(metrics_full['胜率'].strip('%')), float(metrics_causal['胜率'].strip('%'))],
        'Trades': [metrics_full['交易次数'], metrics_causal['交易次数']]
    })
    compare_df.to_csv(os.path.join(RESULT_DIR, "causal_ablation.csv"), index=False)

    # 绘制净值曲线对比
    plt.figure(figsize=(12, 6))
    plt.plot(nav_full.index, nav_full, label='全特征 (15个)', linewidth=2)
    plt.plot(nav_causal.index, nav_causal, label='因果子集 (8个)', linewidth=2, linestyle='--')
    plt.plot((1 + df['close'].pct_change().fillna(0)).cumprod() * 1e6, label='买入持有', linewidth=1, linestyle=':')
    plt.title('策略净值曲线对比：全特征 vs 因果子集')
    plt.xlabel('日期')
    plt.ylabel('净值 (元)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(FIGURE_DIR, 'causal_ablation_nav.png'), dpi=300)
    plt.close()

    # 夏普比率柱状图
    plt.figure(figsize=(6, 5))
    bars = plt.bar(['全特征', '因果子集'], [float(metrics_full['夏普比率']), float(metrics_causal['夏普比率'])],
                   color=['blue', 'green'])
    plt.axhline(0, color='red', linestyle='--')
    plt.title('夏普比率对比')
    plt.ylabel('夏普比率')
    for bar, val in zip(bars, [float(metrics_full['夏普比率']), float(metrics_causal['夏普比率'])]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f'{val:.2f}', ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'causal_sharpe_compare.png'), dpi=300)
    plt.close()

    print("\n对比图表已保存至 data/figures/")
    print("step33 完成。")


if __name__ == "__main__":
    main()