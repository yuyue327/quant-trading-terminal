#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step32_rf_strategy.py
随机森林 + 概率校准 + 多空策略（基于个性化特征）
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
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

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def load_individual_features(stock_name):
    """从 step27 结果加载个性化特征"""
    json_path = os.path.join(FEATURE_SEL_DIR, "selected_features_per_stock.json")
    with open(json_path, 'r') as f:
        all_selected = json.load(f)
    if stock_name not in all_selected:
        raise KeyError(f"股票 {stock_name} 未找到")
    features = all_selected[stock_name]
    features = list(dict.fromkeys(features))
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
    """
    滚动窗口随机森林预测，返回预测概率 Series（索引为预测日期）
    """
    dates = []
    probs = []
    total_windows = len(range(window, len(df) - 1, stride))
    for test_idx in tqdm(range(window, len(df) - 1, stride), desc="滚动窗口"):
        train_start = test_idx - window
        train_df = df.iloc[train_start:test_idx]
        test_df = df.iloc[test_idx:test_idx + 1]  # 单日测试

        X_train = train_df[features].values
        y_train = train_df['label'].values
        X_test = test_df[features].values

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 随机森林
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        if calibrate:
            # 概率校准（Platt scaling）
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
    """
    多空策略：概率 > long_thresh 做多，< short_thresh 做空，仓位 = |prob-0.5|*2
    """
    common_idx = probs.index.intersection(df.index)
    probs = probs.loc[common_idx]
    close = df.loc[common_idx, 'close']

    # 方向：1（做多），-1（做空），0（空仓）
    direction = np.zeros_like(probs)
    direction[probs > long_thresh] = 1
    direction[probs < short_thresh] = -1

    # 仓位大小：|prob-0.5|*2，映射到[0,1]
    position_size = np.abs(probs - 0.5) * 2
    position = direction * position_size
    position = pd.Series(position, index=probs.index).shift(1).fillna(0)

    # 日收益率
    returns = close.pct_change().fillna(0)
    strategy_returns = position * returns
    trade_costs = position.diff().abs() * transaction_cost
    net_returns = strategy_returns - trade_costs
    nav = (1 + net_returns).cumprod() * 1e6
    bench_nav = (1 + returns).cumprod() * 1e6

    # 绩效指标
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
    print("step32：随机森林 + 概率校准 + 多空策略")
    print("=" * 60)
    stock = "A_sh.600036_招商银行"
    print(f"加载股票 {stock} 的个性化特征...")
    features = load_individual_features(stock)
    print(f"特征数量: {len(features)}")
    print(f"特征列表: {features}")

    df = load_stock_data(stock, features)
    # 确保有 high/low 列（可选，本策略未使用）
    if 'high' not in df.columns:
        df['high'] = df['close'] * 1.02
    if 'low' not in df.columns:
        df['low'] = df['close'] * 0.98

    # 滚动预测（使用概率校准）
    cache_path = os.path.join(RESULT_DIR, "rf_probs.csv")
    if os.path.exists(cache_path):
        print("加载缓存的预测概率...")
        probs = pd.read_csv(cache_path, index_col=0, parse_dates=True)['prob']
    else:
        print("开始滚动窗口随机森林预测（约5分钟）...")
        probs = rolling_rf_predict(df, features, window=360, stride=20, calibrate=True)
        probs.to_csv(cache_path, header=True)
        print(f"预测概率已保存至 {cache_path}")

    # 多空回测（可调整阈值）
    print("执行多空策略回测...")
    nav, bench_nav, net_returns, metrics, pos = backtest_long_short(
        probs, df, long_thresh=0.55, short_thresh=0.45, transaction_cost=0.001
    )

    print("\n=== 策略绩效指标（随机森林+校准） ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(nav.index, nav, label='策略净值 (RF+多空)', linewidth=2)
    plt.plot(bench_nav.index, bench_nav, label='买入持有', linewidth=1.5, linestyle='--')
    plt.title('随机森林多空策略净值曲线')
    plt.xlabel('日期')
    plt.ylabel('净值 (元)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(FIGURE_DIR, 'rf_strategy_nav.png'), dpi=300)
    plt.close()

    # 预测概率分布
    plt.figure(figsize=(10, 4))
    plt.hist(probs, bins=50, alpha=0.7, color='blue')
    plt.axvline(0.5, color='red', linestyle='--')
    plt.title('随机森林预测概率分布')
    plt.xlabel('上涨概率')
    plt.ylabel('频次')
    plt.savefig(os.path.join(FIGURE_DIR, 'rf_prob_dist.png'), dpi=150)
    plt.close()

    # 保存指标
    pd.DataFrame([metrics]).to_csv(os.path.join(RESULT_DIR, "rf_strategy_metrics.csv"), index=False)
    print("\n所有图表已保存至 data/figures/")
    print("step32 完成。")


if __name__ == "__main__":
    main()