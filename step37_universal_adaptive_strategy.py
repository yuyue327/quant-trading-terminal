#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step37_universal_adaptive_strategy.py
自适应阈值 + 波动率目标仓位，提升多股票策略的泛化能力
"""

import os
import json
import numpy as np
import pandas as pd
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
os.makedirs(RESULT_DIR, exist_ok=True)

STOCKS = [
    "A_sh.600036_招商银行",
    "A_sz.000001_平安银行",
    "A_sh.600030_中信证券",
    "A_sh.600519_贵州茅台",
    "A_sz.000858_五粮液",
    "A_sz.300750_宁德时代"
]


def load_individual_features(stock_name):
    json_path = os.path.join(FEATURE_SEL_DIR, "selected_features_per_stock.json")
    with open(json_path, 'r') as f:
        all_selected = json.load(f)
    if stock_name not in all_selected:
        return ['close', 'volume', 'MA5', 'RSI', 'MACD', 'ATR', 'volume_ratio', 'pct_change', 'llm_score']
    features = all_selected[stock_name]
    features = list(dict.fromkeys(features))
    if 'llm_score' not in features:
        features.append('llm_score')
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


def rolling_rf_predict(df, features, window=360, stride=20):
    dates, probs = [], []
    for test_idx in range(window, len(df) - 1, stride):
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
        calibrated_rf = CalibratedClassifierCV(rf, method='sigmoid', cv=3)
        calibrated_rf.fit(X_train_scaled, y_train)
        prob = calibrated_rf.predict_proba(X_test_scaled)[0, 1]
        dates.append(df.index[test_idx])
        probs.append(prob)
    return pd.Series(probs, index=dates, name='prob')


def adaptive_backtest(probs, df, vol_target=0.15, lookback=60):
    """
    自适应阈值 + 波动率目标仓位
    - 每月根据过去 lookback 天的预测概率与未来收益，寻找最优阈值
    - 仓位 = 方向 * 仓位大小 * (vol_target / 实际波动率)
    """
    common_idx = probs.index.intersection(df.index)
    probs = probs.loc[common_idx]
    close = df.loc[common_idx, 'close']
    returns = close.pct_change().fillna(0)

    # 滚动寻找最优阈值（每月重调）
    position = pd.Series(0.0, index=probs.index)
    for month_start in probs.resample('M').first().index:
        # 训练阈值的数据：当前月之前 lookback 天
        train_start = month_start - pd.Timedelta(days=lookback)
        train_probs = probs[(probs.index >= train_start) & (probs.index < month_start)]
        train_returns = returns[(returns.index >= train_start) & (returns.index < month_start)]
        if len(train_probs) < 20:
            continue
        best_sharpe = -np.inf
        best_thresh = (0.55, 0.45)
        for long_th in np.arange(0.51, 0.65, 0.02):
            for short_th in np.arange(0.35, 0.49, 0.02):
                direction = np.zeros_like(train_probs)
                direction[train_probs > long_th] = 1
                direction[train_probs < short_th] = -1
                pos_size = np.abs(train_probs - 0.5) * 2
                test_pos = direction * pos_size
                test_pos = pd.Series(test_pos, index=train_probs.index).shift(1).fillna(0)
                strat_ret = test_pos * train_returns
                if strat_ret.std() == 0:
                    continue
                sharpe = np.sqrt(252) * strat_ret.mean() / strat_ret.std()
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_thresh = (long_th, short_th)
        # 当月使用最优阈值
        month_probs = probs[(probs.index >= month_start) & (probs.index < month_start + pd.Timedelta(days=32))]
        if len(month_probs) == 0:
            continue
        direction = np.zeros_like(month_probs)
        direction[month_probs > best_thresh[0]] = 1
        direction[month_probs < best_thresh[1]] = -1
        pos_size = np.abs(month_probs - 0.5) * 2
        raw_pos = direction * pos_size
        # 波动率缩放
        vol = returns.rolling(20).std() * np.sqrt(252)
        vol_scaler = vol_target / vol.clip(lower=0.05, upper=0.5)
        adj_pos = raw_pos * vol_scaler.loc[month_probs.index]
        adj_pos = adj_pos.clip(-0.5, 0.5)
        position.loc[month_probs.index] = adj_pos.shift(1).fillna(0)

    # 计算最终绩效
    daily_returns = returns
    strategy_returns = position * daily_returns
    # 简化交易成本（双边0.2%）
    trade_costs = position.diff().abs() * 0.002
    net_returns = strategy_returns - trade_costs
    nav = (1 + net_returns).cumprod() * 1e6
    bench_nav = (1 + daily_returns).cumprod() * 1e6
    total_ret = nav.iloc[-1] / 1e6 - 1
    bench_ret = bench_nav.iloc[-1] / 1e6 - 1
    trading_days = len(nav)
    annual_ret = (1 + total_ret) ** (252 / trading_days) - 1 if total_ret > -1 else np.nan
    excess_ret = net_returns - 0.03 / 252
    sharpe = np.sqrt(252) * excess_ret.mean() / excess_ret.std() if excess_ret.std() != 0 else np.nan
    max_dd = (nav / nav.cummax() - 1).min()
    win_rate = (net_returns[net_returns != 0] > 0).mean() if (net_returns != 0).any() else 0
    trade_count = (position.diff().abs() > 0).sum()
    metrics = {
        '总收益率': f"{total_ret:.2%}",
        '年化收益率': f"{annual_ret:.2%}",
        '夏普比率': f"{sharpe:.2f}",
        '最大回撤': f"{max_dd:.2%}",
        '胜率': f"{win_rate:.2%}",
        '交易次数': int(trade_count)
    }
    return metrics, nav


def main():
    print("=" * 60)
    print("step37：自适应阈值 + 波动率目标策略（多股票）")
    print("=" * 60)
    all_results = []
    for stock in STOCKS:
        print(f"\n处理 {stock}...")
        try:
            features = load_individual_features(stock)
            df = load_stock_data(stock, features)
            cache_path = os.path.join(RESULT_DIR, f"rf_probs_{stock}.csv")
            if not os.path.exists(cache_path):
                print("  生成预测概率...")
                probs = rolling_rf_predict(df, features, window=360, stride=20)
                probs.to_csv(cache_path)
            else:
                probs = pd.read_csv(cache_path, index_col=0, parse_dates=True)['prob']
            metrics, _ = adaptive_backtest(probs, df, vol_target=0.15, lookback=60)
            metrics['股票'] = stock.split('_')[-1]
            all_results.append(metrics)
            print(f"  夏普比率: {metrics['夏普比率']}, 年化收益: {metrics['年化收益率']}")
        except Exception as e:
            print(f"  错误: {e}")
    results_df = pd.DataFrame(all_results)
    results_df = results_df[['股票', '夏普比率', '年化收益率', '最大回撤', '胜率', '交易次数']]
    results_df.to_csv(os.path.join(RESULT_DIR, "multi_stock_adaptive_results.csv"), index=False)
    print("\n=== 多股票自适应策略绩效汇总 ===")
    print(results_df.to_string(index=False))
    print(f"\n平均夏普比率: {results_df['夏普比率'].astype(float).mean():.2f}")
    print("step37 完成。")


if __name__ == "__main__":
    main()