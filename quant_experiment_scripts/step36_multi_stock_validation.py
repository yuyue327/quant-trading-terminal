#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step36_multi_stock_validation.py
多股票验证：对 step24 中的 6 只股票执行相同的随机森林策略，输出每只股票的绩效指标
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

# 股票列表（与 step24 一致）
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
        # 如果没找到，使用默认特征集（从 step27 的共同特征或全特征）
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
    dates = []
    probs = []
    for test_idx in range(window, len(df) - 1, stride):
        train_start = test_idx - window
        train_df = df.iloc[train_start:test_idx]
        test_df = df.iloc[test_idx:test_idx+1]
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
        '年化收益率': f"{annual_ret:.2%}",
        '夏普比率': f"{sharpe:.2f}",
        '最大回撤': f"{max_dd:.2%}",
        '胜率': f"{win_rate:.2%}",
        '交易次数': int(trade_count)
    }
    return metrics, nav

def main():
    print("="*60)
    print("step36：多股票验证（随机森林+多空策略）")
    print("="*60)
    all_results = []
    for stock in STOCKS:
        print(f"\n处理 {stock}...")
        try:
            features = load_individual_features(stock)
            print(f"  特征数量: {len(features)}")
            df = load_stock_data(stock, features)
            # 确保有足够数据
            if len(df) < 500:
                print(f"  数据长度不足 ({len(df)}), 跳过")
                continue
            cache_path = os.path.join(RESULT_DIR, f"rf_probs_{stock}.csv")
            if os.path.exists(cache_path):
                probs = pd.read_csv(cache_path, index_col=0, parse_dates=True)['prob']
            else:
                probs = rolling_rf_predict(df, features, window=360, stride=20)
                probs.to_csv(cache_path, header=True)
            metrics, nav = backtest_long_short(probs, df)
            metrics['股票'] = stock.split('_')[-1]
            all_results.append(metrics)
            print(f"  夏普比率: {metrics['夏普比率']}, 年化收益: {metrics['年化收益率']}")
        except Exception as e:
            print(f"  错误: {e}")
    results_df = pd.DataFrame(all_results)
    results_df = results_df[['股票', '夏普比率', '年化收益率', '最大回撤', '胜率', '交易次数']]
    results_df.to_csv(os.path.join(RESULT_DIR, "multi_stock_performance.csv"), index=False)
    print("\n=== 多股票绩效汇总 ===")
    print(results_df.to_string(index=False))
    print(f"\n平均夏普比率: {results_df['夏普比率'].astype(float).mean():.2f}")
    print("step36 完成。")

if __name__ == "__main__":
    main()