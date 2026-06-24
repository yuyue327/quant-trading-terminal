#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step38_robust_multi_stock.py
稳健多股票策略：固定阈值(0.55/0.45) + 波动率目标仓位 + 1% 止损
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


def robust_backtest(probs, df, long_thresh=0.55, short_thresh=0.45,
                    vol_target=0.15, stop_loss=0.01, transaction_cost=0.001):
    """
    固定阈值 + 波动率目标仓位 + 固定止损 1%
    """
    common_idx = probs.index.intersection(df.index)
    probs = probs.loc[common_idx]
    close = df.loc[common_idx, 'close']
    returns = close.pct_change().fillna(0)

    # 方向
    direction = np.zeros_like(probs)
    direction[probs > long_thresh] = 1
    direction[probs < short_thresh] = -1
    # 仓位大小
    position_size = np.abs(probs - 0.5) * 2
    raw_position = direction * position_size

    # 波动率缩放
    vol = returns.rolling(20).std() * np.sqrt(252)
    vol_scaler = vol_target / vol.clip(lower=0.05, upper=0.5)
    vol_scaler = vol_scaler.fillna(1)
    adj_position = raw_position * vol_scaler
    adj_position = adj_position.clip(-0.5, 0.5)

    # 次日生效
    position = adj_position.shift(1).fillna(0)

    # 止损逻辑（基于收盘价回撤）
    stop_price = 0
    in_position = False
    final_pos = []
    for idx in position.index:
        pos = position.loc[idx]
        price = close.loc[idx]
        if not in_position and pos != 0:
            in_position = True
            entry_price = price
            stop_price = entry_price * (1 - stop_loss) if pos > 0 else entry_price * (1 + stop_loss)
            final_pos.append(pos)
        elif in_position:
            # 检查止损
            if (pos > 0 and price < stop_price) or (pos < 0 and price > stop_price):
                in_position = False
                final_pos.append(0)
            else:
                # 若信号变为0则平仓
                if pos == 0:
                    in_position = False
                    final_pos.append(0)
                else:
                    # 移动止损（有利方向）
                    if pos > 0:
                        stop_price = max(stop_price, price * (1 - stop_loss))
                    else:
                        stop_price = min(stop_price, price * (1 + stop_loss))
                    final_pos.append(pos)
        else:
            final_pos.append(0)
    final_position = pd.Series(final_pos, index=position.index)

    # 计算收益
    strategy_returns = final_position * returns
    trade_costs = final_position.diff().abs() * transaction_cost
    net_returns = strategy_returns - trade_costs
    nav = (1 + net_returns).cumprod() * 1e6
    bench_nav = (1 + returns).cumprod() * 1e6

    total_ret = nav.iloc[-1] / 1e6 - 1
    bench_ret = bench_nav.iloc[-1] / 1e6 - 1
    trading_days = len(nav)
    annual_ret = (1 + total_ret) ** (252 / trading_days) - 1 if total_ret > -1 else np.nan
    excess_ret = net_returns - 0.03 / 252
    sharpe = np.sqrt(252) * excess_ret.mean() / excess_ret.std() if excess_ret.std() != 0 else np.nan
    max_dd = (nav / nav.cummax() - 1).min()
    win_rate = (net_returns[net_returns != 0] > 0).mean() if (net_returns != 0).any() else 0
    trade_count = (final_position.diff().abs() > 0).sum()
    metrics = {
        '总收益率': f"{total_ret:.2%}",
        '年化收益率': f"{annual_ret:.2%}",
        '夏普比率': f"{sharpe:.2f}",
        '最大回撤': f"{max_dd:.2%}",
        '胜率': f"{win_rate:.2%}",
        '交易次数': int(trade_count)
    }
    return metrics


def main():
    print("=" * 60)
    print("step38：稳健多股票策略（固定阈值+波动率目标+止损）")
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
            metrics = robust_backtest(probs, df, long_thresh=0.55, short_thresh=0.45,
                                      vol_target=0.15, stop_loss=0.01, transaction_cost=0.001)
            metrics['股票'] = stock.split('_')[-1]
            all_results.append(metrics)
            print(f"  夏普比率: {metrics['夏普比率']}, 年化收益: {metrics['年化收益率']}")
        except Exception as e:
            print(f"  错误: {e}")
    results_df = pd.DataFrame(all_results)
    results_df = results_df[['股票', '夏普比率', '年化收益率', '最大回撤', '胜率', '交易次数']]
    results_df.to_csv(os.path.join(RESULT_DIR, "multi_stock_robust_results.csv"), index=False)
    print("\n=== 稳健策略多股票绩效汇总 ===")
    print(results_df.to_string(index=False))
    print(f"\n平均夏普比率: {results_df['夏普比率'].astype(float).mean():.2f}")
    print("step38 完成。")


if __name__ == "__main__":
    main()