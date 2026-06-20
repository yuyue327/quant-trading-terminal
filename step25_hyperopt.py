#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤25：超参数优化 - 使用网格搜索寻找最佳策略参数
优化目标：最大化夏普比率
"""

import os
import itertools
import numpy as np
import pandas as pd
from step4_causal_attribution import load_data, FEATURE_COLS
from step22_robust_strategy import get_predictions_for_backtest, backtest_strategy

RESULT_DIR = "data/results"
os.makedirs(RESULT_DIR, exist_ok=True)

def grid_search():
    stock = "A_sh.600036_招商银行"
    df = load_data(stock)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
    if 'high' not in df.columns:
        df['high'] = df['close'] * 1.02
    if 'low' not in df.columns:
        df['low'] = df['close'] * 0.98

    features = FEATURE_COLS + ['llm_score']
    # 加载或生成概率
    probs = get_predictions_for_backtest(df, features, seq_len=20, window=60)
    common_idx = probs.index.intersection(df.index)
    probs = probs[common_idx]
    df_aligned = df.loc[common_idx]

    # 参数网格
    trend_windows = [15, 20, 25]
    max_positions = [0.4, 0.5, 0.6]
    atr_stops = [1.5, 2.0, 2.5]
    vol_targets = [0.12, 0.15, 0.18]

    best_sharpe = -np.inf
    best_params = None
    best_metrics = None
    results = []

    total_combos = len(trend_windows) * len(max_positions) * len(atr_stops) * len(vol_targets)
    count = 0
    for tw, mp, atr, vt in itertools.product(trend_windows, max_positions, atr_stops, vol_targets):
        count += 1
        print(f"测试组合 {count}/{total_combos}: tw={tw}, mp={mp}, atr={atr}, vt={vt}")
        try:
            _, _, _, metrics = backtest_strategy(
                probs, df_aligned,
                trend_window=tw,
                max_position=mp,
                atr_stop_mult=atr,
                volatility_target=vt,
                transaction_cost=0.001
            )
            sharpe = float(metrics['夏普比率'])
            results.append({
                'trend_window': tw,
                'max_position': mp,
                'atr_stop': atr,
                'vol_target': vt,
                'sharpe': sharpe,
                'total_return': metrics['总收益率'],
                'max_drawdown': metrics['最大回撤']
            })
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = (tw, mp, atr, vt)
                best_metrics = metrics
                print(f"  新最佳夏普: {sharpe:.2f}")
        except Exception as e:
            print(f"  失败: {e}")
            continue

    print("\n=== 最优参数 ===")
    print(f"trend_window={best_params[0]}, max_position={best_params[1]}, atr_stop={best_params[2]}, vol_target={best_params[3]}")
    for k, v in best_metrics.items():
        print(f"{k}: {v}")

    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULT_DIR, "hyperopt_results.csv"), index=False)
    print("\n超参数优化结果已保存")
    print("步骤25完成！")

if __name__ == "__main__":
    grid_search()