#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤20：改进策略（做空 + 动态仓位）
- 基于预测概率：概率>0.55做多（仓位=概率），概率<0.45做空（仓位=1-概率），其余空仓
- 回测并绘制改进后的净值曲线
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from step4_causal_attribution import load_data, FEATURE_COLS
from step15_backtest import rolling_predict, backtest  # 复用信号生成

RESULT_DIR = "data/results"
FIGURE_DIR = "data/figures"
os.makedirs(RESULT_DIR, exist_ok=True)


def backtest_long_short(signal_probs, close_prices, long_thresh=0.55, short_thresh=0.45, transaction_cost=0.001,
                        initial_capital=1e6):
    """
    做多+做空策略，仓位与概率成正比
    """
    combined = pd.DataFrame({'signal_prob': signal_probs, 'close': close_prices}).dropna().sort_index()
    combined = combined.astype(float)

    # 计算对数收益率（用于连续复利，简化用普通收益率）
    returns = combined['close'].pct_change().fillna(0)

    # 仓位：做多正，做空负
    probs = combined['signal_prob']
    positions = np.where(probs > long_thresh, probs,
                         np.where(probs < short_thresh, -(1 - probs), 0))
    positions = pd.Series(positions, index=combined.index)
    # 次日生效
    positions = positions.shift(1).fillna(0)

    # 日收益率 = 仓位 * 标的收益率
    daily_returns = positions * returns
    # 交易成本：当仓位绝对值变化时收取
    trade_cost = positions.diff().abs() * transaction_cost
    trade_cost.iloc[0] = 0
    daily_returns_net = daily_returns - trade_cost

    nav = (1 + daily_returns_net).cumprod()
    nav = nav / nav.iloc[0] * initial_capital
    bench_nav = (1 + returns).cumprod()
    bench_nav = bench_nav / bench_nav.iloc[0] * initial_capital

    total_return = nav.iloc[-1] / initial_capital - 1
    bench_return = bench_nav.iloc[-1] / initial_capital - 1
    trading_days = len(nav)
    annual_return = (1 + total_return) ** (252 / trading_days) - 1 if total_return > -1 else np.nan
    bench_annual = (1 + bench_return) ** (252 / trading_days) - 1 if bench_return > -1 else np.nan
    excess_returns = daily_returns_net - 0.03 / 252
    sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else np.nan
    peak = nav.cummax()
    drawdown = (nav - peak) / peak
    max_drawdown = drawdown.min()
    trade_days = daily_returns_net != 0
    win_rate = (daily_returns_net[trade_days] > 0).sum() / trade_days.sum() if trade_days.sum() > 0 else np.nan
    trade_times = positions.diff().abs().sum()

    metrics = {
        '总收益率': f"{total_return:.2%}",
        '基准收益率': f"{bench_return:.2%}",
        '年化收益率': f"{annual_return:.2%}",
        '基准年化': f"{bench_annual:.2%}",
        '夏普比率': f"{sharpe:.2f}",
        '最大回撤': f"{max_drawdown:.2%}",
        '胜率': f"{win_rate:.2%}",
        '交易次数': int(trade_times)
    }
    return nav, bench_nav, metrics


def main():
    print("=" * 60)
    print("步骤20：改进策略（做空 + 动态仓位）")
    print("=" * 60)

    stock = "A_sh.600036_招商银行"
    df = load_data(stock)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)

    features = FEATURE_COLS + ['llm_score']
    close_prices = df['close'].copy()

    print("正在生成交易信号（可能需要10-15分钟）...")
    signal_probs = rolling_predict(df, features, seq_len=20, window=60)
    print(f"信号数量：{len(signal_probs)}")

    print("回测做多+做空策略...")
    nav, bench_nav, metrics = backtest_long_short(signal_probs, close_prices, long_thresh=0.55, short_thresh=0.45)

    print("\n=== 改进策略绩效 ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(nav.index, nav, label='Long-Short Strategy (Dynamic)', linewidth=2)
    plt.plot(bench_nav.index, bench_nav, label='Buy & Hold', linewidth=2, linestyle='--')
    plt.title('Improved Strategy with Short Selling')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (CNY)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(FIGURE_DIR, 'improved_longshort_curve.png'), dpi=300)
    print(f"净值曲线图保存至：{FIGURE_DIR}/improved_longshort_curve.png")

    # 保存绩效
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(RESULT_DIR, "improved_backtest_metrics.csv"), index=False)
    print("绩效指标已保存")


if __name__ == "__main__":
    main()