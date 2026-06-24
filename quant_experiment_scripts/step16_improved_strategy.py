#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤16：策略改进（阈值调优 + 止损）
基于集成模型预测概率，测试不同阈值，并加入5%止损机制。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from step4_causal_attribution import load_data, FEATURE_COLS
from step15_backtest import rolling_predict, backtest  # 复用步骤15的预测生成和回测函数

RESULT_DIR = "data/results"
FIGURE_DIR = "data/figures"
os.makedirs(RESULT_DIR, exist_ok=True)


def backtest_with_stop(signal_probs, close_prices, threshold=0.5, stop_loss=0.05, initial_capital=1e6,
                       transaction_cost=0.001):
    """
    改进版回测：支持阈值调整和止损
    - threshold: 买入信号的概率阈值（>=threshold 买入，否则空仓）
    - stop_loss: 止损线（相对于买入后最高价的回撤比例，例如0.05表示5%止损）
    """
    combined = pd.DataFrame({'signal_prob': signal_probs, 'close': close_prices}).dropna().sort_index()
    if combined.empty:
        raise ValueError("无有效数据")
    combined = combined.astype(float)

    # 原始信号
    raw_signals = (combined['signal_prob'] >= threshold).astype(int)
    # 引入止损：持仓时如果价格回撤超过 stop_loss，则强制平仓（信号变为0）
    positions = pd.Series(0, index=combined.index)
    in_position = False
    entry_price = 0.0
    peak_price = 0.0

    for i, idx in enumerate(combined.index):
        if i == 0:
            continue
        # 当前价格
        curr_price = combined.loc[idx, 'close']

        # 根据原始信号决定是否要开仓/平仓，但需考虑止损
        if raw_signals.loc[idx] == 1 and not in_position:
            # 开仓
            in_position = True
            entry_price = curr_price
            peak_price = curr_price
            positions.loc[idx] = 1
        elif in_position:
            # 更新最高价
            if curr_price > peak_price:
                peak_price = curr_price
            # 计算回撤
            drawdown = (peak_price - curr_price) / peak_price
            if drawdown >= stop_loss:
                # 触发止损，平仓
                in_position = False
                positions.loc[idx] = 0
            else:
                # 继续持有，但检查原始信号是否要求平仓
                if raw_signals.loc[idx] == 0:
                    in_position = False
                    positions.loc[idx] = 0
                else:
                    positions.loc[idx] = 1
        else:
            positions.loc[idx] = 0

    # 持仓次日生效（避免未来信息）
    positions = positions.shift(1).fillna(0)

    # 计算收益率
    returns = combined['close'].pct_change()
    returns.iloc[0] = 0.0
    returns = returns.fillna(0.0)

    daily_returns = positions * returns
    trade_cost = positions.diff().abs() * transaction_cost
    daily_returns_net = daily_returns - trade_cost
    daily_returns_net = daily_returns_net.fillna(0.0)

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
        '阈值': threshold,
        '止损': stop_loss,
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
    print("步骤16：策略改进（阈值调优 + 止损）")
    print("=" * 60)

    stock = "A_sh.600036_招商银行"
    df = load_data(stock)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)

    features = FEATURE_COLS + ['llm_score']
    close_prices = df['close'].copy()

    # 生成交易信号（复用步骤15的滚动预测，但会重新训练模型，耗时较长）
    print("正在生成交易信号（这可能需要10-15分钟）...")
    signal_probs = rolling_predict(df, features, seq_len=20, window=60)
    print(f"信号数量：{len(signal_probs)}")

    # 测试不同阈值和止损组合
    thresholds = [0.5, 0.55, 0.6, 0.65]
    stop_losses = [0.05, 0.07, 0.1]
    results = []

    best_sharpe = -np.inf
    best_metrics = None
    best_nav = None
    best_params = None

    for th in thresholds:
        for sl in stop_losses:
            print(f"\n测试阈值={th}, 止损={sl}")
            try:
                nav, bench_nav, metrics = backtest_with_stop(signal_probs, close_prices, threshold=th, stop_loss=sl)
                results.append(metrics)
                sharpe = float(metrics['夏普比率'])
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_metrics = metrics
                    best_nav = nav
                    best_params = (th, sl)
            except Exception as e:
                print(f"回测失败：{e}")

    # 输出最优结果
    print("\n=== 最优策略参数 ===")
    for k, v in best_metrics.items():
        print(f"{k}: {v}")

    # 绘制最优策略的净值曲线对比
    plt.figure(figsize=(12, 6))
    plt.plot(best_nav.index, best_nav, label=f'Improved Strategy (th={best_params[0]}, stop={best_params[1]})',
             linewidth=2)
    bench_nav = (1 + close_prices.pct_change().fillna(0)).cumprod()
    bench_nav = bench_nav / bench_nav.iloc[0] * 1e6
    plt.plot(bench_nav.index, bench_nav, label='Buy & Hold', linewidth=2, linestyle='--')
    plt.title('Improved Strategy vs Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (CNY)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'improved_backtest_curve.png'), dpi=300)
    print(f"\n净值曲线图已保存：{FIGURE_DIR}/improved_backtest_curve.png")

    # 保存所有测试结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULT_DIR, "threshold_stop_tuning.csv"), index=False)
    print(f"调参结果已保存：{RESULT_DIR}/threshold_stop_tuning.csv")

    # 绘制不同阈值的夏普比率热力图
    pivot = results_df.pivot(index='阈值', columns='止损', values='夏普比率')
    plt.figure(figsize=(8, 6))
    plt.imshow(pivot.values, cmap='RdYlGn', aspect='auto')
    plt.colorbar(label='Sharpe Ratio')
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel('Stop Loss')
    plt.ylabel('Threshold')
    plt.title('Sharpe Ratio Heatmap')
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            plt.text(j, i, f"{pivot.values[i, j]:.2f}", ha='center', va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'sharpe_heatmap.png'), dpi=300)
    print(f"夏普热力图已保存：{FIGURE_DIR}/sharpe_heatmap.png")

    print("\n步骤16完成！")


if __name__ == "__main__":
    main()