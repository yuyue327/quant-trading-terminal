#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step30_backtest_with_uncertainty.py
基于集成预测（概率 + 不确定性）的动态仓位回测
包含趋势过滤、波动率缩放、ATR止损
兼容没有 uncertainty 列的情况
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from step4_causal_attribution import load_data, FEATURE_COLS

# 配置
RESULT_DIR = "data/results"
FIGURE_DIR = "data/figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def compute_atr(df, window=14):
    """计算ATR（平均真实波幅）"""
    high = df['high'] if 'high' in df.columns else df['close'] * 1.02
    low = df['low'] if 'low' in df.columns else df['close'] * 0.98
    close = df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr.fillna(method='bfill').fillna(0)


def backtest_with_uncertainty(probs_df, df,
                              trend_window=20,
                              max_position=0.5,
                              atr_stop_mult=1.5,
                              volatility_target=0.18,
                              uncertainty_threshold=0.2,
                              transaction_cost=0.001):
    """
    不确定性感知回测
    - probs_df: DataFrame with columns 'prob' (概率) and optionally 'uncertainty' (标准差)
    - df: 原始数据（需包含 close, high, low, label 等）
    """
    # 对齐数据
    common_idx = probs_df.index.intersection(df.index)
    probs = probs_df.loc[common_idx, 'prob']

    # 处理 uncertainty 列：若不存在则基于滚动窗口计算概率的标准差作为 uncertainty
    if 'uncertainty' not in probs_df.columns:
        print("警告: 未找到 uncertainty 列，将使用滚动窗口标准差作为不确定性估计")
        uncertainty = probs.rolling(20, min_periods=5).std().fillna(0.1)
        uncertainty = np.clip(uncertainty, 0.05, 0.3)
    else:
        uncertainty = probs_df.loc[common_idx, 'uncertainty']

    df_aligned = df.loc[common_idx]
    close = df_aligned['close']

    # 趋势方向
    ma = close.rolling(trend_window).mean()
    trend_dir = np.where(close > ma, 1, 0)  # 只做多，不做空

    # 信号强度：概率偏离0.5的程度映射到0-1
    signal_strength = np.abs(probs - 0.5) * 2  # [0,1]

    # 不确定性惩罚：当不确定性 > threshold 时，降低仓位因子
    uncertainty_penalty = 1 / (1 + uncertainty / uncertainty_threshold)
    uncertainty_penalty = np.clip(uncertainty_penalty, 0.3, 1.0)

    # 波动率缩放（年化波动率目标调整）
    returns = close.pct_change().fillna(0)
    vol = returns.rolling(20).std() * np.sqrt(252)
    vol_scaler = volatility_target / vol.clip(lower=0.05, upper=0.5)
    vol_scaler = vol_scaler.fillna(1)

    # 原始仓位 = 方向 * 信号强度 * 波动率缩放 * 不确定性惩罚
    raw_position = trend_dir * signal_strength * vol_scaler * uncertainty_penalty
    position = np.clip(raw_position, 0, max_position)  # 只做多，限制最大仓位

    # 次日生效
    position = pd.Series(position, index=probs.index).shift(1).fillna(0)

    # ATR 止损
    atr = compute_atr(df_aligned, 14)

    # 执行止损逻辑
    final_positions = []
    in_position = False
    entry_price = 0
    stop_price = 0
    for idx in position.index:
        pos = position.loc[idx]
        price = close.loc[idx]
        atr_val = atr.loc[idx]
        if not in_position and pos > 0:
            in_position = True
            entry_price = price
            stop_price = entry_price - atr_stop_mult * atr_val
            final_positions.append(pos)
        elif in_position:
            # 检查止损
            if price < stop_price:
                in_position = False
                final_positions.append(0)
            else:
                # 移动止损（向上移动）
                new_stop = price - atr_stop_mult * atr_val
                if new_stop > stop_price:
                    stop_price = new_stop
                # 若信号变为0则平仓
                if pos == 0:
                    in_position = False
                    final_positions.append(0)
                else:
                    final_positions.append(pos)
        else:
            final_positions.append(0)

    final_positions = pd.Series(final_positions, index=position.index)

    # 计算收益
    daily_returns = close.pct_change().fillna(0)
    strategy_returns = final_positions * daily_returns
    trade_costs = final_positions.diff().abs() * transaction_cost
    net_returns = strategy_returns - trade_costs
    nav = (1 + net_returns).cumprod() * 1e6
    bench_nav = (1 + daily_returns).cumprod() * 1e6

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
    trade_count = (final_positions.diff().abs() > 0).sum()

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

    return nav, bench_nav, net_returns, metrics, final_positions


def plot_results(nav, bench_nav, net_returns, metrics):
    """生成论文质量图表（修复月度热力图错误）"""
    # 1. 净值曲线
    plt.figure(figsize=(12, 6))
    plt.plot(nav.index, nav, label='策略净值 (不确定性感知)', linewidth=2)
    plt.plot(bench_nav.index, bench_nav, label='买入持有', linewidth=1.5, linestyle='--')
    plt.title('策略 vs 基准净值曲线')
    plt.xlabel('日期')
    plt.ylabel('净值 (元)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'strategy_nav_uncertainty.png'), dpi=300)
    plt.close()

    # 2. 回撤曲线
    drawdown = (nav / nav.cummax() - 1)
    plt.figure(figsize=(12, 4))
    plt.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
    plt.plot(drawdown.index, drawdown, color='red', linewidth=1)
    plt.title('策略回撤曲线')
    plt.xlabel('日期')
    plt.ylabel('回撤')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'drawdown_uncertainty.png'), dpi=300)
    plt.close()

    # 3. 滚动夏普比率 (60天)
    rolling_sharpe = net_returns.rolling(60).apply(
        lambda x: np.sqrt(252) * x.mean() / x.std() if x.std() != 0 else 0)
    plt.figure(figsize=(12, 4))
    plt.plot(rolling_sharpe.index, rolling_sharpe, color='green')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title('滚动夏普比率 (60天窗口)')
    plt.xlabel('日期')
    plt.ylabel('夏普比率')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'rolling_sharpe_uncertainty.png'), dpi=300)
    plt.close()

    # 4. 月度收益热力图（修正版）
    # 计算月度收益率（基于净值）
    monthly_ret = nav.resample('M').last().pct_change().dropna()
    if len(monthly_ret) > 0:
        # 构建年份和月份
        years = monthly_ret.index.year
        months = monthly_ret.index.month
        # 创建透视表
        pivot_data = pd.DataFrame({'year': years, 'month': months, 'return': monthly_ret.values})
        pivot_table = pivot_data.pivot(index='year', columns='month', values='return')
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot_table, annot=True, fmt='.1%', cmap='RdYlGn', center=0)
        plt.title('月度收益率热力图')
        plt.xlabel('月份')
        plt.ylabel('年份')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, 'monthly_returns_heatmap_uncertainty.png'), dpi=300)
        plt.close()
    else:
        print("警告：月度收益率数据不足，跳过热力图")

    # 5. 收益分布直方图
    plt.figure(figsize=(10, 5))
    trading_returns = net_returns[net_returns != 0]
    if len(trading_returns) > 0:
        sns.histplot(trading_returns, bins=50, kde=True, color='blue')
        plt.title('策略日收益率分布 (仅交易日)')
        plt.xlabel('收益率')
        plt.ylabel('频次')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, 'return_distribution_uncertainty.png'), dpi=300)
        plt.close()


def main():
    print("=" * 60)
    print("step30：不确定性感知动态仓位回测")
    print("=" * 60)

    # 加载预测概率和不确定性
    cache_path = os.path.join(RESULT_DIR, "ensemble_probs.csv")
    if not os.path.exists(cache_path):
        print(f"错误：未找到 {cache_path}，请先运行 step29_ensemble_predict.py")
        return
    probs_df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    print(f"预测数据形状: {probs_df.shape}, 列: {probs_df.columns.tolist()}")

    # 加载股票数据
    stock = "A_sh.600036_招商银行"
    df = load_data(stock)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
    # 确保有 high/low 列
    if 'high' not in df.columns:
        df['high'] = df['close'] * 1.02
    if 'low' not in df.columns:
        df['low'] = df['close'] * 0.98

    # 回测参数
    nav, bench_nav, net_returns, metrics, positions = backtest_with_uncertainty(
        probs_df, df,
        trend_window=20,
        max_position=0.5,
        atr_stop_mult=1.5,
        volatility_target=0.18,
        uncertainty_threshold=0.2,
        transaction_cost=0.001
    )

    print("\n=== 策略绩效指标 ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # 保存指标
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(RESULT_DIR, "strategy_metrics_uncertainty.csv"), index=False)

    # 绘图
    plot_results(nav, bench_nav, net_returns, metrics)

    # 绘制仓位变化
    plt.figure(figsize=(12, 4))
    plt.plot(positions.index, positions, alpha=0.7, color='purple')
    plt.title('策略仓位变化 (基于不确定性调整)')
    plt.xlabel('日期')
    plt.ylabel('仓位')
    plt.ylim(0, 0.6)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'position_sizing.png'), dpi=300)
    plt.close()

    print(f"\n所有图表已保存至 {FIGURE_DIR}")
    print("step30 完成。")


if __name__ == "__main__":
    main()