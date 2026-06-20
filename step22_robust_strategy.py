#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤22：修正版稳健策略（趋势过滤+波动率缩放+ATR止损）
包含完整的回测、绩效指标和高级图表
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from step4_causal_attribution import load_data, FEATURE_COLS
from step15_backtest import rolling_predict, train_lstm_quick, train_transformer_quick, DEVICE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import torch

# 配置
RESULT_DIR = "data/results"
FIGURE_DIR = "data/figures"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


# ---------------------------- 辅助函数 ----------------------------
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


def get_predictions_for_backtest(df, features, seq_len=20, window=60):
    """
    生成集成模型的预测概率（加权软投票），并返回与df索引对齐的Series
    同时保存到文件以便复用
    """
    cache_path = os.path.join(RESULT_DIR, "ensemble_probs.csv")
    if os.path.exists(cache_path):
        print("加载缓存的预测概率...")
        probs = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return probs['prob']
    else:
        print("生成预测概率（可能需要15分钟）...")
        probs_list = []
        dates_list = []
        total_points = len(df) - (window + seq_len + 1)
        for idx, i in enumerate(range(window + seq_len, len(df) - 1)):
            train_end = i - seq_len
            train_start = train_end - window
            train_val_df = df.iloc[train_start:train_end + seq_len]
            X_test_seq = df[features].iloc[i - seq_len:i].values.reshape(1, seq_len, -1)

            # 准备LSTM/Transformer数据
            X_all, y_all = [], []
            for j in range(seq_len, len(train_val_df)):
                X_all.append(train_val_df[features].iloc[j - seq_len:j].values)
                y_all.append(train_val_df['label'].iloc[j])
            X_all = np.array(X_all, dtype=np.float32)
            y_all = np.array(y_all, dtype=np.float32)
            if len(X_all) == 0:
                continue
            split = int(0.8 * len(X_all))
            X_train_seq, X_val_seq = X_all[:split], X_all[split:]
            y_train_seq, y_val_seq = y_all[:split], y_all[split:]

            scaler = StandardScaler()
            X_train_flat = X_train_seq.reshape(-1, X_train_seq.shape[-1])
            X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train_seq.shape)
            X_val_scaled = scaler.transform(X_val_seq.reshape(-1, X_val_seq.shape[-1])).reshape(X_val_seq.shape)
            X_test_scaled = scaler.transform(X_test_seq.reshape(-1, X_test_seq.shape[-1])).reshape(X_test_seq.shape)

            # LSTM
            lstm_model = train_lstm_quick(X_train_scaled, y_train_seq, X_val_scaled, y_val_seq,
                                          input_dim=len(features), epochs=30)
            # Transformer
            trans_model = train_transformer_quick(X_train_scaled, y_train_seq, X_val_scaled, y_val_seq,
                                                  input_dim=len(features), epochs=20)
            # RF
            train_df = df.iloc[train_start:train_end]
            X_train_rf = train_df[features].values
            y_train_rf = train_df['label'].values
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train_rf, y_train_rf)
            X_test_rf = df[features].iloc[i].values.reshape(1, -1)
            rf_prob = rf.predict_proba(X_test_rf)[0, 1]

            lstm_model.eval()
            trans_model.eval()
            with torch.no_grad():
                lstm_prob = lstm_model(torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)).item()
                trans_prob = trans_model(torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)).item()

            # 加权软投票权重（基于全局F1）
            f1_rf, f1_lstm, f1_trans = 0.4527, 0.4451, 0.4481
            w_sum = f1_rf + f1_lstm + f1_trans
            weights = np.array([f1_rf, f1_lstm, f1_trans]) / w_sum
            ensemble_prob = weights[0] * rf_prob + weights[1] * lstm_prob + weights[2] * trans_prob

            dates_list.append(df.index[i + 1])
            probs_list.append(ensemble_prob)

            if (idx + 1) % 200 == 0:
                print(f"已处理 {idx + 1}/{total_points} 个测试点")
        probs_series = pd.Series(probs_list, index=dates_list, name='prob')
        # 保存缓存
        probs_series.to_csv(cache_path, header=True)
        print("预测概率已缓存")
        return probs_series


def backtest_strategy(probs, df, initial_capital=1e6,
                      trend_window=20,
                      max_position=0.5,
                      atr_stop_mult=2.0,
                      volatility_target=0.15,
                      transaction_cost=0.001):
    """
    完整策略：
    1. 趋势方向：价格 > MA(trend_window) 做多，否则做空（可选，这里使用绝对方向）
    2. 信号强度：abs(prob - 0.5)*2 （映射到0-1）
    3. 仓位 = 方向 * 信号强度 * 波动率缩放因子
    4. ATR移动止损
    """
    close = df['close']
    # 计算趋势MA
    ma = close.rolling(trend_window).mean()
    # 方向：上涨趋势做多，下跌趋势做空
    trend_dir = np.where(close > ma, 1, -1)

    # 预测方向（基于概率）
    pred_dir = np.where(probs >= 0.5, 1, -1)
    # 最终方向：趋势与预测一致则交易，否则0
    final_dir = np.where(trend_dir == pred_dir, trend_dir, 0)

    # 信号强度
    signal_strength = np.abs(probs - 0.5) * 2  # 0~1
    # 波动率缩放（年化波动率目标调整）
    returns = close.pct_change().fillna(0)
    vol = returns.rolling(20).std() * np.sqrt(252)
    vol_scaler = volatility_target / vol.clip(lower=0.05, upper=0.5)
    vol_scaler = vol_scaler.fillna(1)

    # 原始仓位
    raw_position = final_dir * signal_strength * vol_scaler
    position = np.clip(raw_position, -max_position, max_position)
    # 次日生效
    position = pd.Series(position, index=probs.index).shift(1).fillna(0)

    # ATR
    atr = compute_atr(df, 14)

    # 执行止损逻辑
    final_positions = []
    in_position = False
    entry_price = 0
    current_dir = 0
    stop_price = 0

    for idx in position.index:
        pos = position.loc[idx]
        price = close.loc[idx]
        atr_val = atr.loc[idx]
        if not in_position and pos != 0:
            # 开仓
            in_position = True
            entry_price = price
            current_dir = pos
            stop_price = entry_price - atr_stop_mult * atr_val if pos > 0 else entry_price + atr_stop_mult * atr_val
            final_positions.append(pos)
        elif in_position:
            # 检查止损
            if (current_dir > 0 and price < stop_price) or (current_dir < 0 and price > stop_price):
                in_position = False
                final_positions.append(0)
            else:
                # 移动止损
                if current_dir > 0:
                    stop_price = max(stop_price, price - atr_stop_mult * atr_val)
                else:
                    stop_price = min(stop_price, price + atr_stop_mult * atr_val)
                # 检查信号是否改变方向（如果新信号方向相反，平仓）
                if pos * current_dir < 0:
                    in_position = False
                    final_positions.append(0)
                else:
                    final_positions.append(current_dir)
        else:
            final_positions.append(0)

    final_positions = pd.Series(final_positions, index=position.index)

    # 计算收益
    daily_returns = close.pct_change().fillna(0)
    strategy_returns = final_positions * daily_returns
    trade_costs = final_positions.diff().abs() * transaction_cost
    net_returns = strategy_returns - trade_costs
    nav = (1 + net_returns).cumprod() * initial_capital
    bench_nav = (1 + daily_returns).cumprod() * initial_capital

    # 绩效指标
    total_ret = nav.iloc[-1] / initial_capital - 1
    bench_ret = bench_nav.iloc[-1] / initial_capital - 1
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
    return nav, bench_nav, net_returns, metrics


def plot_monthly_returns(nav):
    """月度收益热图（修复版）"""
    # 计算月度收益率
    monthly_ret = nav.resample('M').last().pct_change().dropna()
    # 构造年份和月份
    years = monthly_ret.index.year
    months = monthly_ret.index.month
    # 创建透视表
    pivot = pd.DataFrame({
        'year': years,
        'month': months,
        'return': monthly_ret.values
    }).pivot(index='year', columns='month', values='return')
    plt.figure(figsize=(12,6))
    sns.heatmap(pivot, annot=True, fmt='.1%', cmap='RdYlGn', center=0)
    plt.title('Monthly Returns Heatmap')
    plt.xlabel('Month')
    plt.ylabel('Year')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'monthly_returns_heatmap.png'), dpi=300)
    plt.close()

def plot_rolling_sharpe(net_returns, window=60):
    """滚动夏普比率"""
    rolling_sharpe = net_returns.rolling(window).apply(
        lambda x: np.sqrt(252) * x.mean() / x.std() if x.std() != 0 else 0)
    plt.figure(figsize=(10, 4))
    plt.plot(rolling_sharpe.index, rolling_sharpe, color='blue')
    plt.axhline(0, linestyle='--', color='gray')
    plt.title(f'Rolling Sharpe Ratio (window={window} days)')
    plt.ylabel('Sharpe')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'rolling_sharpe.png'), dpi=300)
    plt.close()


def plot_return_distribution(net_returns):
    """收益分布直方图"""
    plt.figure(figsize=(8, 5))
    sns.histplot(net_returns[net_returns != 0], bins=50, kde=True)
    plt.title('Daily Return Distribution (Trading Days Only)')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'return_distribution.png'), dpi=300)
    plt.close()


def main():
    print("=" * 60)
    print("步骤22：修正版稳健策略（完整回测与图表）")
    print("=" * 60)

    stock = "A_sh.600036_招商银行"
    df = load_data(stock)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
    # 确保有high/low列
    if 'high' not in df.columns:
        df['high'] = df['close'] * 1.02
    if 'low' not in df.columns:
        df['low'] = df['close'] * 0.98

    features = FEATURE_COLS + ['llm_score']

    # 获取预测概率（带缓存）
    probs = get_predictions_for_backtest(df, features, seq_len=20, window=60)
    print(f"有效预测数量：{len(probs)}")

    # 对齐数据
    common_idx = probs.index.intersection(df.index)
    probs = probs[common_idx]
    df_aligned = df.loc[common_idx]

    # 运行策略
    print("执行策略回测...")
    nav, bench_nav, net_returns, metrics = backtest_strategy(
        probs, df_aligned,
        trend_window=20,
        max_position=0.5,
        atr_stop_mult=2.0,
        volatility_target=0.15,
        transaction_cost=0.001
    )

    print("\n=== 策略绩效 ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # 绘图
    plt.figure(figsize=(12, 5))
    plt.plot(nav.index, nav, label='Robust Strategy')
    plt.plot(bench_nav.index, bench_nav, label='Buy & Hold', linestyle='--')
    plt.title('Equity Curve Comparison')
    plt.ylabel('Portfolio Value (CNY)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(FIGURE_DIR, 'robust_strategy_curve.png'), dpi=300)
    plt.close()

    plot_monthly_returns(nav)
    plot_rolling_sharpe(net_returns)
    plot_return_distribution(net_returns)

    # 保存指标
    pd.DataFrame([metrics]).to_csv(os.path.join(RESULT_DIR, "robust_strategy_metrics.csv"), index=False)
    print("\n所有图表已保存至 data/figures/")
    print("步骤22完成！")


if __name__ == "__main__":
    main()