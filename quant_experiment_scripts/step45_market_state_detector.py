#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step45_market_state_detector.py
市场状态判别器：基于波动率 + 趋势强度将市场划分为4种状态
可独立运行，输出状态分布直方图
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def detect_market_state(df, window=20, vol_quantile=0.7, trend_threshold=0.0):
    """
    参数:
        df: 必须包含 'close', 'pct_change' 列（最好也有 'volume'）
        window: 滚动窗口
        vol_quantile: 波动率阈值（高于此值为高波动）
        trend_threshold: 趋势强度阈值（高于此值为上涨）
    返回:
        state_series: 0=低波上涨, 1=高波上涨, 2=低波下跌, 3=高波下跌
    """
    # 计算波动率
    volatility = df['pct_change'].rolling(window).std()
    # 趋势强度：短期均线(5) - 长期均线(20)
    ma_short = df['close'].rolling(5).mean()
    ma_long = df['close'].rolling(window).mean()
    trend_strength = (ma_short - ma_long) / ma_long
    # 计算波动率阈值
    vol_threshold = volatility.quantile(vol_quantile)

    state = np.zeros(len(df), dtype=int)
    for i in range(window, len(df)):
        vol_flag = 1 if volatility.iloc[i] >= vol_threshold else 0
        trend_flag = 1 if trend_strength.iloc[i] >= trend_threshold else 0
        state[i] = (vol_flag << 1) | trend_flag  # 编码 0-3
    return pd.Series(state, index=df.index, name='market_state')


def plot_state_distribution(state_series, title="Market State Distribution"):
    plt.figure(figsize=(8, 5))
    state_series.value_counts().sort_index().plot(kind='bar')
    plt.xlabel("State (0=LowVol Bull, 1=HighVol Bull, 2=LowVol Bear, 3=HighVol Bear)")
    plt.ylabel("Days")
    plt.title(title)
    plt.tight_layout()
    os.makedirs("data/figures", exist_ok=True)
    plt.savefig("data/figures/market_state_distribution.png")
    plt.show()


if __name__ == "__main__":
    # 测试：招商银行
    stock_file = "data/features/A_sh.600036_招商银行.parquet"
    if os.path.exists(stock_file):
        df = pd.read_parquet(stock_file)
        state = detect_market_state(df)
        print("状态分布：")
        print(state.value_counts().sort_index())
        plot_state_distribution(state)
    else:
        print(f"文件不存在: {stock_file}")