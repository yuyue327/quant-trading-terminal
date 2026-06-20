#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载美股真实数据 (AAPL, MSFT, NVDA) 到指定文件夹
使用 yfinance，无需 VPN（国内网络可能较慢，建议开 VPN 但非必须）
"""

import os
import time
import yfinance as yf
import pandas as pd

# ========== 设置保存路径 ==========
# 你的项目根目录
PROJECT_DIR = "/Users/mac/Desktop/ACML"
RAW_DATA_DIR = os.path.join(PROJECT_DIR, "data", "raw")

# 确保目录存在
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# 股票列表
TICKERS = ["AAPL", "MSFT", "NVDA"]
START_DATE = "2021-01-01"
END_DATE = "2026-06-01"

def download_with_retry(ticker, max_retries=3):
    """带重试机制的下载函数"""
    for attempt in range(max_retries):
        try:
            print(f"正在下载 {ticker} ... (尝试 {attempt+1}/{max_retries})")
            # 下载数据，增加超时设置
            data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, timeout=60)
            if data.empty:
                print(f"  {ticker} 无数据，重试中...")
                time.sleep(5)
                continue
            # 重置索引
            data = data.reset_index()
            # 保存文件：命名格式 US_{股票代码}_{股票代码}.csv
            filename = os.path.join(RAW_DATA_DIR, f"US_{ticker}_{ticker}.csv")
            data.to_csv(filename, index=False)
            print(f"  ✓ {ticker} 成功保存到 {filename}")
            return True
        except Exception as e:
            print(f"  ✗ {ticker} 下载失败: {e}")
            time.sleep(5)
    return False

def main():
    print("="*60)
    print("下载美股真实数据 (yfinance)")
    print(f"保存目录: {RAW_DATA_DIR}")
    print("="*60)
    success_count = 0
    for ticker in TICKERS:
        if download_with_retry(ticker):
            success_count += 1
        # 避免请求过快
        time.sleep(2)
    print(f"\n下载完成: 成功 {success_count}/{len(TICKERS)} 支股票")
    print("数据保存在:", RAW_DATA_DIR)

if __name__ == "__main__":
    main()