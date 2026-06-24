#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step51_generate_remaining_stocks.py
补齐剩余 9 只 A 股的预测概率（adaptive_probs_*.csv）
仅处理尚未生成预测文件的 A 股
"""
import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# 导入你的模型和工具
from step47_train_adaptive_patchtst import rolling_predict_adaptive, set_seed
from step38_robust_multi_stock import load_individual_features, load_stock_data

# ===== 配置 =====
RESULT_DIR = "data/results"
os.makedirs(RESULT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ===== 定义所有 15 只 A 股 =====
ALL_A_STOCKS = [
    # --- 已有的 6 只（跳过） ---
    "A_sh.600036_招商银行",
    "A_sz.000858_五粮液",
    "A_sz.000001_平安银行",
    "A_sh.600030_中信证券",
    "A_sh.600519_贵州茅台",
    "A_sz.300750_宁德时代",
    # --- 需要补齐的 9 只（剩余 A 股） ---
    "A_sh.600887_伊利股份",
    "A_sh.601012_隆基绿能",
    "A_sh.601688_华泰证券",
    "A_sz.000333_美的集团",
    "A_sz.000568_泸州老窖",
    "A_sz.000651_格力电器",
    "A_sz.002142_宁波银行",
    "A_sz.002594_比亚迪",
    "A_sz.300059_东方财富",
]

def check_if_exists(stock):
    """检查该股票的预测概率文件是否已存在"""
    path = os.path.join(RESULT_DIR, f"adaptive_probs_{stock}.csv")
    return os.path.exists(path)

def main():
    print("=" * 60)
    print("step51：补齐剩余 9 只 A 股预测")
    print("=" * 60)

    # 筛选出需要处理的股票
    stocks_to_run = []
    for stock in ALL_A_STOCKS:
        if check_if_exists(stock):
            print(f"⏭️  {stock} 已存在，跳过")
        else:
            stocks_to_run.append(stock)
            print(f"🔄 {stock} 需要生成")

    if not stocks_to_run:
        print("✅ 所有 A 股预测文件已齐全！")
        return

    print(f"\n📌 共需处理 {len(stocks_to_run)} 只股票")
    print("=" * 60)

    # 遍历处理
    for idx, stock in enumerate(stocks_to_run, 1):
        print(f"\n[{idx}/{len(stocks_to_run)}] 处理 {stock} ...")

        try:
            # 1. 加载特征和数据
            features = load_individual_features(stock)
            df = load_stock_data(stock, features)
            print(f"  特征数: {len(features)}, 数据长度: {len(df)}")

            # 2. 运行滚动预测（复用 step47 的函数）
            set_seed(42)  # 固定种子确保可复现
            probs, unc = rolling_predict_adaptive(df, features)

            if probs.empty:
                print(f"  ❌ 预测结果为空，跳过 {stock}")
                continue

            # 3. 保存
            result_df = pd.DataFrame({
                'prob': probs,
                'uncertainty': unc
            })
            save_path = os.path.join(RESULT_DIR, f"adaptive_probs_{stock}.csv")
            result_df.to_csv(save_path)
            print(f"  ✅ 保存至 {save_path}")
            print(f"     预测点数: {len(probs)}")

        except FileNotFoundError as e:
            print(f"  ❌ 数据文件不存在: {e}")
            print(f"     请确认 data/features/{stock}.parquet 是否存在")
        except Exception as e:
            print(f"  ❌ 处理 {stock} 时出错: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("✅ step51 完成")
    print("现在你的系统下拉菜单将显示全部 15 只 A 股！")
    print("=" * 60)

if __name__ == "__main__":
    main()