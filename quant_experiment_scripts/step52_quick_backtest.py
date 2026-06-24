#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step52_quick_backtest.py
为新增的 9 只 A 股快速生成回测结果（单次，不重复）
仅生成 adaptive_repeat_*.csv，供汇总使用
"""
import pandas as pd
from step38_robust_multi_stock import load_individual_features, load_stock_data, robust_backtest

NEW_STOCKS = [
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

for stock in NEW_STOCKS:
    print(f"处理 {stock}...")
    try:
        # 加载预测概率（文件已由 step51 生成）
        probs = pd.read_csv(
            f"data/results/adaptive_probs_{stock}.csv",
            index_col=0,
            parse_dates=True
        )['prob']

        # 加载特征和数据
        features = load_individual_features(stock)
        df = load_stock_data(stock, features)

        # 回测（参数与 step49 完全一致）
        metrics = robust_backtest(
            probs, df,
            long_thresh=0.55,
            short_thresh=0.45,
            vol_target=0.15,
            stop_loss=0.01,
            transaction_cost=0.001
        )

        # 保存为 repeat 文件（单次结果）
        pd.DataFrame([metrics]).to_csv(
            f"data/results/adaptive_repeat_{stock}.csv",
            index=False
        )
        print(f"  ✅ 已保存 adaptive_repeat_{stock}.csv")

    except Exception as e:
        print(f"  ❌ 错误: {e}")
        import traceback

        traceback.print_exc()

print("\n✅ step52 完成！")
print("现在请执行汇总重建命令，将新增的 9 只股票合并到汇总文件中。")