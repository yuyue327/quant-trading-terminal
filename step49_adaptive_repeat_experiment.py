#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step49_adaptive_repeat_experiment.py
自适应 PatchTST 多股票（含美股）重复实验，输出稳健绩效统计
"""
import os
import sys
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# 导入必要模块（请确保这些文件存在且函数可调用）
from step38_robust_multi_stock import load_individual_features, load_stock_data, robust_backtest
from step47_train_adaptive_patchtst import rolling_predict_adaptive, set_seed

# =========================== 配置 ===========================
RESULT_DIR = "data/results"
os.makedirs(RESULT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# 股票列表（A股 + 美股）
STOCKS = [
    # A股
    "A_sh.600036_招商银行",
    "A_sz.000858_五粮液",
    "A_sz.000001_平安银行",
    "A_sh.600030_中信证券",
    "A_sh.600519_贵州茅台",
    "A_sz.300750_宁德时代",
    # 美股
    "US_AAPL_AAPL",
    "US_MSFT_MSFT",
    "US_NVDA_NVDA",
]

N_REPEATS = 5  # 每只股票重复次数（论文推荐至少5次，可改为3以节省时间）
LONG_THRESH = 0.55
SHORT_THRESH = 0.45
VOL_TARGET = 0.15
STOP_LOSS = 0.01
TRANSACTION_COST = 0.001


# =========================== 辅助函数 ===========================
def parse_metric_value(value):
    """将 robust_backtest 返回的百分比字符串或数字转为浮点数"""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        if value.endswith('%'):
            return float(value.rstrip('%')) / 100.0
        else:
            return float(value)
    return np.nan


def run_single_stock_repeats(stock, n_repeats):
    """对一只股票进行多次独立重复实验"""
    print(f"\n>>> 处理股票: {stock}")
    # 加载特征和原始数据
    try:
        features = load_individual_features(stock)
        df = load_stock_data(stock, features)
    except Exception as e:
        print(f"    错误：无法加载股票 {stock} 的数据或特征，跳过。错误: {e}")
        return None

    print(f"    特征数量: {len(features)}")
    print(f"    数据长度: {len(df)}")

    all_metrics = []
    for rep in range(n_repeats):
        seed = 42 + rep
        set_seed(seed)
        print(f"      重复实验 {rep + 1}/{n_repeats} (seed={seed})", flush=True)

        # 滚动预测（每次独立训练）
        try:
            probs, unc = rolling_predict_adaptive(df, features)
        except Exception as e:
            print(f"        预测失败: {e}")
            continue

        if probs.empty:
            print(f"        预测结果为空，跳过")
            continue

        # 回测
        metrics = robust_backtest(probs, df,
                                  long_thresh=LONG_THRESH,
                                  short_thresh=SHORT_THRESH,
                                  vol_target=VOL_TARGET,
                                  stop_loss=STOP_LOSS,
                                  transaction_cost=TRANSACTION_COST)

        # 提取并转换指标
        record = {
            'seed': seed,
            'sharpe_ratio': parse_metric_value(metrics.get('夏普比率', np.nan)),
            'annual_return': parse_metric_value(metrics.get('年化收益率', np.nan)),
            'max_drawdown': parse_metric_value(metrics.get('最大回撤', np.nan)),
            'win_rate': parse_metric_value(metrics.get('胜率', np.nan)),
            'total_trades': parse_metric_value(metrics.get('交易次数', np.nan))
        }
        all_metrics.append(record)

    if len(all_metrics) == 0:
        print(f"    警告：股票 {stock} 没有任何成功实验，返回空")
        return None

    # 汇总统计
    df_metrics = pd.DataFrame(all_metrics)
    summary = {
        'stock': stock,
        'n_success': len(all_metrics),
        'sharpe_median': df_metrics['sharpe_ratio'].median(),
        'sharpe_std': df_metrics['sharpe_ratio'].std(),
        'sharpe_min': df_metrics['sharpe_ratio'].min(),
        'sharpe_max': df_metrics['sharpe_ratio'].max(),
        'annual_return_median': df_metrics['annual_return'].median(),
        'annual_return_std': df_metrics['annual_return'].std(),
        'max_drawdown_median': df_metrics['max_drawdown'].median(),
        'max_drawdown_std': df_metrics['max_drawdown'].std(),
        'win_rate_median': df_metrics['win_rate'].median(),
        'total_trades_median': df_metrics['total_trades'].median(),
    }
    # 保存详细结果（每只股票一个文件）
    detail_path = os.path.join(RESULT_DIR, f"adaptive_repeat_{stock}.csv")
    df_metrics.to_csv(detail_path, index=False)
    print(f"      保存详细结果至 {detail_path}")
    return summary


def main():
    print("=" * 60)
    print(f"自适应 PatchTST 多股票重复实验（每只股票 {N_REPEATS} 次）")
    print("=" * 60)

    all_summaries = []
    for stock in STOCKS:
        summary = run_single_stock_repeats(stock, N_REPEATS)
        if summary is not None:
            all_summaries.append(summary)

    if not all_summaries:
        print("没有任何有效结果，退出。")
        return

    results_df = pd.DataFrame(all_summaries)
    results_df.to_csv(os.path.join(RESULT_DIR, "adaptive_repeat_summary.csv"), index=False)

    print("\n" + "=" * 60)
    print("汇总结果（中位数及标准差）:")
    print(results_df[['stock', 'sharpe_median', 'sharpe_std', 'annual_return_median', 'max_drawdown_median']])
    print("=" * 60)

    # 额外生成一个可读性好的表格（用于论文）
    results_df[['stock', 'sharpe_median', 'sharpe_std', 'annual_return_median', 'max_drawdown_median']].to_csv(
        os.path.join(RESULT_DIR, "adaptive_paper_table.csv"), index=False)
    print("论文表格已保存至 adaptive_paper_table.csv")


if __name__ == "__main__":
    main()