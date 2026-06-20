#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step48_multi_stock_adaptive.py
多股票自适应多专家 PatchTST 回测
"""
import os
import pandas as pd
from step38_robust_multi_stock import load_individual_features, load_stock_data, robust_backtest
from step47_train_adaptive_patchtst import rolling_predict_adaptive, set_seed
import warnings

warnings.filterwarnings('ignore')

RESULT_DIR = "data/results"
os.makedirs(RESULT_DIR, exist_ok=True)

# 股票列表（请根据实际存在的文件修改）
STOCKS = [
    "A_sh.600036_招商银行",
    "A_sz.000858_五粮液",
    "A_sz.000001_平安银行",
    "A_sh.600030_中信证券",
    "A_sh.600519_贵州茅台",
    "A_sz.300750_宁德时代"
]


def process_stock(stock):
    print(f"\n{'=' * 60}")
    print(f"处理股票: {stock}")
    print('=' * 60)
    try:
        features = load_individual_features(stock)
        if features is None or len(features) == 0:
            print(f"跳过 {stock}: 无特征")
            return None
        print(f"特征数量: {len(features)}")
        df = load_stock_data(stock, features)
        if df is None or len(df) == 0:
            print(f"跳过 {stock}: 无数据")
            return None

        # 滚动预测
        cache_file = os.path.join(RESULT_DIR, f"adaptive_probs_{stock}.csv")
        if os.path.exists(cache_file):
            probs = pd.read_csv(cache_file, index_col=0, parse_dates=True)['prob']
            print("从缓存加载预测概率")
        else:
            print("开始滚动预测（自适应模型）...")
            probs, unc = rolling_predict_adaptive(df, features)
            result_df = pd.DataFrame({'prob': probs, 'uncertainty': unc})
            result_df.to_csv(cache_file)
            print(f"保存至 {cache_file}")

        # 回测
        metrics = robust_backtest(probs, df,
                                  long_thresh=0.55, short_thresh=0.45,
                                  vol_target=0.15, stop_loss=0.01,
                                  transaction_cost=0.001)
        metrics['stock'] = stock
        return metrics
    except Exception as e:
        print(f"处理 {stock} 时出错: {e}")
        return None


def main():
    set_seed(42)
    all_metrics = []
    for stock in STOCKS:
        m = process_stock(stock)
        if m:
            all_metrics.append(m)
    if not all_metrics:
        print("没有成功处理任何股票")
        return
    results_df = pd.DataFrame(all_metrics)
    # 重排列
    results_df = results_df[['stock', '总收益率', '年化收益率', '夏普比率', '最大回撤', '胜率', '交易次数']]
    print("\n" + "=" * 60)
    print("多股票自适应策略绩效汇总")
    print("=" * 60)
    print(results_df.to_string(index=False))
    # 保存
    results_df.to_csv(os.path.join(RESULT_DIR, "multi_stock_adaptive_results.csv"), index=False)
    print(f"\n结果已保存至 {RESULT_DIR}/multi_stock_adaptive_results.csv")


if __name__ == "__main__":
    main()