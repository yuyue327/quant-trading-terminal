#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step27_feature_selection.py
使用互信息 + 随机森林特征重要性选择与未来收益率最相关的特征子集
支持在每个滚动窗口内独立筛选，避免未来信息泄露
"""
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from step4_causal_attribution import load_data, FEATURE_COLS

# 配置
DATA_DIR = "data/features"
OUTPUT_DIR = "data/feature_selection"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 特征列（不含 label 和 llm_score，后面会单独加 llm_score 测试）
BASE_FEATURES = FEATURE_COLS.copy()  # 23个技术指标
TOP_K = 15  # 每只股票选择 top k 特征（包含 llm_score 则选 k-1 个技术指标）


def select_features_for_window(df_window, candidate_features, top_k=TOP_K):
    """
    在单个训练窗口内进行特征选择
    df_window: 训练窗口的DataFrame（已经过时间切片）
    返回: selected_feature_list (list)
    """
    X = df_window[candidate_features].dropna().values
    y = df_window['label'].dropna().values
    if len(X) == 0 or len(y) == 0:
        return candidate_features[:top_k]  # 若窗口数据不足，返回默认特征

    # 对齐长度
    min_len = min(len(X), len(y))
    X = X[:min_len]
    y = y[:min_len]

    # 1. 互信息
    mi = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi, index=candidate_features)

    # 2. 随机森林重要性
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    imp = rf.feature_importances_
    imp_series = pd.Series(imp, index=candidate_features)

    # 综合得分：归一化后取平均
    mi_norm = (mi_series - mi_series.min()) / (mi_series.max() - mi_series.min() + 1e-9)
    imp_norm = (imp_series - imp_series.min()) / (imp_series.max() - imp_series.min() + 1e-9)
    combined = (mi_norm + imp_norm) / 2
    combined_sorted = combined.sort_values(ascending=False)

    selected = combined_sorted.head(top_k).index.tolist()
    if 'llm_score' not in selected:
        selected[-1] = 'llm_score'

    return selected


def rolling_feature_selection_for_stock(stock_name, window=360, stride=20):
    """
    对一只股票在整个时间序列上，每个滚动窗口内独立执行特征选择
    返回: dict {test_idx: selected_features}
    """
    df = load_data(stock_name)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)

    if 'llm_score' not in df.columns:
        df['llm_score'] = 0.5
        print(f"  警告: {stock_name} 缺少 llm_score，使用默认 0.5")

    candidate_features = BASE_FEATURES + ['llm_score']
    window_features = {}
    total_windows = (len(df) - window) // stride

    for i in range(total_windows):
        test_idx = window + i * stride
        if test_idx >= len(df):
            break
        train_df = df.iloc[test_idx - window:test_idx]
        selected = select_features_for_window(train_df, candidate_features, TOP_K)
        window_features[test_idx] = selected

    return window_features


def select_features_for_stock(stock_name):
    """
    原有函数：在全量数据上一次性筛选（保留用于兼容旧代码，但论文描述中将不再使用此结果）
    """
    df = load_data(stock_name)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)

    if 'llm_score' not in df.columns:
        df['llm_score'] = 0.5
        print(f"  警告: {stock_name} 缺少 llm_score，使用默认 0.5")

    candidate_features = BASE_FEATURES + ['llm_score']
    selected, _ = select_features_for_window(df, candidate_features, TOP_K)
    return selected, None  # 返回选中的特征列表（忽略重要性）


def main():
    print("=" * 60)
    print("步骤27：特征选择（互信息 + 随机森林重要性）")
    print("=" * 60)
    print("⚠️  注意：为保证无未来信息泄露，实际实验中使用滚动窗口独立筛选。")
    print("    本脚本只生成静态特征列表作为后续步骤的默认配置，并保留滚动筛选函数。")
    print("    实际特征选择在训练窗口内动态进行，详见 rolling_feature_selection_for_stock()。")

    # 获取所有股票名称
    stock_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.parquet') and f != 'all_stocks_features.parquet']
    stock_names = [f.replace('.parquet', '') for f in stock_files]

    if not stock_names:
        stock_names = [
            'A_sh.600036_招商银行', 'A_sz.000001_平安银行', 'A_sz.002142_宁波银行',
            'A_sh.600030_中信证券', 'A_sh.601688_华泰证券', 'A_sz.300059_东方财富',
            'A_sh.600519_贵州茅台', 'A_sz.000858_五粮液', 'A_sz.000568_泸州老窖',
            'A_sz.000333_美的集团', 'A_sz.000651_格力电器', 'A_sh.600887_伊利股份',
            'A_sz.300750_宁德时代', 'A_sz.002594_比亚迪', 'A_sh.601012_隆基绿能',
            'US_AAPL_AAPL', 'US_MSFT_MSFT', 'US_NVDA_NVDA'
        ]

    print(f"对 {len(stock_names)} 只股票生成静态特征列表（仅用于代码兼容，实际实验采用滚动筛选）")
    all_selected = {}
    importance_dict = {}
    for name in tqdm(stock_names, desc="生成静态特征"):
        try:
            selected, imp = select_features_for_stock(name)
            all_selected[name] = selected
            # 重要：这里只保存静态列表，不覆盖滚动筛选的用途
        except Exception as e:
            print(f"  处理 {name} 失败: {e}")

    # 保存静态特征列表（仅供后向兼容）
    with open(os.path.join(OUTPUT_DIR, "selected_features_per_stock.json"), "w") as f:
        json.dump(all_selected, f, indent=2)

    # 同时保存一个示例滚动筛选结果（用于论文描述佐证）
    demo_stock = 'A_sh.600036_招商银行'
    if demo_stock in stock_names:
        print(f"\n生成 {demo_stock} 的滚动窗口特征选择示例（仅演示，不实际运行全部）...")
        window_features = rolling_feature_selection_for_stock(demo_stock, window=360, stride=20)
        # 只保存前3个窗口作为样例
        demo = {k: window_features[k] for k in list(window_features.keys())[:3]}
        with open(os.path.join(OUTPUT_DIR, "rolling_selection_demo.json"), "w") as f:
            json.dump(demo, f, indent=2)
        print(f"  已保存滚动筛选示例至 rolling_selection_demo.json")

    print("\n✅ 步骤27完成。")
    print("请确保在模型训练脚本（step47）中调用 rolling_feature_selection_for_stock() 进行动态特征选择，")
    print("以避免未来信息泄露。当前代码仅提供静态特征列表作为默认配置。")

if __name__ == "__main__":
    main()