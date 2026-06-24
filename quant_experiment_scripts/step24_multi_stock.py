#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤24：多股票验证
在招商银行之外的其它股票上测试RF模型（或集成模型）的表现
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from step4_causal_attribution import load_data, FEATURE_COLS

RESULT_DIR = "data/results"
os.makedirs(RESULT_DIR, exist_ok=True)

# 选择其他5只代表性股票（包含银行、券商、消费、科技）
OTHER_STOCKS = [
    "A_sh.600036_招商银行",       # 原股票，用于对比
    "A_sz.000001_平安银行",
    "A_sh.600030_中信证券",
    "A_sh.600519_贵州茅台",
    "A_sz.000858_五粮液",
    "A_sz.300750_宁德时代"
]

def evaluate_single_stock(stock, features, window=60, seq_len=20):
    """滚动评估RF，返回F1和AUC"""
    try:
        df = load_data(stock)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
        # 确保特征都存在
        available_features = [f for f in features if f in df.columns]
        if 'llm_score' not in df.columns:
            # 如果没有llm_score，则创建模拟列（实际上你的数据应该都有）
            df['llm_score'] = 0.5
            available_features = features
        preds = []
        probs = []
        true = []
        for i in range(window + seq_len, len(df) - 1):
            train_end = i - seq_len
            train_start = train_end - window
            train_df = df.iloc[train_start:train_end]
            X_train = train_df[available_features].values
            y_train = train_df['label'].values
            X_test = df[available_features].iloc[i].values.reshape(1, -1)
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            prob = rf.predict_proba(X_test)[0, 1]
            probs.append(prob)
            preds.append(1 if prob >= 0.5 else 0)
            true.append(df['label'].iloc[i+1])
        f1 = f1_score(true, preds, zero_division=0)
        auc = roc_auc_score(true, probs)
        return f1, auc
    except Exception as e:
        print(f"  评估 {stock} 失败: {e}")
        return np.nan, np.nan

def main():
    print("="*60)
    print("步骤24：多股票验证")
    print("="*60)
    features = FEATURE_COLS + ['llm_score']
    results = []
    for stock in OTHER_STOCKS:
        print(f"\n评估 {stock} ...")
        f1, auc = evaluate_single_stock(stock, features)
        results.append({
            'Stock': stock.split('_')[-1],
            'F1': f1,
            'AUC': auc
        })
        print(f"  F1: {f1:.4f}, AUC: {auc:.4f}")
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULT_DIR, "multi_stock_results.csv"), index=False)
    print("\n多股票验证结果已保存至 data/results/multi_stock_results.csv")
    print("步骤24完成！")

if __name__ == "__main__":
    main()