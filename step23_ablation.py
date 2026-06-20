#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤23：消融实验 - 移除 LLM 特征，对比模型性能
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from step4_causal_attribution import load_data, FEATURE_COLS

RESULT_DIR = "data/results"
os.makedirs(RESULT_DIR, exist_ok=True)


def evaluate_without_llm(df, features_without_llm, seq_len=20, window=60):
    """滚动评估RF（无LLM特征）"""
    preds = []
    true = []
    for i in range(window + seq_len, len(df) - 1):
        train_end = i - seq_len
        train_start = train_end - window
        train_df = df.iloc[train_start:train_end]
        X_train = train_df[features_without_llm].values
        y_train = train_df['label'].values
        X_test = df[features_without_llm].iloc[i].values.reshape(1, -1)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        prob = rf.predict_proba(X_test)[0, 1]
        pred = 1 if prob >= 0.5 else 0
        preds.append(pred)
        true.append(df['label'].iloc[i + 1])
    f1 = f1_score(true, preds, zero_division=0)
    auc = roc_auc_score(true, preds)  # 使用预测标签计算AUC近似，或使用概率
    # 更精确：保存概率计算AUC
    # 重新计算使用概率
    probs = []
    for i in range(window + seq_len, len(df) - 1):
        train_end = i - seq_len
        train_start = train_end - window
        train_df = df.iloc[train_start:train_end]
        X_train = train_df[features_without_llm].values
        y_train = train_df['label'].values
        X_test = df[features_without_llm].iloc[i].values.reshape(1, -1)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        prob = rf.predict_proba(X_test)[0, 1]
        probs.append(prob)
    auc_prob = roc_auc_score(true, probs)
    return f1, auc_prob


def main():
    print("=" * 60)
    print("步骤23：消融实验 - 移除LLM特征")
    print("=" * 60)
    stock = "A_sh.600036_招商银行"
    df = load_data(stock)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)

    # 原始特征（包含LLM）
    features_full = FEATURE_COLS + ['llm_score']
    # 去除LLM
    features_no_llm = FEATURE_COLS  # llm_score被移除

    print("评估包含LLM特征的RF...")
    f1_full, auc_full = evaluate_without_llm(df, features_full)  # 重用函数但注意名字，实际应调用通用函数

    # 但上面的函数名是 evaluate_without_llm，实际评估任意特征集。为了清晰，重新写一个通用函数
    # 为了避免重复，直接调用两次即可
    def evaluate_rf(df, features, window=60, seq_len=20):
        preds = []
        probs = []
        true = []
        for i in range(window + seq_len, len(df) - 1):
            train_end = i - seq_len
            train_start = train_end - window
            train_df = df.iloc[train_start:train_end]
            X_train = train_df[features].values
            y_train = train_df['label'].values
            X_test = df[features].iloc[i].values.reshape(1, -1)
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            prob = rf.predict_proba(X_test)[0, 1]
            probs.append(prob)
            preds.append(1 if prob >= 0.5 else 0)
            true.append(df['label'].iloc[i + 1])
        f1 = f1_score(true, preds, zero_division=0)
        auc = roc_auc_score(true, probs)
        return f1, auc

    print("评估包含LLM特征的RF...")
    f1_full, auc_full = evaluate_rf(df, features_full)
    print(f"含LLM - F1: {f1_full:.4f}, AUC: {auc_full:.4f}")

    print("评估不含LLM特征的RF...")
    f1_no, auc_no = evaluate_rf(df, features_no_llm)
    print(f"无LLM - F1: {f1_no:.4f}, AUC: {auc_no:.4f}")

    # 保存结果
    results = pd.DataFrame({
        'Features': ['With LLM', 'Without LLM'],
        'F1': [f1_full, f1_no],
        'AUC': [auc_full, auc_no]
    })
    results.to_csv(os.path.join(RESULT_DIR, "ablation_results.csv"), index=False)
    print("\n消融结果已保存。")
    print("步骤23完成！")


if __name__ == "__main__":
    main()