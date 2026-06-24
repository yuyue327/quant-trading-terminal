#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤18（修正）：鲁棒性检验（仅招商银行）
- 时间序列交叉验证（不同窗口大小）
- McNemar检验比较RF与集成模型
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import torch
from step4_causal_attribution import load_data, FEATURE_COLS
from step15_backtest import train_lstm_quick, train_transformer_quick, DEVICE

RESULT_DIR = "data/results"
os.makedirs(RESULT_DIR, exist_ok=True)


def rolling_cv_f1(df, features, window_sizes=[30, 60, 90, 120], test_size=30):
    """对不同训练窗口大小进行滚动评估，返回平均F1"""
    results = {}
    for window in window_sizes:
        preds = []
        true = []
        for i in range(window + 20, len(df) - 1, test_size):
            train_end = i
            train_start = train_end - window
            test_start = train_end
            test_end = min(test_start + test_size, len(df) - 1)
            if test_end - test_start < 10:
                continue
            train_df = df.iloc[train_start:train_end]
            test_df = df.iloc[test_start:test_end]
            X_train = train_df[features].values
            y_train = train_df['label'].values
            X_test = test_df[features].values
            y_test = test_df['label'].values
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            preds.extend(y_pred)
            true.extend(y_test)
        f1 = f1_score(true, preds, zero_division=0)
        results[window] = f1
    return results


def mcnemar_test(y_true, y_pred_model1, y_pred_model2):
    """McNemar检验，返回p值"""
    from statsmodels.stats.contingency_tables import mcnemar
    a = np.sum((y_pred_model1 == y_true) & (y_pred_model2 == y_true))
    b = np.sum((y_pred_model1 != y_true) & (y_pred_model2 == y_true))
    c = np.sum((y_pred_model1 == y_true) & (y_pred_model2 != y_true))
    d = np.sum((y_pred_model1 != y_true) & (y_pred_model2 != y_true))
    table = [[a, b], [c, d]]
    result = mcnemar(table, exact=False, correction=True)
    return result.pvalue


def rolling_predictions_for_test(df, features, seq_len=20, window=60):
    """
    滚动评估，记录RF预测和集成预测（加权软投票）
    返回 y_true, y_pred_rf, y_pred_ensemble
    """
    y_true = []
    y_pred_rf = []
    y_pred_ensemble = []

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

        lstm_model = train_lstm_quick(X_train_scaled, y_train_seq, X_val_scaled, y_val_seq,
                                      input_dim=len(features), epochs=30)
        trans_model = train_transformer_quick(X_train_scaled, y_train_seq, X_val_scaled, y_val_seq,
                                              input_dim=len(features), epochs=20)

        # 随机森林
        train_df = df.iloc[train_start:train_end]
        X_train_rf = train_df[features].values
        y_train_rf = train_df['label'].values
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_rf, y_train_rf)
        X_test_rf = df[features].iloc[i].values.reshape(1, -1)
        rf_prob = rf.predict_proba(X_test_rf)[0, 1]
        rf_pred = 1 if rf_prob >= 0.5 else 0

        lstm_model.eval()
        trans_model.eval()
        with torch.no_grad():
            lstm_prob = lstm_model(torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)).item()
            trans_prob = trans_model(torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)).item()
        f1_rf, f1_lstm, f1_trans = 0.4527, 0.4451, 0.4481
        w_sum = f1_rf + f1_lstm + f1_trans
        weights = np.array([f1_rf, f1_lstm, f1_trans]) / w_sum
        ensemble_prob = weights[0] * rf_prob + weights[1] * lstm_prob + weights[2] * trans_prob
        ensemble_pred = 1 if ensemble_prob >= 0.5 else 0

        y_true.append(df['label'].iloc[i + 1])
        y_pred_rf.append(rf_pred)
        y_pred_ensemble.append(ensemble_pred)

        if (idx + 1) % 100 == 0:
            print(f"已处理 {idx + 1}/{total_points} 个测试点")

    return np.array(y_true), np.array(y_pred_rf), np.array(y_pred_ensemble)


def main():
    print("=" * 60)
    print("步骤18（修正）：鲁棒性检验（仅招商银行）")
    print("=" * 60)

    stock = "A_sh.600036_招商银行"
    df = load_data(stock)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
    features = FEATURE_COLS + ['llm_score']

    # 1. 时间序列交叉验证
    print("\n1. 时间序列交叉验证...")
    cv_results = rolling_cv_f1(df, features, window_sizes=[30, 60, 90, 120])
    cv_df = pd.DataFrame(list(cv_results.items()), columns=['Window Size', 'F1 Score'])
    print(cv_df)
    cv_df.to_csv(os.path.join(RESULT_DIR, "rolling_cv_results.csv"), index=False)

    # 2. McNemar检验（可选，耗时较长）
    print("\n2. 进行McNemar检验（滚动预测中，可能需要10-15分钟）...")
    y_true, y_pred_rf, y_pred_ensemble = rolling_predictions_for_test(df, features, seq_len=20, window=60)
    f1_rf = f1_score(y_true, y_pred_rf)
    f1_ens = f1_score(y_true, y_pred_ensemble)
    print(f"RF F1: {f1_rf:.4f}, Ensemble F1: {f1_ens:.4f}")
    p_val = mcnemar_test(y_true, y_pred_rf, y_pred_ensemble)
    print(f"McNemar检验 p-value: {p_val:.6f}")
    with open(os.path.join(RESULT_DIR, "mcnemar_result.txt"), "w") as f:
        f.write(f"RF F1: {f1_rf:.4f}\n")
        f.write(f"Ensemble F1: {f1_ens:.4f}\n")
        f.write(f"McNemar p-value: {p_val:.6f}\n")
    print("结果已保存到 data/results/mcnemar_result.txt")

    print("\n步骤18完成！")


if __name__ == "__main__":
    main()