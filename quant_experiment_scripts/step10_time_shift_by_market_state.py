#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤10：时间偏移细分分析 - 按市场状态（牛/熊/震荡）评估模型性能
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from step4_causal_attribution import load_data, FEATURE_COLS

plt.rcParams["font.family"] = ["Arial Unicode MS"]

RESULT_DIR = "data/results"
FIGURE_DIR = "data/figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

def classify_market_state(df, window=20):
    """
    根据价格与MA20的关系及MA20的斜率划分市场状态
    返回: series of 'bull', 'bear', 'sideways'
    """
    df = df.copy()
    df['MA20'] = df['close'].rolling(window).mean()
    df['MA20_slope'] = df['MA20'].diff(5)  # 5日斜率
    conditions = [
        (df['close'] > df['MA20']) & (df['MA20_slope'] > 0),
        (df['close'] < df['MA20']) & (df['MA20_slope'] < 0)
    ]
    choices = ['bull', 'bear']
    df['market_state'] = np.select(conditions, choices, default='sideways')
    return df['market_state']

def rolling_evaluation_by_state(df, features, window=60, state_col='market_state'):
    """按市场状态分别评估滚动窗口F1"""
    results = {'bull': [], 'bear': [], 'sideways': []}
    for i in range(window, len(df) - 1):
        train_idx = list(range(i-window, i))
        test_idx = i+1
        X_train = df.iloc[train_idx][features].values
        y_train = df.iloc[train_idx]['label'].values
        X_test = df.iloc[test_idx][features].values.reshape(1, -1)
        y_test = df.iloc[test_idx]['label']
        state = df.iloc[test_idx][state_col]
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)[0]
        if state in results:
            results[state].append((y_test, pred))
    # 计算每个状态的F1
    f1_dict = {}
    for state, pairs in results.items():
        if pairs:
            y_true = [p[0] for p in pairs]
            y_pred = [p[1] for p in pairs]
            f1_dict[state] = f1_score(y_true, y_pred, zero_division=0)
        else:
            f1_dict[state] = np.nan
    return f1_dict

def main():
    print("="*60)
    print("步骤10：时间偏移细分 - 市场状态分析")
    print("="*60)

    stock = "A_sh.600036_招商银行"
    print(f"分析股票: {stock}")

    df = load_data(stock)
    # 需要close列计算MA20（已在features中）
    df['market_state'] = classify_market_state(df)
    print("市场状态分布:")
    print(df['market_state'].value_counts())

    # 使用全特征 + LLM
    features = FEATURE_COLS + ['llm_score']
    f1_by_state = rolling_evaluation_by_state(df, features, window=60)

    print("\n各市场状态下的F1分数:")
    for state, f1 in f1_by_state.items():
        print(f"  {state}: {f1:.4f}")

    # 可视化
    states = list(f1_by_state.keys())
    scores = [f1_by_state[s] for s in states]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(states, scores, color=['green', 'red', 'gray'])
    plt.ylim(0, 0.6)
    plt.ylabel('F1 Score')
    plt.title('不同市场状态下的预测性能')
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{score:.3f}', ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'market_state_f1.png'), dpi=150)
    plt.close()

    # 保存结果
    result_df = pd.DataFrame([f1_by_state])
    result_df.to_csv(os.path.join(RESULT_DIR, "market_state_f1.csv"), index=False)
    print(f"\n结果已保存: {RESULT_DIR}/market_state_f1.csv")
    print(f"图表已保存: {FIGURE_DIR}/market_state_f1.png")

if __name__ == "__main__":
    main()