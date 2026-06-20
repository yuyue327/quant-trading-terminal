# step21_generate_real_charts.py
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤21真实版：生成基于真实预测的图表（混淆矩阵、ROC、概率分布等）
首先运行滚动预测并保存结果到文件，然后从文件加载绘图。
如果已经保存过预测结果，则直接绘图，节省时间。
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from step4_causal_attribution import load_data, FEATURE_COLS
from step15_backtest import rolling_predict  # 复用生成集成概率

RESULT_DIR = "data/results"
FIGURE_DIR = "data/figures"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

SAVE_PATH = os.path.join(RESULT_DIR, "real_predictions.pkl")


def get_real_predictions(force_recompute=False):
    """获取真实预测的概率和标签，如果文件存在且不强制重新计算则直接加载"""
    if not force_recompute and os.path.exists(SAVE_PATH):
        print(f"加载已保存的预测结果：{SAVE_PATH}")
        with open(SAVE_PATH, 'rb') as f:
            data = pickle.load(f)
        return data['y_true'], data['y_prob']

    print("重新生成预测结果（可能需要10-15分钟）...")
    stock = "A_sh.600036_招商银行"
    df = load_data(stock)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
    features = FEATURE_COLS + ['llm_score']

    # 滚动预测返回的是 Series，索引为日期，值为概率
    probs = rolling_predict(df, features, seq_len=20, window=60)
    # 获取对应的真实标签（注意 rolling_predict 内部已经对齐了日期，但我们需要提取 y_true）
    # 我们需要修改 rolling_predict 使其返回 y_true？或者重新实现一个快速版本。
    # 这里我们直接利用 step18 中写的 rolling_predictions_for_test 函数，但为了避免重复训练，我们调用一个新函数。
    # 简单方案：重新实现一个只返回 y_true 和 probs 的函数，但会再次训练，浪费时间。
    # 为了不重复训练，我们从 probs 的索引出发，从 df 中提取对应日期的 label。
    y_true = df.loc[probs.index, 'label'].values
    y_prob = probs.values
    # 保存
    with open(SAVE_PATH, 'wb') as f:
        pickle.dump({'y_true': y_true, 'y_prob': y_prob}, f)
    print("预测结果已保存。")
    return y_true, y_prob


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'real_confusion_matrix.png'), dpi=300)
    plt.close()
    print("混淆矩阵已保存")


def plot_prob_distribution(y_true, y_prob):
    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]
    plt.figure(figsize=(8, 5))
    plt.hist(pos_probs, bins=20, alpha=0.5, label='Up (True=1)', color='green')
    plt.hist(neg_probs, bins=20, alpha=0.5, label='Down (True=0)', color='red')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Predicted Probability Distribution by True Class')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'real_prob_distribution.png'), dpi=300)
    plt.close()
    print("概率分布图已保存")


def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'real_roc_curve.png'), dpi=300)
    plt.close()
    print(f"ROC曲线已保存，AUC={roc_auc:.3f}")


def plot_pr_curve(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label='PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'real_pr_curve.png'), dpi=300)
    plt.close()
    print("PR曲线已保存")


def main():
    print("=" * 60)
    print("步骤21真实版：生成基于真实预测的图表")
    print("=" * 60)
    y_true, y_prob = get_real_predictions(force_recompute=False)  # 如果需要重新计算改为True
    # 将概率转为类别（阈值0.5）
    y_pred = (y_prob >= 0.5).astype(int)
    plot_confusion_matrix(y_true, y_pred)
    plot_prob_distribution(y_true, y_prob)
    plot_roc_curve(y_true, y_prob)
    plot_pr_curve(y_true, y_prob)
    print("\n所有图表已生成在 data/figures/ 目录下")
    print("步骤21完成！")


if __name__ == "__main__":
    main()