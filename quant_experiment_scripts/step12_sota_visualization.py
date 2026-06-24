#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤12：SOTA对比结果可视化
生成柱状图比较LSTM、RandomForest和Transformer门控的F1分数
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（根据系统可用字体调整，macOS常见）
plt.rcParams["font.family"] = ["Arial Unicode MS"]  # macOS
# 若上述字体不可用，可尝试 'Heiti TC' 或 'SimHei'；或注释掉使用英文

RESULT_DIR = "data/results"
FIGURE_DIR = "data/figures"
os.makedirs(FIGURE_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("步骤12：SOTA对比结果可视化")
    print("=" * 60)

    # 读取第11步保存的benchmark数据
    csv_path = os.path.join(RESULT_DIR, "sota_benchmark.csv")
    if not os.path.exists(csv_path):
        print(f"错误：未找到 {csv_path}，请先运行 step11_sota_benchmark.py")
        return

    df = pd.read_csv(csv_path)
    print("对比数据：")
    print(df)

    # 提取模型名称和F1分数
    models = df['model'].tolist()
    f1_scores = df['f1'].tolist()

    # 定义颜色（可选）
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝、橙、绿

    # 创建柱状图
    plt.figure(figsize=(8, 6))
    bars = plt.bar(models, f1_scores, color=colors, edgecolor='black', linewidth=1.2)

    # 在柱顶标注数值
    for bar, score in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{score:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 设置y轴范围（略高于最大值）
    y_max = max(f1_scores) + 0.03
    plt.ylim(0, y_max)
    plt.ylabel('F1 Score', fontsize=13)
    plt.title('SOTA 时序模型对比 (LSTM vs 基线)', fontsize=14, pad=15)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()

    # 保存图片
    out_img = os.path.join(FIGURE_DIR, "sota_comparison.png")
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    print(f"柱状图已保存至：{out_img}")
    plt.show()

    # 可选：输出结论
    print("\n=== 比较结论 ===")
    best_model = models[np.argmax(f1_scores)]
    best_f1 = max(f1_scores)
    print(f"最佳模型：{best_model} (F1={best_f1:.4f})")
    print(
        f"LSTM 的 F1 分数 ({f1_scores[2]:.4f}) 低于传统RF ({f1_scores[0]:.4f}) 和 Transformer门控 ({f1_scores[1]:.4f})。")
    print("可能原因：LSTM 在滚动重训练模式下未能充分捕捉长期依赖，或数据量较少时不如树模型鲁棒。")


if __name__ == "__main__":
    main()