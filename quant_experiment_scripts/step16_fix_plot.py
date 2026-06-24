import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RESULT_DIR = "data/results"
FIGURE_DIR = "data/figures"

results_df = pd.read_csv(os.path.join(RESULT_DIR, "threshold_stop_tuning.csv"))
# 转换夏普比率为数值
results_df['夏普比率'] = results_df['夏普比率'].astype(float)
pivot = results_df.pivot(index='阈值', columns='止损', values='夏普比率')

plt.figure(figsize=(8, 6))
plt.imshow(pivot.values, cmap='RdYlGn', aspect='auto')
plt.colorbar(label='Sharpe Ratio')
plt.xticks(range(len(pivot.columns)), pivot.columns)
plt.yticks(range(len(pivot.index)), pivot.index)
plt.xlabel('Stop Loss')
plt.ylabel('Threshold')
plt.title('Sharpe Ratio Heatmap')
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        plt.text(j, i, f"{pivot.values[i, j]:.2f}", ha='center', va='center')
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'sharpe_heatmap.png'), dpi=300)
print("热力图已修复并保存")