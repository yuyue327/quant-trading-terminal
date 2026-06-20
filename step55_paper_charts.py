#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step55_paper_charts.py
生成论文图表：消融对比箱线图、夏普散点图、回撤对比
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

RESULT_DIR = "data/results"
FIGURE_DIR = "data/figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

# 读取消融结果
df_ablation = pd.read_csv(os.path.join(RESULT_DIR, "ablation_full_results.csv"))

print("=" * 60)
print("step55：论文图表生成")
print("=" * 60)

# --- 图1：消融实验箱线图 ---
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='config', y='sharpe', data=df_ablation, palette='Set2', ax=ax)
ax.set_xlabel('配置')
ax.set_ylabel('夏普比率')
ax.set_title('消融实验：各配置夏普比率分布（18只股票）')
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "ablation_boxplot.png"), dpi=300)
print("✅ 图1：ablation_boxplot.png")

# --- 图2：多股票夏普散点图 ---
pivot_sharpe = df_ablation.pivot(index='stock', columns='config', values='sharpe')
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#00E5FF' if s.startswith('A_') else '#FF6B6B' for s in pivot_sharpe.index]
ax.scatter(pivot_sharpe['完整模型'], pivot_sharpe['无状态自适应'], c=colors, alpha=0.7, s=50)
ax.plot([-2, 7], [-2, 7], '--', color='gray', alpha=0.5)
ax.set_xlabel('完整模型 夏普')
ax.set_ylabel('无状态自适应 夏普')
ax.set_title('完整模型 vs 无状态自适应')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "sharpe_scatter.png"), dpi=300)
print("✅ 图2：sharpe_scatter.png")

# --- 图3：回撤对比 ---
pivot_dd = df_ablation.pivot(index='stock', columns='config', values='max_drawdown')
pivot_dd_mean = pivot_dd.mean(axis=0)
fig, ax = plt.subplots(figsize=(10, 6))
pivot_dd_mean.plot(kind='bar', ax=ax, color=['#00E5FF', '#F59E0B', '#FF6B6B', '#8B5CF6'])
ax.set_xlabel('配置')
ax.set_ylabel('平均最大回撤 (%)')
ax.set_title('各配置平均最大回撤（18只股票）')
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "drawdown_comparison.png"), dpi=300)
print("✅ 图3：drawdown_comparison.png")

# --- 图4：性能雷达图（分行业）---
# 按行业分组
sectors = {
    '银行': ['A_sh.600036_招商银行', 'A_sz.000001_平安银行', 'A_sz.002142_宁波银行'],
    '券商': ['A_sh.600030_中信证券', 'A_sh.601688_华泰证券', 'A_sz.300059_东方财富'],
    '消费': ['A_sz.000858_五粮液', 'A_sh.600519_贵州茅台', 'A_sh.600887_伊利股份', 'A_sz.000568_泸州老窖'],
    '新能源': ['A_sz.300750_宁德时代', 'A_sh.601012_隆基绿能', 'A_sz.002594_比亚迪'],
    '科技': ['US_AAPL_AAPL', 'US_MSFT_MSFT', 'US_NVDA_NVDA'],
}
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(sectors))
width = 0.2
for i, config in enumerate(['完整模型', '无状态自适应', '无波动率目标', '无止损失']):
    means = []
    for sector, stocks in sectors.items():
        sector_data = df_ablation[(df_ablation['stock'].isin(stocks)) & (df_ablation['config'] == config)]
        means.append(sector_data['sharpe'].mean())
    ax.bar(x + i*width, means, width, label=config)
ax.set_xlabel('行业')
ax.set_ylabel('平均夏普比率')
ax.set_title('各行业在不同配置下的平均夏普')
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(sectors.keys())
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "sector_analysis.png"), dpi=300)
print("✅ 图4：sector_analysis.png")

print(f"\n✅ step55 完成！所有图表已保存至 {FIGURE_DIR}/")