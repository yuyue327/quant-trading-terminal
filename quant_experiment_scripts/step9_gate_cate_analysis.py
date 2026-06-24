#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤9：门控合理性分析 - 验证门控权重与CATE的相关性
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from step4_causal_attribution import causal_forest_cate, load_data, FEATURE_COLS

# 设置中文字体（Mac）
plt.rcParams["font.family"] = ["Arial Unicode MS"]

# 路径配置
RESULT_DIR = "data/results"
FIGURE_DIR = "data/figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

def compute_cate_timeseries(stock_name):
    """计算每只股票的逐日CATE（使用因果森林）"""
    df = load_data(stock_name)
    # 需要IV列，但step4中用了模拟IV，这里复用
    np.random.seed(42)
    df['iv_temperature'] = df['llm_score'].diff().fillna(0) + np.random.normal(0, 5, len(df))
    cate, _ = causal_forest_cate(df, FEATURE_COLS)
    # 对齐日期（最后len(cate)个样本）
    dates = df['date'].iloc[-len(cate):].values
    return dates, cate

def main():
    print("=" * 60)
    print("步骤9：门控权重与CATE相关性分析")
    print("=" * 60)

    stock = "A_sh.600036_招商银行"
    print(f"\n分析股票: {stock}")

    # 1. 加载门控权重（已有）
    gate_df = pd.read_csv(os.path.join(RESULT_DIR, "transformer_gate_weights.csv"))
    gate_df['date'] = pd.to_datetime(gate_df['date'])
    print(f"门控权重数据: {len(gate_df)} 条")

    # 2. 计算CATE时间序列
    dates_cate, cate = compute_cate_timeseries(stock)
    cate_df = pd.DataFrame({'date': dates_cate, 'cate': cate})
    print(f"CATE数据: {len(cate_df)} 条")

    # 3. 对齐两个时间序列（inner join on date）
    merged = pd.merge(gate_df, cate_df, on='date', how='inner')
    print(f"对齐后数据: {len(merged)} 条")
    if len(merged) == 0:
        print("错误：门控权重和CATE时间无重叠，请检查日期范围")
        return

    # 4. 计算相关系数
    corr = merged['gate_weight'].corr(merged['cate'])
    print(f"门控权重与CATE的Pearson相关系数: {corr:.4f}")

    # 5. 绘图：散点图 + 回归线
    plt.figure(figsize=(10, 6))
    sns.regplot(data=merged, x='cate', y='gate_weight', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    plt.title(f'门控权重 vs CATE (相关性: {corr:.3f})')
    plt.xlabel('CATE (LLM因果效应强度)')
    plt.ylabel('Transformer门控权重 (越高越信任LLM)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'gate_vs_cate_scatter.png'), dpi=150)
    plt.close()

    # 6. 箱线图：按CATE高低分组
    merged['cate_group'] = pd.cut(merged['cate'], bins=[-np.inf, 0, np.inf], labels=['负/零CATE', '正CATE'])
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=merged, x='cate_group', y='gate_weight')
    plt.title('不同CATE组下的门控权重分布')
    plt.xlabel('CATE类别')
    plt.ylabel('门控权重')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'gate_vs_cate_box.png'), dpi=150)
    plt.close()

    # 7. 保存对齐后的数据供后续分析
    merged.to_csv(os.path.join(RESULT_DIR, "gate_cate_aligned.csv"), index=False)
    print(f"\n结果已保存:")
    print(f"  - 散点图: {FIGURE_DIR}/gate_vs_cate_scatter.png")
    print(f"  - 箱线图: {FIGURE_DIR}/gate_vs_cate_box.png")
    print(f"  - 对齐数据: {RESULT_DIR}/gate_cate_aligned.csv")

if __name__ == "__main__":
    main()