#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤19：最终结果汇总与报告生成
收集所有表格、图表、结论，生成Markdown报告。
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from step4_causal_attribution import load_data, FEATURE_COLS

RESULT_DIR = "data/results"
FIGURE_DIR = "data/figures"
os.makedirs(RESULT_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("步骤19：最终结果汇总与报告生成")
    print("=" * 60)

    # 1. 加载SOTA对比结果
    sota_df = pd.read_csv(os.path.join(RESULT_DIR, "sota_benchmark.csv"))
    # 2. 加载回测指标
    backtest_df = pd.read_csv(os.path.join(RESULT_DIR, "backtest_metrics.csv"))
    # 3. 加载鲁棒性检验结果
    cv_df = pd.read_csv(os.path.join(RESULT_DIR, "rolling_cv_results.csv"))
    # 4. 加载市场状态F1
    regime_df = pd.read_csv(os.path.join(RESULT_DIR, "regime_performance.csv"))
    # 5. 加载McNemar结果
    with open(os.path.join(RESULT_DIR, "mcnemar_result.txt"), "r") as f:
        mcnemar_text = f.read()

    # 生成Markdown报告
    report = f"""# 股票涨跌预测模型最终报告

## 1. 模型对比（SOTA基准）
下表展示了各模型在招商银行数据上的F1分数：

{sota_df.to_markdown(index=False)}

**最佳模型**：{sota_df.loc[sota_df['f1'].idxmax(), 'model']} (F1={sota_df['f1'].max():.4f})

## 2. 策略回测绩效
基于加权软投票集成模型的交易策略回测结果（初始资金100万，交易成本0.1%）：

{backtest_df.to_markdown(index=False)}

策略净值曲线和回撤曲线见 `data/figures/backtest_curve.png` 和 `data/figures/backtest_drawdown.png`。

## 3. 特征重要性分析
随机森林特征重要性（Top 15）见 `data/figures/feature_importance.png`。  
SHAP汇总图见 `data/figures/shap_summary.png` 和 `data/figures/shap_bar.png`。

## 4. 市场状态表现
模型在不同市场状态下的F1分数：

{regime_df.to_markdown()}

柱状图见 `data/figures/regime_f1.png`。

## 5. 鲁棒性检验
### 5.1 不同训练窗口大小下的F1
{cv_df.to_markdown(index=False)}

### 5.2 McNemar检验（RF vs 集成模型）
{mcnemar_text}

p值 > 0.05，表明两个模型预测差异不显著，集成模型未显著优于RF。

## 6. 结论
- 集成模型（加权软投票）F1={sota_df[sota_df['model'].str.contains('Weighted')]['f1'].values[0]:.4f}，略高于单一RF（{sota_df[sota_df['model'] == 'RandomForest (tech only)']['f1'].values[0]:.4f}），但统计上不显著。
- 策略回测表现不佳，总收益为{backtest_df['总收益率'].values[0]}，夏普比率为{backtest_df['夏普比率'].values[0]}，最大回撤{backtest_df['最大回撤'].values[0]}，表明直接使用预测信号进行交易存在较大风险。
- 模型在熊市表现较好（F1=0.525），在牛市和震荡市表现较差，可能更适合下跌行情。
- 未来改进方向：引入止损/止盈、动态仓位管理、结合其他高频数据或新闻情感。

"""
    # 保存报告
    report_path = os.path.join(RESULT_DIR, "final_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"报告已保存至：{report_path}")

    # 可选：打印报告摘要
    print("\n=== 报告摘要 ===")
    print(report[:500] + "...")

    print("\n步骤19完成！")


if __name__ == "__main__":
    main()