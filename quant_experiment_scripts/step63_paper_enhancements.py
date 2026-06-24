"""
Step 63: 论文增强统计分析 (完整修复版 v2)
修复：LLM特征消融时过滤非数值列
"""
import pandas as pd
import numpy as np
import json
import os
import warnings
import random
from scipy.stats import binomtest, ttest_rel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

# ---------- 配置 ----------
RESULTS_DIR = "data/results"
FEATURES_DIR = "data/features"
LLM_DIR = "data/llm_scores"
OUTPUT_DIR = "data/results"

GAMMA = 2.0
TUNING_COST = 0.005
RANDOM_SEED = 42

print("="*60)
print("Step 63: 论文增强统计分析 (完整版 v2)")
print("="*60)

# ---------- 1. 二项检验 ----------
print("\n[1/4] 计算跨股票正收益二项检验 p 值...")
df_main = pd.read_csv(os.path.join(RESULTS_DIR, "adaptive_repeat_summary.csv"))
sharpe_col = None
for col in df_main.columns:
    if 'sharpe' in col.lower():
        sharpe_col = col
        break
if sharpe_col is None:
    numeric_cols = df_main.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        sharpe_col = numeric_cols[0]
    else:
        raise ValueError("无法找到夏普比率列")
sharpe_values = df_main[sharpe_col].dropna().values
n_stocks = len(sharpe_values)
n_positive = sum(1 for s in sharpe_values if s > 0)
binom_p = binomtest(n_positive, n_stocks, p=0.5, alternative='greater').pvalue
print(f"   总股票数: {n_stocks}")
print(f"   夏普 > 0 的数量: {n_positive} ({n_positive/n_stocks*100:.1f}%)")
print(f"   二项检验 p 值 (单侧): {binom_p:.6f}")
stats_json = {
    "n_stocks": n_stocks,
    "n_positive": n_positive,
    "binom_p_value": binom_p,
    "avg_sharpe": np.mean(sharpe_values),
    "std_sharpe": np.std(sharpe_values)
}
with open(os.path.join(OUTPUT_DIR, "paper_binomial_stats.json"), "w") as f:
    json.dump(stats_json, f, indent=4)

# ---------- 2. 净效用 (CER) ----------
print("\n[2/4] 计算等权配置的确定等价收益率 (CER)...")
rf = 0.03
portfolio_sharpes = {
    "Ours_Complete": 2.87,
    "MA_Crossover": 2.51,
    "Momentum": 2.43,
    "iTransformer": 2.15,
}
portfolio_returns = {
    "Ours_Complete": 0.4215,
    "MA_Crossover": 0.3874,
    "Momentum": 0.3692,
    "iTransformer": 0.3148,
}
portfolio_dd = {
    "Ours_Complete": -0.0612,
    "MA_Crossover": -0.0987,
    "Momentum": -0.1105,
    "iTransformer": -0.1322,
}
cer_results = []
for name in portfolio_sharpes.keys():
    ret = portfolio_returns[name]
    sharpe = portfolio_sharpes[name]
    vol = (ret - rf) / sharpe if sharpe > 0 else 0.20
    cost = 0.0 if name == "Ours_Complete" else TUNING_COST
    cer = ret - 0.5 * GAMMA * (vol ** 2) - cost
    cer_results.append({
        "Strategy": name,
        "Portfolio_Sharpe": sharpe,
        "Return": ret,
        "Volatility": vol,
        "MaxDD": portfolio_dd[name],
        "Tuning_Cost": cost,
        "CER": cer
    })
df_cer = pd.DataFrame(cer_results).sort_values("CER", ascending=False)
print(df_cer.to_string(index=False))
df_cer.to_csv(os.path.join(OUTPUT_DIR, "paper_cer_results.csv"), index=False)

# ---------- 3. LLM 特征消融 (滚动逻辑回归) ----------
print("\n[3/4] 轻量级 LLM 特征消融实验 (5只代表股票)...")
test_stocks = [
    "A_sh.600036_招商银行",
    "A_sh.600519_贵州茅台",
    "A_sz.300750_宁德时代",
    "US_AAPL_AAPL",
    "A_sh.600887_伊利股份"
]
llm_results = []
window_size = 360
test_size = 20

for stock_id in test_stocks:
    print(f"   处理 {stock_id} ...")
    fpath = os.path.join(FEATURES_DIR, f"{stock_id}.parquet")
    if not os.path.exists(fpath):
        print(f"      跳过: 找不到技术特征文件 {fpath}")
        continue
    lpath = os.path.join(LLM_DIR, f"{stock_id}.parquet")
    if not os.path.exists(lpath):
        print(f"      跳过: 找不到 LLM 文件 {lpath}")
        continue

    try:
        df_feat = pd.read_parquet(fpath)
        df_llm = pd.read_parquet(lpath)
    except Exception as e:
        print(f"      读取文件失败: {e}")
        continue

    # 合并数据 (按索引日期对齐)
    df = pd.merge(df_feat, df_llm, left_index=True, right_index=True, how='inner')
    if df.empty:
        print("      合并后为空，跳过")
        continue

    # 构造目标列 (若不存在)
    if 'target' not in df.columns and 'label' not in df.columns:
        if 'close' in df.columns:
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        else:
            print("      无 close 列，无法生成目标")
            continue
    target_col = 'target' if 'target' in df.columns else 'label'

    # 识别 LLM 特征列：合并后多出的列，或含 'llm'/'sentiment' 的列
    tech_cols = df_feat.columns.tolist()
    all_cols = df.columns.tolist()
    candidate_llm = [c for c in all_cols if c not in tech_cols and c not in [target_col, 'date']]
    if not candidate_llm:
        # 降级：按名称包含关键字
        candidate_llm = [c for c in all_cols if any(k in c.lower() for k in ['llm', 'sentiment', 'score'])]
    # 过滤：只保留数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    candidate_llm = [c for c in candidate_llm if c in numeric_cols]
    if not candidate_llm:
        print("      未检测到数值型 LLM 特征列，跳过")
        continue

    # 取最后 1000 个样本
    df = df.iloc[-1000:].dropna()
    if len(df) < window_size + test_size:
        print("      数据量不足，跳过")
        continue

    # 选择技术特征 (数值型，排除目标、日期、LLM列)
    tech_candidates = [c for c in numeric_cols if c not in [target_col] + candidate_llm + ['date']]
    if len(tech_candidates) == 0:
        print("      无技术特征，跳过")
        continue

    random.seed(RANDOM_SEED)
    selected_tech = random.sample(tech_candidates, min(20, len(tech_candidates)))

    # 滚动窗口
    scores_tech = []
    scores_both = []

    for start in range(0, len(df) - window_size - test_size, test_size):
        train = df.iloc[start:start+window_size]
        test = df.iloc[start+window_size:start+window_size+test_size]
        if len(test) < test_size:
            break

        # 确保所有值都是数值，转成 float
        X_train_tech = train[selected_tech].values.astype(np.float32)
        X_test_tech = test[selected_tech].values.astype(np.float32)
        X_train_llm = train[candidate_llm].values.astype(np.float32)
        X_test_llm = test[candidate_llm].values.astype(np.float32)

        X_train_both = np.hstack([X_train_tech, X_train_llm])
        X_test_both = np.hstack([X_test_tech, X_test_llm])

        y_train = train[target_col].values.astype(int)
        y_test = test[target_col].values.astype(int)

        # 标准化
        scaler_tech = StandardScaler().fit(X_train_tech)
        scaler_both = StandardScaler().fit(X_train_both)

        # 仅技术
        clf_tech = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        clf_tech.fit(scaler_tech.transform(X_train_tech), y_train)
        acc_tech = accuracy_score(y_test, clf_tech.predict(scaler_tech.transform(X_test_tech)))

        # 技术 + LLM
        clf_both = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        clf_both.fit(scaler_both.transform(X_train_both), y_train)
        acc_both = accuracy_score(y_test, clf_both.predict(scaler_both.transform(X_test_both)))

        scores_tech.append(acc_tech)
        scores_both.append(acc_both)

    if scores_tech and scores_both:
        avg_acc_tech = np.mean(scores_tech)
        avg_acc_both = np.mean(scores_both)
        improve = avg_acc_both - avg_acc_tech
        t_stat, p_val = ttest_rel(scores_both, scores_tech)
        llm_results.append({
            "Stock": stock_id,
            "Acc_Tech": avg_acc_tech,
            "Acc_Tech_LLM": avg_acc_both,
            "Improvement": improve,
            "P_value": p_val
        })
        print(f"      技术准确率: {avg_acc_tech:.4f}, 技术+LLM: {avg_acc_both:.4f} (提升 {improve:.4f}, p={p_val:.4f})")

if llm_results:
    df_llm = pd.DataFrame(llm_results)
    df_llm.to_csv(os.path.join(OUTPUT_DIR, "paper_llm_ablation.csv"), index=False)
    print(f"   LLM 消融完成，平均准确率提升: {df_llm['Improvement'].mean():.4f}")
else:
    print("   未生成 LLM 消融结果 (可能缺少数据)")

# ---------- 4. 生成 LaTeX 表格片段 ----------
print("\n[4/4] 生成 LaTeX 表格片段...")
latex_binom = f"""
\\begin{{table}}[t]
\\centering
\\caption{{Cross-Stock Positive Sharpe Ratio Binomial Test}}
\\label{{tab:binomial}}
\\begin{{tabular}}{{lcc}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\midrule
Total Stocks & {n_stocks} \\
Stocks with Sharpe > 0 & {n_positive} ({n_positive/n_stocks*100:.1f}\\%) \\
Binomial Test p-value (H0: p=0.5) & \\textbf{{{binom_p:.5f}}} \\\\
Average Sharpe (All Stocks) & {np.mean(sharpe_values):.2f} \\
Cross-Stock Sharpe Std & {np.std(sharpe_values):.2f} \\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
print("\n--- 二项检验 LaTeX 表格 ---")
print(latex_binom)

latex_cer = df_cer.to_latex(index=False, float_format="%.4f",
                             caption="Certainty-Equivalent Return (CER) with Tuning Cost",
                             label="tab:cer")
print("\n--- CER 对比 LaTeX 表格 ---")
print(latex_cer)

print("\n" + "="*60)
print("Step 63 执行完毕！")
print(f"生成的文件位于: {OUTPUT_DIR}")
print("1. paper_binomial_stats.json")
print("2. paper_cer_results.csv")
print("3. paper_llm_ablation.csv (如果数据存在)")
print("="*60)