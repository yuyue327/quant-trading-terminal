#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段7：强化学习自适应门控（上下文赌博机）
- LinUCB算法在线学习最优门控策略
- 与静态门控、硬门控对比
- 累积遗憾分析
"""
import seaborn as sns
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ========== 配置 ==========
FEATURE_DIR = "data/features"
LLM_DIR = "data/llm_scores"
RESULT_DIR = "data/results"
FIGURE_DIR = "data/figures"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

STOCKS = {
    'A_sh.600036_招商银行': 'bank',
    'A_sz.000001_平安银行': 'bank',
    'A_sz.002142_宁波银行': 'bank',
}

FEATURE_COLS = [
    'MA5', 'MA10', 'MA20', 'MA60', 'EMA12', 'EMA26',
    'MACD', 'MACD_signal', 'MACD_hist', 'RSI',
    'BB_upper', 'BB_middle', 'BB_lower', 'BB_width', 'BB_pct',
    'ATR', 'volume_ratio', 'pct_change', 'high_low_pct',
    'close_position', 'volatility_5', 'volatility_20'
]

def load_data(stock_name):
    feat_path = os.path.join(FEATURE_DIR, f"{stock_name}.parquet")
    llm_path = os.path.join(LLM_DIR, f"{stock_name}.parquet")
    df_feat = pd.read_parquet(feat_path)
    df_llm = pd.read_parquet(llm_path)
    return df_feat.merge(df_llm, on='date', how='inner')

class LinUCB:
    """LinUCB上下文赌博机算法"""
    def __init__(self, n_actions, n_features, alpha=1.0):
        self.n_actions = n_actions
        self.alpha = alpha
        self.A = [np.identity(n_features) for _ in range(n_actions)]
        self.b = [np.zeros(n_features) for _ in range(n_actions)]

    def predict(self, context):
        """选择动作（使用UCB）"""
        p = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            p[a] = theta @ context + self.alpha * np.sqrt(context @ A_inv @ context)
        return np.argmax(p)

    def update(self, action, context, reward):
        """更新模型参数"""
        self.A[action] += np.outer(context, context)
        self.b[action] += reward * context

def contextual_bandit_gate(stock_name, window=60, alpha=1.0):
    """使用LinUCB在线学习门控策略"""
    df = load_data(stock_name)
    n_features = len(FEATURE_COLS) + 1  # 加上LLM分数作为特征
    bandit = LinUCB(n_actions=2, n_features=n_features, alpha=alpha)

    actions = []  # 记录选择（0=仅技术指标，1=LLM+技术指标）
    rewards = []  # 记录即时奖励（预测正确=1，错误=0）
    regret = []   # 累积遗憾

    for i in tqdm(range(window, len(df)-1), desc=f"Online learning {stock_name}"):
        # 构建上下文特征（当前时刻的技术指标 + LLM分数）
        context = np.hstack([df[FEATURE_COLS].iloc[i].values, [df['llm_score'].iloc[i] / 100.0]])
        context = context.ravel()

        # 选择动作
        action = bandit.predict(context)
        actions.append(action)

        # 根据动作训练模型并预测
        train_idx = list(range(i-window, i))
        test_idx = [i+1]  # 预测下一日

        if action == 0:
            # 仅技术指标
            X_train = df.iloc[train_idx][FEATURE_COLS].values
        else:
            # LLM+技术指标
            X_train = df.iloc[train_idx][FEATURE_COLS + ['llm_score']].values

        y_train = df.iloc[train_idx]['label'].values
        X_test = df.iloc[test_idx][FEATURE_COLS + (['llm_score'] if action == 1 else [])].values
        y_test = df.iloc[test_idx]['label'].values

        clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)[0]

        reward = 1 if pred == y_test else 0
        rewards.append(reward)
        bandit.update(action, context, reward)

        # 计算遗憾（与最优固定策略的差距）
        # 最优固定策略：取历史平均奖励最高的动作
        avg_reward_0 = np.mean([rewards[j] for j in range(len(rewards)) if actions[j]==0]) if any(a==0 for a in actions) else 0
        avg_reward_1 = np.mean([rewards[j] for j in range(len(rewards)) if actions[j]==1]) if any(a==1 for a in actions) else 0
        best_avg = max(avg_reward_0, avg_reward_1)
        current_avg = np.mean(rewards)
        regret.append(best_avg - current_avg)

    # 计算累积遗憾
    cumulative_regret = np.cumsum(regret)

    # 计算整体F1
    f1 = f1_score(df['label'].iloc[window+1:len(df)], rewards, zero_division=0)

    return actions, rewards, cumulative_regret, f1

def static_baseline(df, window, use_llm=True):
    """静态基线（固定策略）"""
    preds = []
    for i in range(window, len(df)-1):
        train_idx = list(range(i-window, i))
        test_idx = [i+1]
        if use_llm:
            X_train = df.iloc[train_idx][FEATURE_COLS + ['llm_score']].values
        else:
            X_train = df.iloc[train_idx][FEATURE_COLS].values
        y_train = df.iloc[train_idx]['label'].values
        X_test = df.iloc[test_idx][FEATURE_COLS + (['llm_score'] if use_llm else [])].values
        y_test = df.iloc[test_idx]['label'].values
        clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)[0]
        preds.append(pred)
    y_true = df['label'].iloc[window+1:len(df)].values
    return f1_score(y_true, preds, zero_division=0)

def run_all_bandit():
    print("="*60)
    print("阶段7：强化学习自适应门控（LinUCB）")
    print("="*60)

    results = []
    for stock in STOCKS.keys():
        print(f"\n处理 {stock}...")
        actions, rewards, cum_regret, f1_bandit = contextual_bandit_gate(stock, window=60, alpha=1.0)
        df = load_data(stock)
        f1_tech = static_baseline(df, 60, use_llm=False)
        f1_llm = static_baseline(df, 60, use_llm=True)

        results.append({
            'stock': stock,
            'f1_tech': f1_tech,
            'f1_llm': f1_llm,
            'f1_bandit': f1_bandit
        })

        # 可视化
        plt.figure(figsize=(12, 8))
        plt.subplot(2,1,1)
        plt.plot(cum_regret)
        plt.title(f'Cumulative Regret ({stock})')
        plt.xlabel('Time step')
        plt.ylabel('Cumulative Regret')

        plt.subplot(2,1,2)
        plt.plot(actions[:200], alpha=0.5)
        plt.title('Actions (0=Tech, 1=LLM+Tech)')
        plt.xlabel('Time step')
        plt.ylabel('Action')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, f'bandit_{stock}.png'), dpi=150)
        plt.close()

    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(RESULT_DIR, "bandit_results.csv"), index=False)
    print("\n结果对比:")
    print(df_results)

    # 汇总图表
    df_melt = df_results.melt(id_vars=['stock'], value_vars=['f1_tech', 'f1_llm', 'f1_bandit'], var_name='method', value_name='f1')
    plt.figure(figsize=(10,6))
    sns.barplot(data=df_melt, x='method', y='f1')
    plt.title('F1 Score Comparison: Bandit vs Static')
    plt.ylim(0,0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'bandit_comparison.png'), dpi=150)
    plt.close()

    print("\n阶段7完成！")

if __name__ == "__main__":
    run_all_bandit()