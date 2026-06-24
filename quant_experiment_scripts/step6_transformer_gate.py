#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段6：时序因果动态门控
1. Transformer时序编码技术指标
2. 动态门控网络（可微分门控）
3. 门控权重可视化与解释
4. 与静态门控、硬门控对比
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
    'A_sh.600030_中信证券': 'securities',
    'A_sh.601688_华泰证券': 'securities',
    'A_sz.300059_东方财富': 'securities',
    'A_sh.600519_贵州茅台': 'liquor',
    'A_sz.000858_五粮液': 'liquor',
    'A_sz.000568_泸州老窖': 'liquor',
    'A_sz.000333_美的集团': 'consumer',
    'A_sz.000651_格力电器': 'consumer',
    'A_sh.600887_伊利股份': 'consumer',
    'A_sz.300750_宁德时代': 'new_energy',
    'A_sz.002594_比亚迪': 'new_energy',
    'A_sh.601012_隆基绿能': 'new_energy',
    'US_AAPL_AAPL': 'us_tech',
    'US_MSFT_MSFT': 'us_tech',
    'US_NVDA_NVDA': 'us_tech',
}

FEATURE_COLS = [
    'MA5', 'MA10', 'MA20', 'MA60', 'EMA12', 'EMA26',
    'MACD', 'MACD_signal', 'MACD_hist', 'RSI',
    'BB_upper', 'BB_middle', 'BB_lower', 'BB_width', 'BB_pct',
    'ATR', 'volume_ratio', 'pct_change', 'high_low_pct',
    'close_position', 'volatility_5', 'volatility_20'
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

def load_data(stock_name):
    feat_path = os.path.join(FEATURE_DIR, f"{stock_name}.parquet")
    llm_path = os.path.join(LLM_DIR, f"{stock_name}.parquet")
    df_feat = pd.read_parquet(feat_path)
    df_llm = pd.read_parquet(llm_path)
    return df_feat.merge(df_llm, on='date', how='inner')


class TransformerGate(nn.Module):
    """基于Transformer的时序门控网络"""
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_gate = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.transformer(x)
        # 取最后一个时间步的输出
        x = x[:, -1, :]
        gate = self.sigmoid(self.fc_gate(x))
        return gate


def prepare_sequences(df, features, seq_len=20):
    """构建时序序列数据"""
    X_seq, y_seq, llm_seq = [], [], []
    for i in range(seq_len, len(df) - 1):
        X_seq.append(df[features].iloc[i-seq_len:i].values)
        y_seq.append(df['label'].iloc[i+1])  # 预测下一日
        llm_seq.append(df['llm_score'].iloc[i])
    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32)
    llm_seq = np.array(llm_seq, dtype=np.float32)
    return X_seq, y_seq, llm_seq


def train_transformer_gate(stock_name, seq_len=20, epochs=50, batch_size=32, lr=0.001):
    """训练Transformer门控网络"""
    df = load_data(stock_name)
    X_seq, y_seq, llm_seq = prepare_sequences(df, FEATURE_COLS, seq_len)
    # 标准化
    scaler = StandardScaler()
    X_flat = X_seq.reshape(-1, X_seq.shape[-1])
    X_scaled = scaler.fit_transform(X_flat).reshape(X_seq.shape)
    # 划分训练/验证
    split = int(0.8 * len(X_seq))
    train_X, val_X = X_scaled[:split], X_scaled[split:]
    train_y, val_y = y_seq[:split], y_seq[split:]
    train_llm, val_llm = llm_seq[:split], llm_seq[split:]

    train_X = torch.tensor(train_X, dtype=torch.float32).to(DEVICE)
    val_X = torch.tensor(val_X, dtype=torch.float32).to(DEVICE)
    train_y = torch.tensor(train_y, dtype=torch.float32).to(DEVICE)
    val_y = torch.tensor(val_y, dtype=torch.float32).to(DEVICE)
    train_llm = torch.tensor(train_llm, dtype=torch.float32).to(DEVICE)
    val_llm = torch.tensor(val_llm, dtype=torch.float32).to(DEVICE)

    model = TransformerGate(input_dim=len(FEATURE_COLS)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_X))
        total_loss = 0
        for i in range(0, len(train_X), batch_size):
            idx = perm[i:i+batch_size]
            batch_X = train_X[idx]
            batch_y = train_y[idx]
            optimizer.zero_grad()
            gate = model(batch_X).squeeze()
            loss = criterion(gate, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证
        model.eval()
        with torch.no_grad():
            val_gate = model(val_X).squeeze()
            val_loss = criterion(val_gate, val_y)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(RESULT_DIR, f"transformer_gate_{stock_name}.pt"))

        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(train_X):.4f}, Val Loss: {val_loss:.4f}")

    return model, scaler


def evaluate_gate(stock_name, model, scaler, seq_len=20):
    """评估门控网络的预测性能，并与硬门控、静态融合对比"""
    df = load_data(stock_name)
    X_seq, y_seq, llm_seq = prepare_sequences(df, FEATURE_COLS, seq_len)
    X_flat = X_seq.reshape(-1, X_seq.shape[-1])
    X_scaled = scaler.transform(X_flat).reshape(X_seq.shape)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)

    model.eval()
    with torch.no_grad():
        gate_weights = model(X_tensor).squeeze().cpu().numpy()

    # 使用门控权重进行预测（加权融合）
    # 简单融合：预测 = gate * LLM_pred + (1-gate) * tech_pred
    # 这里我们用滚动窗口随机森林作为基预测器
    window = 60
    tech_preds, llm_preds, gate_preds = [], [], []
    for i in range(window, len(df) - 1):
        train_idx = list(range(i-window, i))
        test_idx = [i]
        # 技术指标模型
        X_train_tech = df.iloc[train_idx][FEATURE_COLS].values
        y_train = df.iloc[train_idx]['label'].values
        X_test_tech = df.iloc[test_idx][FEATURE_COLS].values
        clf_tech = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf_tech.fit(X_train_tech, y_train)
        tech_pred = clf_tech.predict(X_test_tech)[0]
        tech_preds.append(tech_pred)

        # LLM模型
        X_train_llm = df.iloc[train_idx][FEATURE_COLS + ['llm_score']].values
        X_test_llm = df.iloc[test_idx][FEATURE_COLS + ['llm_score']].values
        clf_llm = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf_llm.fit(X_train_llm, y_train)
        llm_pred = clf_llm.predict(X_test_llm)[0]
        llm_preds.append(llm_pred)

        # 门控融合
        weight = gate_weights[i - window]
        final_pred = 1 if (weight * llm_pred + (1-weight) * tech_pred) > 0.5 else 0
        gate_preds.append(final_pred)

    y_true = df['label'].iloc[window:-1].values
    f1_tech = f1_score(y_true, tech_preds, zero_division=0)
    f1_llm = f1_score(y_true, llm_preds, zero_division=0)
    f1_gate = f1_score(y_true, gate_preds, zero_division=0)

    return f1_tech, f1_llm, f1_gate, gate_weights


def run_all_transformer_gate():
    print("="*60)
    print("阶段6：时序因果动态门控")
    print("="*60)

    # 以招商银行为例训练Transformer门控
    stock = 'A_sh.600036_招商银行'
    print(f"\n训练Transformer门控 on {stock}...")
    model, scaler = train_transformer_gate(stock, seq_len=20, epochs=30)

    # 评估
    f1_tech, f1_llm, f1_gate, gate_weights = evaluate_gate(stock, model, scaler)
    print(f"\n评估结果 on {stock}:")
    print(f"  F1 (Technical only): {f1_tech:.4f}")
    print(f"  F1 (LLM + Technical): {f1_llm:.4f}")
    print(f"  F1 (Transformer Gate): {f1_gate:.4f}")

    # 可视化门控权重
    df = load_data(stock)
    X_seq, _, _ = prepare_sequences(df, FEATURE_COLS, 20)
    dates = df['date'].iloc[20:len(X_seq)+20].values
    plt.figure(figsize=(14, 5))
    plt.plot(dates, gate_weights, alpha=0.7)
    plt.axhline(y=0.5, color='red', linestyle='--', label='Threshold')
    plt.title(f'Transformer Gate Weights over Time ({stock})')
    plt.xlabel('Date')
    plt.ylabel('Gate Weight (higher = trust LLM more)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'transformer_gate_weights.png'), dpi=150)
    plt.close()

    # 保存结果
    results = pd.DataFrame({
        'date': dates[:len(gate_weights)],
        'gate_weight': gate_weights
    })
    results.to_csv(os.path.join(RESULT_DIR, "transformer_gate_weights.csv"), index=False)

    # 扩展到其他代表性股票（简化版：直接使用训练好的模型权重，不再重训练）
    print("\n扩展到其他股票（使用已训练模型评估）...")
    other_stocks = ['A_sz.000001_平安银行', 'A_sz.300750_宁德时代', 'A_sh.600519_贵州茅台']
    all_results = []
    for s in other_stocks:
        try:
            _, _, f1_g, _ = evaluate_gate(s, model, scaler)
            all_results.append({'stock': s, 'f1_gate': f1_g})
        except Exception as e:
            print(f"  Error on {s}: {e}")

    df_extend = pd.DataFrame(all_results)
    df_extend.to_csv(os.path.join(RESULT_DIR, "transformer_gate_extended.csv"), index=False)
    print(df_extend)

    print("\n阶段6完成！结果保存在 data/results/ 和 data/figures/")

if __name__ == "__main__":
    run_all_transformer_gate()