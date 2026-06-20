#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤11：与SOTA时序模型对比（LSTM作为代表）
注意：此处使用简单LSTM，实际论文中可引用PatchTST等
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from step4_causal_attribution import load_data, FEATURE_COLS
import matplotlib.pyplot as plt   # 新增这一行

plt.rcParams["font.family"] = ["Arial Unicode MS"]

RESULT_DIR = "data/results"
FIGURE_DIR = "data/figures"
os.makedirs(FIGURE_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step
        return self.sigmoid(self.fc(out)).squeeze()

def train_lstm(X_train, y_train, X_val, y_val, epochs=50, batch_size=32, lr=0.001):
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleLSTM(input_dim=X_train.shape[-1]).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                pred = model(Xb)
                val_loss += criterion(pred, yb).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(RESULT_DIR, "best_lstm.pt"))
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
    return model

def prepare_sequences(df, features, seq_len=20, forecast_horizon=1):
    X, y = [], []
    for i in range(seq_len, len(df)-forecast_horizon):
        X.append(df[features].iloc[i-seq_len:i].values)
        y.append(df['label'].iloc[i+forecast_horizon-1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def rolling_evaluation_lstm(df, features, seq_len=20, window=60):
    """滚动窗口评估LSTM（每次重新训练）"""
    preds = []
    for i in range(window + seq_len, len(df) - 1):
        # 训练窗口
        train_end = i - seq_len
        train_start = train_end - window
        X_train, y_train = prepare_sequences(df.iloc[train_start:train_end+seq_len], features, seq_len)
        # 测试点
        X_test_seq = df[features].iloc[i-seq_len:i].values.reshape(1, seq_len, -1)
        # 标准化（对每个窗口独立标准化，简单处理）
        scaler = StandardScaler()
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
        X_test_scaled = scaler.transform(X_test_seq.reshape(-1, X_test_seq.shape[-1])).reshape(X_test_seq.shape)

        # 训练模型
        model = SimpleLSTM(input_dim=len(features)).to(DEVICE)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # 快速训练几个epoch
        for epoch in range(20):
            model.train()
            optimizer.zero_grad()
            pred = model(torch.tensor(X_train_scaled, dtype=torch.float32).to(DEVICE))
            loss = criterion(pred, torch.tensor(y_train, dtype=torch.float32).to(DEVICE))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prob = model(torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)).item()
        preds.append(1 if prob >= 0.5 else 0)
    y_true = df['label'].iloc[window+seq_len+1:len(df)].values
    return f1_score(y_true, preds, zero_division=0)

def main():
    print("="*60)
    print("步骤11：与SOTA时序模型对比（LSTM）")
    print("="*60)
    stock = "A_sh.600036_招商银行"
    df = load_data(stock)
    features = FEATURE_COLS + ['llm_score']
    seq_len = 20
    window = 60
    print(f"股票: {stock}, 特征数: {len(features)}")
    f1_lstm = rolling_evaluation_lstm(df, features, seq_len=seq_len, window=window)
    print(f"LSTM滚动窗口F1: {f1_lstm:.4f}")
    # 对比之前的RF + 门控结果
    baseline_f1 = 0.4527  # 纯技术RF
    gate_f1 = 0.4514
    print(f"基线RF F1: {baseline_f1:.4f}, Transformer门控: {gate_f1:.4f}")

    # 保存结果
    pd.DataFrame({
        'model': ['RandomForest (tech only)', 'Transformer Gate', 'LSTM (seq+tech+LLM)'],
        'f1': [baseline_f1, gate_f1, f1_lstm]
    }).to_csv(os.path.join(RESULT_DIR, "sota_benchmark.csv"), index=False)
    print(f"结果已保存: {RESULT_DIR}/sota_benchmark.csv")

if __name__ == "__main__":
    main()