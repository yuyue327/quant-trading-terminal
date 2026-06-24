#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤13：使用标准Transformer编码器进行时序预测对比
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

RESULT_DIR = "data/results"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model))  # 最大长度1000
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                    dim_feedforward=dim_feedforward,
                                                    dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.embedding(x)  # (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)
        x = x[:, -1, :]  # last time step
        return self.sigmoid(self.fc(x)).squeeze()

def prepare_sequences(df, features, seq_len=20, forecast_horizon=1):
    X, y = [], []
    for i in range(seq_len, len(df)-forecast_horizon):
        X.append(df[features].iloc[i-seq_len:i].values)
        y.append(df['label'].iloc[i+forecast_horizon-1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def rolling_evaluation_transformer(df, features, seq_len=20, window=60, epochs=30, batch_size=32, lr=0.001):
    preds = []
    for i in range(window + seq_len, len(df) - 1):
        train_end = i - seq_len
        train_start = train_end - window
        X_train, y_train = prepare_sequences(df.iloc[train_start:train_end+seq_len], features, seq_len)
        X_test_seq = df[features].iloc[i-seq_len:i].values.reshape(1, seq_len, -1)

        # 标准化
        scaler = StandardScaler()
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
        X_test_scaled = scaler.transform(X_test_seq.reshape(-1, X_test_seq.shape[-1])).reshape(X_test_seq.shape)

        # 转换为Tensor
        train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32),
                                      torch.tensor(y_train, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = TransformerEncoder(input_dim=len(features)).to(DEVICE)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

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

        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)
            prob = model(X_test_tensor).item()
        preds.append(1 if prob >= 0.5 else 0)

    y_true = df['label'].iloc[window+seq_len+1:len(df)].values
    return f1_score(y_true, preds, zero_division=0)

def main():
    print("="*60)
    print("步骤13：标准Transformer时序模型对比")
    print("="*60)
    stock = "A_sh.600036_招商银行"
    df = load_data(stock)
    features = FEATURE_COLS + ['llm_score']
    seq_len = 20
    window = 60
    print(f"股票: {stock}, 特征数: {len(features)}")
    f1_trans = rolling_evaluation_transformer(df, features, seq_len=seq_len, window=window)
    print(f"Transformer滚动窗口F1: {f1_trans:.4f}")

    # 加载之前的结果
    bench_df = pd.read_csv(os.path.join(RESULT_DIR, "sota_benchmark.csv"))
    print("\n完整对比:")
    print(bench_df)
    new_row = pd.DataFrame({'model': ['Transformer (standard)'], 'f1': [f1_trans]})
    updated_df = pd.concat([bench_df, new_row], ignore_index=True)
    updated_df.to_csv(os.path.join(RESULT_DIR, "sota_benchmark.csv"), index=False)
    print(f"更新结果保存至: {RESULT_DIR}/sota_benchmark.csv")
    print(f"Transformer F1: {f1_trans:.4f}")

if __name__ == "__main__":
    main()