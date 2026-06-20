#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step42_transformer_baseline.py
Transformer 基线模型（标准编码器）与 PatchTST 对比
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from step38_robust_multi_stock import load_individual_features, load_stock_data, robust_backtest

# 配置
DATA_DIR = "data/features"
FEATURE_SEL_DIR = "data/feature_selection"
RESULT_DIR = "data/results"
os.makedirs(RESULT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Transformer 超参数
SEQ_LEN = 20
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 128
DROPOUT = 0.2
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001
WINDOW = 360
STRIDE = 20

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, seq_len=SEQ_LEN, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x[:, -1, :]  # 取最后一个时间步
        return self.sigmoid(self.fc(x)).squeeze()

def prepare_sequences(df, features, seq_len=SEQ_LEN):
    X, y = [], []
    for i in range(seq_len, len(df)):
        X.append(df[features].iloc[i-seq_len:i].values)
        y.append(df['label'].iloc[i])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y

def train_transformer(X_train, y_train, X_val, y_val, input_dim, epochs=EPOCHS):
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = TransformerClassifier(input_dim=input_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    best_state = None
    for epoch in range(epochs):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                pred = model(Xb)
                val_loss += criterion(pred, yb).item()
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
    model.load_state_dict(best_state)
    return model

def rolling_predict_transformer(df, features, window=WINDOW, stride=STRIDE):
    dates, probs = [], []
    test_indices = list(range(window + SEQ_LEN, len(df) - 1, stride))
    input_dim = len(features)
    for test_idx in tqdm(test_indices, desc="Rolling Transformer"):
        train_end = test_idx - 1
        train_start = train_end - window
        train_df = df.iloc[train_start:train_end]
        split_idx = int(len(train_df) * 0.8)
        train_part = train_df.iloc[:split_idx]
        val_part = train_df.iloc[split_idx:]

        X_train, y_train = prepare_sequences(train_part, features, SEQ_LEN)
        X_val, y_val = prepare_sequences(val_part, features, SEQ_LEN)
        if len(X_train) == 0 or len(X_val) == 0:
            continue
        scaler = StandardScaler()
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
        X_val_flat = X_val.reshape(-1, X_val.shape[-1])
        X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape)
        X_test_seq = df[features].iloc[test_idx-SEQ_LEN:test_idx].values
        X_test_seq = X_test_seq.reshape(1, SEQ_LEN, -1)
        X_test_scaled = scaler.transform(X_test_seq.reshape(-1, X_test_seq.shape[-1])).reshape(X_test_seq.shape)

        model = train_transformer(X_train_scaled, y_train, X_val_scaled, y_val, input_dim, epochs=EPOCHS)
        model.eval()
        with torch.no_grad():
            prob = model(torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)).item()
        dates.append(df.index[test_idx])
        probs.append(prob)
    return pd.Series(probs, index=dates, name='prob')

def main():
    print("="*60)
    print("step42：Transformer 基线模型回测")
    print("="*60)
    stock = "A_sh.600036_招商银行"
    features = load_individual_features(stock)
    df = load_stock_data(stock, features)

    cache_path = os.path.join(RESULT_DIR, "transformer_probs_600036.csv")
    if os.path.exists(cache_path):
        trans_probs = pd.read_csv(cache_path, index_col=0, parse_dates=True)['prob']
        print("加载 Transformer 预测缓存")
    else:
        print("开始 Transformer 滚动预测（约 15-20 分钟）...")
        trans_probs = rolling_predict_transformer(df, features, window=WINDOW, stride=STRIDE)
        trans_probs.to_csv(cache_path)
        print("Transformer 预测完成")

    metrics = robust_backtest(trans_probs, df, long_thresh=0.55, short_thresh=0.45,
                              vol_target=0.15, stop_loss=0.01, transaction_cost=0.001)
    print("\n=== Transformer 策略绩效 ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # 保存结果
    pd.DataFrame([metrics]).to_csv(os.path.join(RESULT_DIR, "transformer_600036_metrics.csv"), index=False)
    print("step42 完成。")

if __name__ == "__main__":
    main()