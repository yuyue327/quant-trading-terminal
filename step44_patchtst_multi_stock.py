#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step44_patchtst_multi_stock.py
对招商银行、五粮液、平安银行运行 PatchTST 滚动预测并回测
（复用 step40 的逻辑，仅更换股票）
"""
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from step39_patchtst_model import PatchTST
from step38_robust_multi_stock import load_individual_features, load_stock_data, robust_backtest

# 配置
STOCKS = {
    "招商银行": "A_sh.600036_招商银行",
    "五粮液": "A_sz.000858_五粮液",
    "平安银行": "A_sz.000001_平安银行"
}
RESULT_DIR = "data/results"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN = 20
PATCH_LEN = 5
STRIDE = 1
D_MODEL = 64
N_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001
WINDOW = 360
TEST_STRIDE = 20

def prepare_sequences(df, features, seq_len=SEQ_LEN):
    X, y = [], []
    for i in range(seq_len, len(df)):
        X.append(df[features].iloc[i-seq_len:i].values)
        y.append(df['label'].iloc[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def train_patchtst(X_train, y_train, X_val, y_val, input_dim):
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = PatchTST(input_dim=input_dim, seq_len=SEQ_LEN, patch_len=PATCH_LEN, stride=STRIDE,
                     d_model=D_MODEL, n_heads=NUM_LAYERS, num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()
    best_loss = float('inf')
    best_state = None
    for epoch in range(EPOCHS):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(Xb), yb).item()
        val_loss /= len(val_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict().copy()
    model.load_state_dict(best_state)
    return model

def rolling_predict(stock_name, features, df):
    dates, probs = [], []
    test_indices = list(range(WINDOW + SEQ_LEN, len(df) - 1, TEST_STRIDE))
    input_dim = len(features)
    for test_idx in tqdm(test_indices, desc=stock_name):
        train_end = test_idx - 1
        train_start = train_end - WINDOW
        train_df = df.iloc[train_start:train_end]
        split = int(0.8 * len(train_df))
        train_part = train_df.iloc[:split]
        val_part = train_df.iloc[split:]
        X_train, y_train = prepare_sequences(train_part, features)
        X_val, y_val = prepare_sequences(val_part, features)
        if len(X_train)==0 or len(X_val)==0:
            continue
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        X_test = df[features].iloc[test_idx-SEQ_LEN:test_idx].values.reshape(1,SEQ_LEN,-1)
        X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        model = train_patchtst(X_train_scaled, y_train, X_val_scaled, y_val, input_dim)
        model.eval()
        with torch.no_grad():
            prob = model(torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)).item()
        dates.append(df.index[test_idx])
        probs.append(prob)
    return pd.Series(probs, index=dates, name='prob')

def main():
    print("="*60)
    print("step44：多股票 PatchTST 验证")
    print("="*60)
    results = []
    for name, stock in STOCKS.items():
        print(f"\n处理 {name}...")
        features = load_individual_features(stock)
        df = load_stock_data(stock, features)
        cache = os.path.join(RESULT_DIR, f"patchtst_probs_{stock}.csv")
        if os.path.exists(cache):
            probs = pd.read_csv(cache, index_col=0, parse_dates=True)['prob']
        else:
            probs = rolling_predict(name, features, df)
            probs.to_csv(cache)
        metrics = robust_backtest(probs, df, long_thresh=0.55, short_thresh=0.45,
                                  vol_target=0.15, stop_loss=0.01, transaction_cost=0.001)
        metrics['股票'] = name
        results.append(metrics)
        print(f"  夏普: {metrics['夏普比率']}, 年化: {metrics['年化收益率']}")
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(RESULT_DIR, "patchtst_multi_stock.csv"), index=False)
    print("\n=== PatchTST 多股票绩效 ===")
    print(df_res[['股票', '夏普比率', '年化收益率', '最大回撤']].to_string(index=False))
    print(f"\n平均夏普: {df_res['夏普比率'].astype(float).mean():.2f}")
    print("step44 完成。")

if __name__ == "__main__":
    main()