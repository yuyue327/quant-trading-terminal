#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step60_strong_baselines_sota.py
补充强基线：iTransformer, TimesNet, TFT, DLinear, TSMixer
✅ 覆盖全部 18 只股票，生成汇总表格
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from step38_robust_multi_stock import load_individual_features, load_stock_data, robust_backtest

# ===== 配置 =====
RESULT_DIR = "data/results"
os.makedirs(RESULT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

SEQ_LEN = 20
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001
WINDOW = 360
TEST_STRIDE = 20

# ✅ 全部 18 只股票
ALL_STOCKS = [
    # A股 15 只
    "A_sh.600036_招商银行",
    "A_sz.000858_五粮液",
    "A_sz.000001_平安银行",
    "A_sh.600030_中信证券",
    "A_sh.600519_贵州茅台",
    "A_sz.300750_宁德时代",
    "A_sh.600887_伊利股份",
    "A_sh.601012_隆基绿能",
    "A_sh.601688_华泰证券",
    "A_sz.000333_美的集团",
    "A_sz.000568_泸州老窖",
    "A_sz.000651_格力电器",
    "A_sz.002142_宁波银行",
    "A_sz.002594_比亚迪",
    "A_sz.300059_东方财富",
    # 美股 3 只
    "US_AAPL_AAPL",
    "US_MSFT_MSFT",
    "US_NVDA_NVDA",
]

# ===== 模型定义 =====
class iTransformer(nn.Module):
    def __init__(self, input_dim, seq_len, d_model=64, n_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.feature_proj = nn.Linear(seq_len, d_model)
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model*4, dropout=dropout, activation='gelu', batch_first=True)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model * input_dim, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, seq_len, input_dim = x.shape
        x = x.permute(0, 2, 1)
        h = self.feature_proj(x)
        for layer in self.encoder_layers:
            h = layer(h)
        h = h.reshape(batch, -1)
        out = self.head(self.dropout(h))
        return torch.sigmoid(out[:, 0]), out[:, 1]

class TimesNet(nn.Module):
    def __init__(self, input_dim, seq_len, d_model=64, num_blocks=2, kernel_size=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2),
                nn.GELU()
            ) for _ in range(num_blocks)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(d_model, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.input_proj(x)
        h = h.permute(0, 2, 1)
        for block in self.blocks:
            h = block(h) + h
        h = self.pool(h).squeeze(-1)
        out = self.head(self.dropout(h))
        return torch.sigmoid(out[:, 0]), out[:, 1]

class TFT(nn.Module):
    def __init__(self, input_dim, seq_len, d_model=64, n_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers, batch_first=True, dropout=dropout)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.head = nn.Linear(d_model, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.input_proj(x)
        h, _ = self.lstm(h)
        h, _ = self.self_attn(h, h, h)
        h = h.mean(dim=1)
        out = self.head(self.dropout(h))
        return torch.sigmoid(out[:, 0]), out[:, 1]

class DLinear(nn.Module):
    def __init__(self, input_dim, seq_len, d_model=64, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.avg_pool = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.linear_trend = nn.Linear(seq_len, d_model)
        self.linear_season = nn.Linear(seq_len, d_model)
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.head = nn.Linear(d_model, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, seq_len, input_dim = x.shape
        x = x.permute(0, 2, 1)
        trend = self.avg_pool(x).mean(dim=1)
        h_trend = self.linear_trend(trend)
        season = x - trend.unsqueeze(1).expand(-1, input_dim, -1)
        season = season.mean(dim=1)
        h_season = self.linear_season(season)
        h = torch.cat([h_trend, h_season], dim=1)
        h = self.fusion(self.dropout(h))
        out = self.head(self.dropout(h))
        return torch.sigmoid(out[:, 0]), out[:, 1]

# ===== 工具函数 =====
def prepare_sequences(df, features, seq_len=SEQ_LEN):
    X, y = [], []
    for i in range(seq_len, len(df)):
        X.append(df[features].iloc[i-seq_len:i].values)
        y.append(df['label'].iloc[i])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y

def rolling_predict_sota(df, features, model_class, stock_name, **kwargs):
    test_indices = list(range(WINDOW + SEQ_LEN, len(df) - 1, TEST_STRIDE))
    if not test_indices:
        return pd.Series(dtype=float)
    input_dim = len(features)
    dates, probs = [], []
    for test_idx in tqdm(test_indices, desc=f"{model_class.__name__}", leave=False):
        train_end = test_idx - 1
        train_start = train_end - WINDOW
        if train_start < 0: continue
        train_df = df.iloc[train_start:train_end]
        split_idx = int(len(train_df) * 0.8)
        if split_idx < SEQ_LEN + 1: continue
        train_part = train_df.iloc[:split_idx]
        val_part = train_df.iloc[split_idx:]
        X_train, y_train = prepare_sequences(train_part, features)
        X_val, y_val = prepare_sequences(val_part, features)
        if len(X_train) == 0 or len(X_val) == 0: continue
        scaler = StandardScaler()
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
        X_val_flat = X_val.reshape(-1, X_val.shape[-1])
        X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape)
        model = model_class(input_dim=input_dim, seq_len=SEQ_LEN, **kwargs).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.BCEWithLogitsLoss()
        train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        best_val_loss = float('inf')
        best_state = None
        for epoch in range(EPOCHS):
            model.train()
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                pred, _ = model(Xb)
                loss = criterion(pred.squeeze(), yb)
                loss.backward()
                optimizer.step()
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                    pred, _ = model(Xb)
                    val_loss += criterion(pred.squeeze(), yb).item()
            val_loss /= len(val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()
        X_test_seq = df[features].iloc[test_idx - SEQ_LEN:test_idx].values
        X_test_seq = X_test_seq.reshape(1, SEQ_LEN, -1)
        X_test_scaled = scaler.transform(X_test_seq.reshape(-1, X_test_seq.shape[-1])).reshape(X_test_seq.shape)
        with torch.no_grad():
            pred, _ = model(torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE))
            prob = pred.squeeze().item()
        dates.append(df.index[test_idx])
        probs.append(prob)
    return pd.Series(probs, index=dates, name='prob')

def run_baseline_for_stock(stock, model_class, model_name, **kwargs):
    try:
        features = load_individual_features(stock)
        df = load_stock_data(stock, features)
        cache_path = os.path.join(RESULT_DIR, f"{model_name}_probs_{stock}.csv")
        if os.path.exists(cache_path):
            probs = pd.read_csv(cache_path, index_col=0, parse_dates=True)['prob']
        else:
            print(f"  运行 {model_name} ...")
            probs = rolling_predict_sota(df, features, model_class, stock, **kwargs)
            probs.to_csv(cache_path)
        metrics = robust_backtest(probs, df, long_thresh=0.55, short_thresh=0.45,
                                  vol_target=0.15, stop_loss=0.01, transaction_cost=0.001)
        sharpe = float(metrics['夏普比率'])
        return sharpe
    except Exception as e:
        print(f"  ❌ {stock} {model_name} 失败: {e}")
        return None

def main():
    print("=" * 60)
    print("step60：补充强基线SOTA模型 (全覆盖 18 只股票)")
    print("=" * 60)

    models = [
        ('iTransformer', iTransformer, {}),
        ('TimesNet', TimesNet, {'num_blocks': 2, 'kernel_size': 3}),
        ('TFT', TFT, {'num_layers': 2}),
        ('DLinear', DLinear, {}),
    ]

    all_results = []

    for stock in tqdm(ALL_STOCKS, desc="股票进度"):
        print(f"\n处理 {stock}...")
        row = {'stock': stock}
        for name, model_class, kwargs in models:
            sharpe = run_baseline_for_stock(stock, model_class, name, **kwargs)
            row[name] = sharpe
        all_results.append(row)

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(RESULT_DIR, "sota_baseline_full_results.csv"), index=False)

    # 计算平均值
    avg_row = {'stock': 'AVERAGE'}
    for name, _, _ in models:
        avg_row[name] = df_results[name].mean()
    df_results = pd.concat([df_results, pd.DataFrame([avg_row])], ignore_index=True)
    df_results.to_csv(os.path.join(RESULT_DIR, "sota_baseline_full_results_with_avg.csv"), index=False)

    print("\n" + "=" * 60)
    print("📊 SOTA基线对比结果（全部 18 只股票）")
    print("=" * 60)
    print(df_results.to_string(index=False))
    print("\n✅ step60 完成！")

if __name__ == "__main__":
    main()