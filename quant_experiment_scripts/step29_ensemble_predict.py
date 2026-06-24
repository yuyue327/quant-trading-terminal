#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step29_ensemble_predict.py
滚动窗口训练 TCN + 因果注意力模型，5 个随机种子集成，输出预测概率及不确定性
内嵌模型定义，不依赖外部文件
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import math
import warnings
warnings.filterwarnings('ignore')

# ==================== 模型定义（与 step28 保持一致） ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class CausalAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        assert d_model % n_heads == 0
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is None:
            seq_len = query.size(1)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=attn_scores.device), diagonal=1).bool()
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        else:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_linear(context)

class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.GELU()
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out + residual)
        return out

class CTANModel(nn.Module):
    def __init__(self, n_features, seq_len=20, d_model=64, n_heads=4, num_conv_blocks=2, dropout=0.2):
        super().__init__()
        self.input_fc = nn.Linear(n_features, d_model)
        self.conv_blocks = nn.ModuleList()
        for i in range(num_conv_blocks):
            dilation = 2 ** i
            self.conv_blocks.append(TemporalConvBlock(d_model, d_model, kernel_size=3, dilation=dilation, dropout=dropout))
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        self.self_attn = CausalAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.input_fc(x)
        x_conv = x.transpose(1, 2)
        for conv in self.conv_blocks:
            x_conv = conv(x_conv)
        x = x_conv.transpose(1, 2)
        x = self.pos_encoder(x)
        attn_out = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        pooled = x.mean(dim=1)
        logits = self.classifier(pooled)
        return torch.sigmoid(logits).squeeze(-1)

# ==================== 配置 ====================
DATA_DIR = "data/features"
RESULT_DIR = "data/results"
os.makedirs(RESULT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# 统一特征集（与 step28 输出一致）
UNIFIED_FEATURES = [
    'volatility_5', 'volume_ratio', 'llm_score', 'pct_change', 'high_low_pct',
    'close_position', 'MACD_hist', 'volatility_20', 'BB_pct', 'RSI',
    'ATR', 'BB_width', 'MACD', 'MACD_signal', 'MA5'
]

# 超参数
SEQ_LEN = 20
WINDOW = 720
STRIDE = 20
N_ENSEMBLE = 5
EPOCHS = 30
BATCH_SIZE = 32
LR = 0.001
D_MODEL = 64
N_HEADS = 4
NUM_CONV_BLOCKS = 2
DROPOUT = 0.2

def load_stock_data(stock_name):
    file_path = os.path.join(DATA_DIR, f"{stock_name}.parquet")
    df = pd.read_parquet(file_path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    for col in UNIFIED_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    if 'label' not in df.columns:
        raise ValueError(f"{stock_name} 缺少 label 列")
    return df

def prepare_sequences(df, features, seq_len=SEQ_LEN):
    X, y = [], []
    for i in range(seq_len, len(df)):
        X.append(df[features].iloc[i-seq_len:i].values)
        y.append(df['label'].iloc[i])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y

def train_one_model(X_train, y_train, X_val, y_val, seed, epochs=EPOCHS):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = CTANModel(n_features=len(UNIFIED_FEATURES), seq_len=SEQ_LEN,
                      d_model=D_MODEL, n_heads=N_HEADS,
                      num_conv_blocks=NUM_CONV_BLOCKS, dropout=DROPOUT).to(DEVICE)
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

def rolling_predict_ensemble(stock_name, seq_len=SEQ_LEN, window=WINDOW, stride=STRIDE, n_ensemble=N_ENSEMBLE):
    df = load_stock_data(stock_name)
    features = UNIFIED_FEATURES
    all_dates = []
    all_probs = []
    all_uncertainties = []
    test_indices = list(range(window + seq_len, len(df) - 1, stride))
    print(f"滚动窗口总数: {len(test_indices)}")
    for test_idx in tqdm(test_indices, desc="Rolling windows"):
        train_end = test_idx - 1
        train_start = train_end - window
        train_df = df.iloc[train_start:train_end]
        split_idx = int(len(train_df) * 0.8)
        train_df_part = train_df.iloc[:split_idx]
        val_df_part = train_df.iloc[split_idx:]
        X_train, y_train = prepare_sequences(train_df_part, features, seq_len)
        X_val, y_val = prepare_sequences(val_df_part, features, seq_len)
        if len(X_train) == 0 or len(X_val) == 0:
            continue
        scaler = StandardScaler()
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
        X_val_flat = X_val.reshape(-1, X_val.shape[-1])
        X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape)
        X_test_seq = df[features].iloc[test_idx-seq_len:test_idx].values
        X_test_seq = X_test_seq.reshape(1, seq_len, -1)
        X_test_scaled = scaler.transform(X_test_seq.reshape(-1, X_test_seq.shape[-1])).reshape(X_test_seq.shape)
        model_probs = []
        for seed in range(n_ensemble):
            model = train_one_model(X_train_scaled, y_train, X_val_scaled, y_val, seed=seed, epochs=EPOCHS)
            model.eval()
            with torch.no_grad():
                prob = model(torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)).item()
            model_probs.append(prob)
        mean_prob = np.mean(model_probs)
        std_prob = np.std(model_probs)
        pred_date = df.index[test_idx]
        all_dates.append(pred_date)
        all_probs.append(mean_prob)
        all_uncertainties.append(std_prob)
    probs_series = pd.Series(all_probs, index=all_dates, name='prob')
    uncertainty_series = pd.Series(all_uncertainties, index=all_dates, name='uncertainty')
    result_df = pd.DataFrame({'prob': all_probs, 'uncertainty': all_uncertainties}, index=all_dates)
    return probs_series, uncertainty_series, result_df

def main():
    print("="*60)
    print("step29：滚动窗口集成预测（TCN + 因果注意力）")
    print("="*60)
    stock = "A_sh.600036_招商银行"
    print(f"处理股票: {stock}")
    cache_path = os.path.join(RESULT_DIR, "ensemble_probs.csv")
    if os.path.exists(cache_path):
        print(f"缓存文件已存在: {cache_path}")
        print("如需重新计算，请删除该文件后重新运行。")
        df_cache = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        probs = df_cache['prob']
        uncertainty = df_cache['uncertainty'] if 'uncertainty' in df_cache.columns else None
        print(f"从缓存加载，共 {len(probs)} 个预测点")
    else:
        print("开始滚动窗口集成预测（可能需要较长时间，约 30-60 分钟）...")
        probs, uncertainty, result_df = rolling_predict_ensemble(stock)
        result_df.to_csv(cache_path)
        print(f"预测结果已保存至 {cache_path}")
    print(f"\n预测概率统计: mean={probs.mean():.4f}, std={probs.std():.4f}")
    if uncertainty is not None:
        print(f"平均不确定性（标准差）: {uncertainty.mean():.4f}")
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.figure(figsize=(10, 4))
    plt.hist(probs, bins=50, alpha=0.7, color='blue')
    plt.axvline(0.5, color='red', linestyle='--')
    plt.title('集成模型预测概率分布')
    plt.xlabel('上涨概率')
    plt.ylabel('频次')
    os.makedirs("data/figures", exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join("data/figures", "ensemble_prob_dist.png"), dpi=150)
    plt.close()
    print("step29 完成。")

if __name__ == "__main__":
    main()