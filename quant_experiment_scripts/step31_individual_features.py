#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step31_individual_features.py
使用股票个性化特征（而非全市场交集）重新训练集成模型并回测
包含完整的模型定义，修复卷积尺寸不匹配问题
"""

import os
import json
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

# ==================== 模型定义（修复版 TemporalConvBlock） ====================
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
        self.kernel_size = kernel_size
        self.dilation = dilation
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
        # 修复尺寸不匹配：裁剪或填充至与 residual 相同长度
        if out.size(-1) > residual.size(-1):
            out = out[:, :, :residual.size(-1)]
        elif out.size(-1) < residual.size(-1):
            pad = residual.size(-1) - out.size(-1)
            out = F.pad(out, (0, pad))
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
FEATURE_SEL_DIR = "data/feature_selection"
RESULT_DIR = "data/results"
os.makedirs(RESULT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

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

def load_individual_features(stock_name):
    """从 step27 输出的 json 中读取该股票的个性化特征列表"""
    json_path = os.path.join(FEATURE_SEL_DIR, "selected_features_per_stock.json")
    with open(json_path, 'r') as f:
        all_selected = json.load(f)
    if stock_name not in all_selected:
        raise KeyError(f"股票 {stock_name} 未在特征选择结果中找到。可用股票: {list(all_selected.keys())[:5]}...")
    features = all_selected[stock_name]
    features = list(dict.fromkeys(features))
    return features

def load_stock_data(stock_name, features):
    """加载股票数据，确保所需的特征列都存在"""
    file_path = os.path.join(DATA_DIR, f"{stock_name}.parquet")
    df = pd.read_parquet(file_path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    for col in features:
        if col not in df.columns:
            print(f"警告: 特征 {col} 不存在于 {stock_name}，用 0 填充")
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

def train_one_model(X_train, y_train, X_val, y_val, seed, input_dim, epochs=EPOCHS):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = CTANModel(n_features=input_dim, seq_len=SEQ_LEN,
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

def rolling_predict_ensemble_individual(stock_name, features, seq_len=SEQ_LEN, window=WINDOW, stride=STRIDE, n_ensemble=N_ENSEMBLE):
    """滚动窗口集成预测，使用个性化特征"""
    df = load_stock_data(stock_name, features)
    all_dates = []
    all_probs = []
    all_uncertainties = []
    test_indices = list(range(window + seq_len, len(df) - 1, stride))
    print(f"滚动窗口总数: {len(test_indices)}")
    input_dim = len(features)
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
            model = train_one_model(X_train_scaled, y_train, X_val_scaled, y_val, seed=seed, input_dim=input_dim, epochs=EPOCHS)
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
    result_df = pd.DataFrame({'prob': all_probs, 'uncertainty': all_uncertainties}, index=all_dates)
    return result_df

def backtest_with_uncertainty(probs_df, df, trend_window=20, max_position=0.5,
                              atr_stop_mult=1.5, volatility_target=0.18,
                              uncertainty_threshold=0.2, transaction_cost=0.001):
    """回测函数（与 step30 一致）"""
    common_idx = probs_df.index.intersection(df.index)
    probs = probs_df.loc[common_idx, 'prob']
    if 'uncertainty' in probs_df.columns:
        uncertainty = probs_df.loc[common_idx, 'uncertainty']
    else:
        uncertainty = probs.rolling(20, min_periods=5).std().fillna(0.1).clip(0.05, 0.3)
    df_aligned = df.loc[common_idx]
    close = df_aligned['close']

    ma = close.rolling(trend_window).mean()
    trend_dir = np.where(close > ma, 1, 0)
    signal_strength = np.abs(probs - 0.5) * 2
    uncertainty_penalty = 1 / (1 + uncertainty / uncertainty_threshold)
    uncertainty_penalty = np.clip(uncertainty_penalty, 0.3, 1.0)
    returns = close.pct_change().fillna(0)
    vol = returns.rolling(20).std() * np.sqrt(252)
    vol_scaler = volatility_target / vol.clip(lower=0.05, upper=0.5)
    vol_scaler = vol_scaler.fillna(1)
    raw_position = trend_dir * signal_strength * vol_scaler * uncertainty_penalty
    position = np.clip(raw_position, 0, max_position)
    position = pd.Series(position, index=probs.index).shift(1).fillna(0)

    # ATR
    high = df_aligned['high'] if 'high' in df_aligned.columns else df_aligned['close'] * 1.02
    low = df_aligned['low'] if 'low' in df_aligned.columns else df_aligned['close'] * 0.98
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().fillna(method='bfill').fillna(0)

    final_positions = []
    in_position = False
    stop_price = 0
    for idx in position.index:
        pos = position.loc[idx]
        price = close.loc[idx]
        atr_val = atr.loc[idx]
        if not in_position and pos > 0:
            in_position = True
            stop_price = price - atr_stop_mult * atr_val
            final_positions.append(pos)
        elif in_position:
            if price < stop_price:
                in_position = False
                final_positions.append(0)
            else:
                new_stop = price - atr_stop_mult * atr_val
                if new_stop > stop_price:
                    stop_price = new_stop
                if pos == 0:
                    in_position = False
                    final_positions.append(0)
                else:
                    final_positions.append(pos)
        else:
            final_positions.append(0)
    final_positions = pd.Series(final_positions, index=position.index)

    daily_returns = close.pct_change().fillna(0)
    strategy_returns = final_positions * daily_returns
    trade_costs = final_positions.diff().abs() * transaction_cost
    net_returns = strategy_returns - trade_costs
    nav = (1 + net_returns).cumprod() * 1e6
    bench_nav = (1 + daily_returns).cumprod() * 1e6

    total_ret = nav.iloc[-1] / 1e6 - 1
    bench_ret = bench_nav.iloc[-1] / 1e6 - 1
    trading_days = len(nav)
    annual_ret = (1 + total_ret) ** (252 / trading_days) - 1 if total_ret > -1 else np.nan
    bench_annual = (1 + bench_ret) ** (252 / trading_days) - 1 if bench_ret > -1 else np.nan
    excess_ret = net_returns - 0.03 / 252
    sharpe = np.sqrt(252) * excess_ret.mean() / excess_ret.std() if excess_ret.std() != 0 else np.nan
    max_dd = (nav / nav.cummax() - 1).min()
    win_rate = (net_returns[net_returns != 0] > 0).mean() if (net_returns != 0).any() else 0
    trade_count = (final_positions.diff().abs() > 0).sum()
    metrics = {
        '总收益率': f"{total_ret:.2%}",
        '基准收益率': f"{bench_ret:.2%}",
        '年化收益率': f"{annual_ret:.2%}",
        '基准年化': f"{bench_annual:.2%}",
        '夏普比率': f"{sharpe:.2f}",
        '最大回撤': f"{max_dd:.2%}",
        '胜率': f"{win_rate:.2%}",
        '交易次数': int(trade_count)
    }
    return nav, bench_nav, net_returns, metrics

def main():
    print("="*60)
    print("step31：使用个性化特征重新训练集成模型并回测")
    print("="*60)
    stock = "A_sh.600036_招商银行"
    print(f"加载股票 {stock} 的个性化特征...")
    try:
        features = load_individual_features(stock)
        print(f"个性化特征数量: {len(features)}")
        print(f"特征列表: {features}")
    except Exception as e:
        print(f"错误: {e}")
        return

    cache_path = os.path.join(RESULT_DIR, "ensemble_probs_individual.csv")
    if os.path.exists(cache_path):
        print(f"发现已有缓存 {cache_path}，直接加载并回测。如需重新训练请删除该文件。")
        probs_df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    else:
        print("开始滚动窗口集成预测（可能需要较长时间）...")
        probs_df = rolling_predict_ensemble_individual(stock, features)
        probs_df.to_csv(cache_path)
        print(f"预测结果已保存至 {cache_path}")

    df = load_stock_data(stock, features)
    if 'high' not in df.columns:
        df['high'] = df['close'] * 1.02
    if 'low' not in df.columns:
        df['low'] = df['close'] * 0.98

    print("执行回测...")
    nav, bench_nav, net_returns, metrics = backtest_with_uncertainty(probs_df, df)
    print("\n=== 策略绩效指标（个性化特征） ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.figure(figsize=(12, 6))
    plt.plot(nav.index, nav, label='策略净值 (个性化特征)', linewidth=2)
    plt.plot(bench_nav.index, bench_nav, label='买入持有', linewidth=1.5, linestyle='--')
    plt.title('策略 vs 基准净值曲线 (个性化特征)')
    plt.xlabel('日期')
    plt.ylabel('净值 (元)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs("data/figures", exist_ok=True)
    plt.savefig(os.path.join("data/figures", "strategy_nav_individual.png"), dpi=300)
    plt.close()
    print("净值曲线已保存至 data/figures/strategy_nav_individual.png")

    pd.DataFrame([metrics]).to_csv(os.path.join(RESULT_DIR, "strategy_metrics_individual.csv"), index=False)
    print("step31 完成。")

if __name__ == "__main__":
    main()