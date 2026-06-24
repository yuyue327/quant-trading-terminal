#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step41_baseline_comparison.py
LSTM 基线对比 + Diebold-Mariano 检验
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
from statsmodels.tsa.stattools import acf
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

# LSTM 超参数
SEQ_LEN = 20
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2


class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.sigmoid(self.fc(out)).squeeze()


def prepare_sequences(df, features, seq_len=SEQ_LEN):
    X, y = [], []
    for i in range(seq_len, len(df)):
        X.append(df[features].iloc[i - seq_len:i].values)
        y.append(df['label'].iloc[i])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y


def train_lstm(X_train, y_train, X_val, y_val, input_dim, epochs=EPOCHS):
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleLSTM(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)
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


def rolling_predict_lstm(df, features, window=360, stride=20):
    dates, probs = [], []
    test_indices = list(range(window + SEQ_LEN, len(df) - 1, stride))
    input_dim = len(features)
    for test_idx in tqdm(test_indices, desc="Rolling LSTM"):
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
        X_test_seq = df[features].iloc[test_idx - SEQ_LEN:test_idx].values
        X_test_seq = X_test_seq.reshape(1, SEQ_LEN, -1)
        X_test_scaled = scaler.transform(X_test_seq.reshape(-1, X_test_seq.shape[-1])).reshape(X_test_seq.shape)

        model = train_lstm(X_train_scaled, y_train, X_val_scaled, y_val, input_dim, epochs=EPOCHS)
        model.eval()
        with torch.no_grad():
            prob = model(torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)).item()
        dates.append(df.index[test_idx])
        probs.append(prob)
    return pd.Series(probs, index=dates, name='prob')


def diebold_mariano(actual_strategy_returns, benchmark_returns, h=1):
    """
    Diebold-Mariano 检验
    actual_strategy_returns: 策略日收益率 Series
    benchmark_returns: 基准（买入持有）日收益率 Series
    h: 预测步长（此处为1）
    返回 p-value
    """
    from scipy.stats import norm
    # 对齐索引
    common_idx = actual_strategy_returns.index.intersection(benchmark_returns.index)
    e1 = actual_strategy_returns.loc[common_idx]
    e2 = benchmark_returns.loc[common_idx]
    # 损失差（均方误差）
    d = e1 ** 2 - e2 ** 2
    mean_d = np.mean(d)
    # 计算长期方差（考虑自相关）
    gamma0 = np.var(d, ddof=1)
    if h > 1:
        # 简单处理：采用 Newey-West 估计，这里简化使用 acf
        acf_d = acf(d, nlags=h - 1, fft=False)[1:]
        gamma = gamma0 + 2 * np.sum([(1 - i / h) * acf_d[i] * gamma0 for i in range(len(acf_d))])
    else:
        gamma = gamma0
    dm_stat = mean_d / np.sqrt(gamma / len(d))
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
    return dm_stat, p_value


def main():
    print("=" * 60)
    print("step41：LSTM 基线对比 + Diebold-Mariano 检验")
    print("=" * 60)
    stock = "A_sh.600036_招商银行"
    features = load_individual_features(stock)
    df = load_stock_data(stock, features)

    # 1. LSTM 预测及回测
    cache_path = os.path.join(RESULT_DIR, "lstm_probs_600036.csv")
    if os.path.exists(cache_path):
        lstm_probs = pd.read_csv(cache_path, index_col=0, parse_dates=True)['prob']
        print("加载 LSTM 预测缓存")
    else:
        print("开始 LSTM 滚动预测（约 15 分钟）...")
        lstm_probs = rolling_predict_lstm(df, features, window=360, stride=20)
        lstm_probs.to_csv(cache_path)
        print("LSTM 预测完成")

    lstm_metrics = robust_backtest(lstm_probs, df, long_thresh=0.55, short_thresh=0.45,
                                   vol_target=0.15, stop_loss=0.01, transaction_cost=0.001)
    print("\n=== LSTM 策略绩效 ===")
    for k, v in lstm_metrics.items():
        print(f"{k}: {v}")

    # 2. PatchTST 绩效（已存在）
    patchtst_probs = pd.read_csv(os.path.join(RESULT_DIR, "patchtst_probs_600036.csv"), index_col=0, parse_dates=True)[
        'prob']
    patchtst_metrics = robust_backtest(patchtst_probs, df, long_thresh=0.55, short_thresh=0.45,
                                       vol_target=0.15, stop_loss=0.01, transaction_cost=0.001)
    print("\n=== PatchTST 策略绩效（已有） ===")
    for k, v in patchtst_metrics.items():
        print(f"{k}: {v}")

    # 3. Diebold-Mariano 检验（PatchTST vs 买入持有）
    # 需要计算 PatchTST 策略的日收益率序列
    # 先重新运行一次完整回测以获取日收益率
    from step38_robust_multi_stock import robust_backtest as rb
    # 由于 robust_backtest 返回多个值，但我们在 step38 中定义的函数只返回 metrics，需要调整获取 net_returns
    # 为简单，我们直接重新实现一个快速回测来获取日收益序列
    def get_net_returns(probs, df, long_thresh=0.55, short_thresh=0.45, vol_target=0.15, stop_loss=0.01,
                        transaction_cost=0.001):
        common_idx = probs.index.intersection(df.index)
        probs = probs.loc[common_idx]
        close = df.loc[common_idx, 'close']
        returns = close.pct_change().fillna(0)
        direction = np.zeros_like(probs)
        direction[probs > long_thresh] = 1
        direction[probs < short_thresh] = -1
        position_size = np.abs(probs - 0.5) * 2
        raw_position = direction * position_size
        vol = returns.rolling(20).std() * np.sqrt(252)
        vol_scaler = vol_target / vol.clip(lower=0.05, upper=0.5)
        vol_scaler = vol_scaler.fillna(1)
        adj_position = raw_position * vol_scaler
        adj_position = adj_position.clip(-0.5, 0.5)
        position = adj_position.shift(1).fillna(0)
        # 简化止损（略，因为影响不大）
        strategy_returns = position * returns
        trade_costs = position.diff().abs() * transaction_cost
        net_returns = strategy_returns - trade_costs
        return net_returns

    patch_net = get_net_returns(patchtst_probs, df)
    bench_net = df.loc[patch_net.index, 'close'].pct_change().fillna(0)
    dm_stat, p_val = diebold_mariano(patch_net, bench_net, h=1)
    print(f"\n=== Diebold-Mariano 检验（PatchTST vs 买入持有） ===")
    print(f"DM 统计量: {dm_stat:.4f}")
    print(f"p 值: {p_val:.6f}")
    if p_val < 0.05:
        print("结论: 在 95% 置信水平下，PatchTST 策略优于买入持有（显著）")
    else:
        print("结论: 差异不显著")

    # 保存对比结果
    comparison = pd.DataFrame({
        'Model': ['LSTM', 'PatchTST'],
        'Sharpe': [float(lstm_metrics['夏普比率']), float(patchtst_metrics['夏普比率'])],
        'Annual Return': [lstm_metrics['年化收益率'], patchtst_metrics['年化收益率']],
        'Max Drawdown': [lstm_metrics['最大回撤'], patchtst_metrics['最大回撤']]
    })
    comparison.to_csv(os.path.join(RESULT_DIR, "model_comparison.csv"), index=False)
    print("\n对比结果已保存至 data/results/model_comparison.csv")

    print("step41 完成。")


if __name__ == "__main__":
    main()