#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤15：策略回测与绩效分析（最终修复版）
确保索引为日期类型，对齐信号和价格数据。
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from step4_causal_attribution import load_data, FEATURE_COLS

# 配置
RESULT_DIR = "data/results"
FIGURE_DIR = "data/figures"
os.makedirs(RESULT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------- 模型定义 ----------------------------
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


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.sigmoid(self.fc(x)).squeeze()


# ---------------------------- 训练函数 ----------------------------
def train_lstm_quick(X_train, y_train, X_val, y_val, input_dim, epochs=30, batch_size=32, lr=0.001):
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = SimpleLSTM(input_dim=input_dim).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
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
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict().copy()
    model.load_state_dict(best_state)
    return model


def train_transformer_quick(X_train, y_train, X_val, y_val, input_dim, epochs=20, batch_size=32, lr=0.001):
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = TransformerEncoder(input_dim=input_dim).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
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
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict().copy()
    model.load_state_dict(best_state)
    return model


# ---------------------------- 滚动预测 ----------------------------
def rolling_predict(df, features, seq_len=20, window=60):
    dates = []
    probs = []
    total_points = len(df) - (window + seq_len + 1)
    for idx, i in enumerate(range(window + seq_len, len(df) - 1)):
        train_end = i - seq_len
        train_start = train_end - window
        train_val_df = df.iloc[train_start:train_end + seq_len]

        X_test_seq = df[features].iloc[i - seq_len:i].values.reshape(1, seq_len, -1)

        # 准备序列数据
        X_all, y_all = [], []
        for j in range(seq_len, len(train_val_df)):
            X_all.append(train_val_df[features].iloc[j - seq_len:j].values)
            y_all.append(train_val_df['label'].iloc[j])
        X_all = np.array(X_all, dtype=np.float32)
        y_all = np.array(y_all, dtype=np.float32)
        if len(X_all) == 0:
            continue
        split = int(0.8 * len(X_all))
        X_train_seq, X_val_seq = X_all[:split], X_all[split:]
        y_train_seq, y_val_seq = y_all[:split], y_all[split:]

        scaler = StandardScaler()
        X_train_flat = X_train_seq.reshape(-1, X_train_seq.shape[-1])
        X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train_seq.shape)
        X_val_scaled = scaler.transform(X_val_seq.reshape(-1, X_val_seq.shape[-1])).reshape(X_val_seq.shape)
        X_test_scaled = scaler.transform(X_test_seq.reshape(-1, X_test_seq.shape[-1])).reshape(X_test_seq.shape)

        lstm_model = train_lstm_quick(X_train_scaled, y_train_seq, X_val_scaled, y_val_seq, input_dim=len(features),
                                      epochs=30)
        trans_model = train_transformer_quick(X_train_scaled, y_train_seq, X_val_scaled, y_val_seq,
                                              input_dim=len(features), epochs=20)

        train_df = df.iloc[train_start:train_end]
        X_train_rf = train_df[features].values
        y_train_rf = train_df['label'].values
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_rf, y_train_rf)
        X_test_rf = df[features].iloc[i].values.reshape(1, -1)
        rf_prob = rf.predict_proba(X_test_rf)[0, 1]

        lstm_model.eval()
        trans_model.eval()
        with torch.no_grad():
            lstm_prob = lstm_model(torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)).item()
            trans_prob = trans_model(torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)).item()

        f1_rf, f1_lstm, f1_trans = 0.4527, 0.4451, 0.4481
        w_sum = f1_rf + f1_lstm + f1_trans
        weights = np.array([f1_rf, f1_lstm, f1_trans]) / w_sum
        ensemble_prob = weights[0] * rf_prob + weights[1] * lstm_prob + weights[2] * trans_prob

        dates.append(df.index[i + 1])
        probs.append(ensemble_prob)

        if (idx + 1) % 100 == 0:
            print(f"已生成 {idx + 1}/{total_points} 个信号")

    return pd.Series(probs, index=dates, name='signal_prob')


# ---------------------------- 回测函数 ----------------------------
def backtest(signal_probs, close_prices, initial_capital=1e6, transaction_cost=0.001):
    # 对齐并去除缺失值
    combined = pd.DataFrame({'signal_prob': signal_probs, 'close': close_prices}).dropna().sort_index()
    if combined.empty:
        raise ValueError("没有可用的对齐数据，请检查日期索引。")

    # 确保数值类型
    combined = combined.astype(float)

    # 生成信号和收益率
    signals = (combined['signal_prob'] >= 0.5).astype(int)
    returns = combined['close'].pct_change()
    # 将第一个 NaN 设为 0
    returns.iloc[0] = 0.0
    returns = returns.fillna(0.0)

    # 持仓（次日生效）
    positions = signals.shift(1)
    positions.iloc[0] = 0.0  # 第一天无持仓
    positions = positions.fillna(0.0)

    # 策略日收益（不含交易成本）
    daily_returns = positions * returns
    # 交易成本：仅当持仓变化时收取
    trade_cost = positions.diff().abs() * transaction_cost
    trade_cost.iloc[0] = 0.0
    daily_returns_net = daily_returns - trade_cost
    daily_returns_net = daily_returns_net.fillna(0.0)

    # 累计净值
    nav = (1 + daily_returns_net).cumprod()
    nav = nav / nav.iloc[0] * initial_capital
    bench_nav = (1 + returns).cumprod()
    bench_nav = bench_nav / bench_nav.iloc[0] * initial_capital

    # 最终指标（确保没有 NaN）
    total_return = nav.iloc[-1] / initial_capital - 1 if not pd.isna(nav.iloc[-1]) else np.nan
    bench_return = bench_nav.iloc[-1] / initial_capital - 1 if not pd.isna(bench_nav.iloc[-1]) else np.nan

    trading_days = len(nav)
    if not pd.isna(total_return) and total_return > -1:
        annual_return = (1 + total_return) ** (252 / trading_days) - 1
    else:
        annual_return = np.nan

    if not pd.isna(bench_return) and bench_return > -1:
        bench_annual = (1 + bench_return) ** (252 / trading_days) - 1
    else:
        bench_annual = np.nan

    excess_returns = daily_returns_net - 0.03 / 252
    sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else np.nan

    peak = nav.cummax()
    drawdown = (nav - peak) / peak
    max_drawdown = drawdown.min() if not drawdown.isnull().all() else np.nan

    trade_days = daily_returns_net != 0
    win_rate = (daily_returns_net[trade_days] > 0).sum() / trade_days.sum() if trade_days.sum() > 0 else np.nan

    trade_times = positions.diff().abs().sum()

    metrics = {
        '总收益率': f"{total_return:.2%}" if not pd.isna(total_return) else "nan%",
        '基准收益率': f"{bench_return:.2%}" if not pd.isna(bench_return) else "nan%",
        '年化收益率': f"{annual_return:.2%}" if not pd.isna(annual_return) else "nan%",
        '基准年化': f"{bench_annual:.2%}" if not pd.isna(bench_annual) else "nan%",
        '夏普比率': f"{sharpe:.2f}" if not pd.isna(sharpe) else "nan",
        '最大回撤': f"{max_drawdown:.2%}" if not pd.isna(max_drawdown) else "nan%",
        '胜率': f"{win_rate:.2%}" if not pd.isna(win_rate) else "nan%",
        '交易次数': int(trade_times)
    }
    return nav, bench_nav, metrics


# ---------------------------- 主程序 ----------------------------
def main():
    print("=" * 60)
    print("步骤15：策略回测与绩效分析（最终修复版）")
    print("=" * 60)

    stock = "A_sh.600036_招商银行"
    df = load_data(stock)

    # 关键修复：将date列设为索引，并排序
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
    else:
        # 如果没有date列，则使用现有索引（假定已是日期）
        if not isinstance(df.index, pd.DatetimeIndex):
            print("警告：索引不是日期类型，回测可能出错。")

    features = FEATURE_COLS + ['llm_score']
    close_prices = df['close'].copy()
    # 确保无缺失值
    if close_prices.isnull().any():
        print("警告：收盘价有缺失，向前填充。")
        close_prices = close_prices.fillna(method='ffill')

    print("正在生成交易信号...")
    signal_probs = rolling_predict(df, features, seq_len=20, window=60)
    print(f"生成信号数量：{len(signal_probs)}")

    print("正在回测...")
    try:
        nav, bench_nav, metrics = backtest(signal_probs, close_prices, initial_capital=1e6, transaction_cost=0.001)
    except Exception as e:
        print(f"回测失败：{e}")
        return

    print("\n=== 策略绩效指标 ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    plt.figure(figsize=(12, 6))
    plt.plot(nav.index, nav, label='Strategy (Weighted Ensemble)', linewidth=2)
    plt.plot(bench_nav.index, bench_nav, label='Buy & Hold', linewidth=2, linestyle='--')
    plt.title('Strategy vs Benchmark Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (CNY)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'backtest_curve.png'), dpi=300)
    print(f"\n净值曲线图已保存：{FIGURE_DIR}/backtest_curve.png")

    peak = nav.cummax()
    drawdown = (nav - peak) / peak
    plt.figure(figsize=(12, 4))
    plt.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
    plt.plot(drawdown.index, drawdown, color='red', linewidth=1)
    plt.title('Strategy Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'backtest_drawdown.png'), dpi=300)
    print(f"回撤曲线图已保存：{FIGURE_DIR}/backtest_drawdown.png")

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(RESULT_DIR, "backtest_metrics.csv"), index=False)
    print(f"绩效指标已保存：{RESULT_DIR}/backtest_metrics.csv")
    print("\n步骤15完成！")


if __name__ == "__main__":
    main()