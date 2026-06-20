#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step56_strong_baselines.py
强基线对比：vanilla PatchTST、LSTM、均线交叉、动量策略
覆盖全部 18 只股票，输出绩效汇总表
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

from step39_patchtst_model import PatchTST
from step38_robust_multi_stock import load_individual_features, load_stock_data, robust_backtest

# ===== 配置 =====
RESULT_DIR = "data/results"
os.makedirs(RESULT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# 超参数（与自适应模型一致）
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

# 全部 18 只股票
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

# ===== 辅助函数 =====
def prepare_sequences(df, features, seq_len=SEQ_LEN):
    X, y = [], []
    for i in range(seq_len, len(df)):
        X.append(df[features].iloc[i-seq_len:i].values)
        y.append(df['label'].iloc[i])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ===== 基线 1：Vanilla PatchTST =====
def rolling_patchtst(df, features):
    """滚动预测 vanilla PatchTST（无自适应）"""
    test_indices = list(range(WINDOW + SEQ_LEN, len(df) - 1, TEST_STRIDE))
    if not test_indices:
        return pd.Series(dtype=float)
    input_dim = len(features)
    dates, probs = [], []
    for test_idx in tqdm(test_indices, desc="PatchTST", leave=False):
        train_end = test_idx - 1
        train_start = train_end - WINDOW
        if train_start < 0:
            continue
        train_df = df.iloc[train_start:train_end]
        split_idx = int(len(train_df) * 0.8)
        if split_idx < SEQ_LEN + 1:
            continue
        train_part = train_df.iloc[:split_idx]
        val_part = train_df.iloc[split_idx:]

        X_train, y_train = prepare_sequences(train_part, features)
        X_val, y_val = prepare_sequences(val_part, features)
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

        # 训练 vanilla PatchTST
        model = PatchTST(input_dim=input_dim, seq_len=SEQ_LEN,
                         patch_len=PATCH_LEN, stride=STRIDE,
                         d_model=D_MODEL, n_heads=N_HEADS,
                         num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.BCELoss()

        train_dataset = TensorDataset(torch.tensor(X_train_scaled), torch.tensor(y_train))
        val_dataset = TensorDataset(torch.tensor(X_val_scaled), torch.tensor(y_val))
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        best_val_loss = float('inf')
        best_state = None
        for _ in range(EPOCHS):
            model.train()
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                pred = model(Xb).squeeze()
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                    pred = model(Xb).squeeze()
                    val_loss += criterion(pred, yb).item()
            val_loss /= len(val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            prob = model(torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)).item()
        dates.append(df.index[test_idx])
        probs.append(prob)
    return pd.Series(probs, index=dates, name='prob')

# ===== 基线 2：LSTM =====
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.sigmoid(self.fc(out)).squeeze(-1)

def rolling_lstm(df, features):
    """滚动预测 LSTM"""
    test_indices = list(range(WINDOW + SEQ_LEN, len(df) - 1, TEST_STRIDE))
    if not test_indices:
        return pd.Series(dtype=float)
    input_dim = len(features)
    dates, probs = [], []
    for test_idx in tqdm(test_indices, desc="LSTM", leave=False):
        train_end = test_idx - 1
        train_start = train_end - WINDOW
        if train_start < 0:
            continue
        train_df = df.iloc[train_start:train_end]
        split_idx = int(len(train_df) * 0.8)
        if split_idx < SEQ_LEN + 1:
            continue
        train_part = train_df.iloc[:split_idx]
        val_part = train_df.iloc[split_idx:]

        X_train, y_train = prepare_sequences(train_part, features)
        X_val, y_val = prepare_sequences(val_part, features)
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

        model = LSTMModel(input_dim, hidden_dim=64, num_layers=2, dropout=0.2).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.BCELoss()

        train_dataset = TensorDataset(torch.tensor(X_train_scaled), torch.tensor(y_train))
        val_dataset = TensorDataset(torch.tensor(X_val_scaled), torch.tensor(y_val))
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        best_val_loss = float('inf')
        best_state = None
        for _ in range(EPOCHS):
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
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            prob = model(torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)).item()
        dates.append(df.index[test_idx])
        probs.append(prob)
    return pd.Series(probs, index=dates, name='prob')

# ===== 基线 3：均线交叉策略（MA5/MA20） =====
def ma_crossover_signal(df):
    """生成均线交叉信号：1 当 MA5 > MA20，-1 当 MA5 < MA20，0 当交叉"""
    close = df['close']
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    signal = pd.Series(0, index=df.index)
    signal[ma5 > ma20] = 1
    signal[ma5 < ma20] = -1
    # 交叉时信号变为0，但回测函数使用概率，我们直接生成概率（1或-1映射到0.8或0.2）
    prob = signal.map({1: 0.8, -1: 0.2, 0: 0.5})
    return prob.fillna(0.5)

# ===== 基线 4：动量策略（过去20日收益） =====
def momentum_signal(df, lookback=20):
    """动量策略：过去20日收益>0则看多，否则看空"""
    ret = df['close'].pct_change(lookback)
    signal = pd.Series(0.5, index=df.index)
    signal[ret > 0] = 0.8
    signal[ret < 0] = 0.2
    return signal

# ===== 主函数 =====
def main():
    print("=" * 60)
    print("step56：强基线对比实验")
    print("=" * 60)

    all_results = []

    for stock in tqdm(ALL_STOCKS, desc="Stocks"):
        print(f"\n处理 {stock} ...")
        try:
            features = load_individual_features(stock)
            df = load_stock_data(stock, features)
            print(f"  数据长度: {len(df)}")

            # ----- 1. Vanilla PatchTST -----
            cache_patch = os.path.join(RESULT_DIR, f"vanilla_patchtst_probs_{stock}.csv")
            if os.path.exists(cache_patch):
                probs_patch = pd.read_csv(cache_patch, index_col=0, parse_dates=True)['prob']
            else:
                print("  运行 Vanilla PatchTST ...")
                set_seed(42)
                probs_patch = rolling_patchtst(df, features)
                probs_patch.to_csv(cache_patch)
            metrics_patch = robust_backtest(probs_patch, df, long_thresh=0.55, short_thresh=0.45,
                                            vol_target=0.15, stop_loss=0.01, transaction_cost=0.001)
            sharpe_patch = float(metrics_patch['夏普比率'])

            # ----- 2. LSTM -----
            cache_lstm = os.path.join(RESULT_DIR, f"lstm_probs_{stock}.csv")
            if os.path.exists(cache_lstm):
                probs_lstm = pd.read_csv(cache_lstm, index_col=0, parse_dates=True)['prob']
            else:
                print("  运行 LSTM ...")
                set_seed(42)
                probs_lstm = rolling_lstm(df, features)
                probs_lstm.to_csv(cache_lstm)
            metrics_lstm = robust_backtest(probs_lstm, df, long_thresh=0.55, short_thresh=0.45,
                                           vol_target=0.15, stop_loss=0.01, transaction_cost=0.001)
            sharpe_lstm = float(metrics_lstm['夏普比率'])

            # ----- 3. 均线交叉 -----
            probs_ma = ma_crossover_signal(df)
            # 只取有预测的日期（与回测对齐）
            common_idx = probs_ma.index.intersection(df.index)
            probs_ma = probs_ma.loc[common_idx]
            metrics_ma = robust_backtest(probs_ma, df, long_thresh=0.55, short_thresh=0.45,
                                         vol_target=0.15, stop_loss=0.01, transaction_cost=0.001)
            sharpe_ma = float(metrics_ma['夏普比率'])

            # ----- 4. 动量策略 -----
            probs_mom = momentum_signal(df, lookback=20)
            probs_mom = probs_mom.loc[common_idx]
            metrics_mom = robust_backtest(probs_mom, df, long_thresh=0.55, short_thresh=0.45,
                                          vol_target=0.15, stop_loss=0.01, transaction_cost=0.001)
            sharpe_mom = float(metrics_mom['夏普比率'])

            # ----- 5. 你的自适应模型（从汇总中读取） -----
            # 从 adaptive_repeat_summary.csv 读取自适应模型的夏普（如果有）
            summary_df = pd.read_csv(os.path.join(RESULT_DIR, "adaptive_repeat_summary.csv"))
            row = summary_df[summary_df['stock'] == stock]
            if not row.empty:
                sharpe_ours = row['sharpe_median'].values[0]
            else:
                sharpe_ours = None

            all_results.append({
                'stock': stock,
                'Vanilla_PatchTST': sharpe_patch,
                'LSTM': sharpe_lstm,
                'MA_Crossover': sharpe_ma,
                'Momentum': sharpe_mom,
                'Ours_Adaptive': sharpe_ours if sharpe_ours is not None else np.nan,
            })

            print(f"  ✅ 结果: Patch={sharpe_patch:.2f}, LSTM={sharpe_lstm:.2f}, MA={sharpe_ma:.2f}, Mom={sharpe_mom:.2f}, Ours={sharpe_ours if sharpe_ours is not None else 'N/A'}")

        except Exception as e:
            print(f"  ❌ 错误: {e}")
            import traceback
            traceback.print_exc()

    # 汇总
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(RESULT_DIR, "strong_baselines_comparison.csv"), index=False)

    # 计算平均
    avg_row = df_results[['Vanilla_PatchTST', 'LSTM', 'MA_Crossover', 'Momentum', 'Ours_Adaptive']].mean()
    avg_row['stock'] = 'AVERAGE'
    df_results = pd.concat([df_results, pd.DataFrame([avg_row])], ignore_index=True)

    df_results.to_csv(os.path.join(RESULT_DIR, "strong_baselines_comparison_with_avg.csv"), index=False)

    print("\n" + "=" * 60)
    print("📊 强基线对比结果（平均夏普）")
    print("=" * 60)
    print(avg_row.to_string())
    print("\n✅ step56 完成！结果保存至 data/results/strong_baselines_comparison*.csv")

if __name__ == "__main__":
    main()