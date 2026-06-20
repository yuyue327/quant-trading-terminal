#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step47_train_adaptive_patchtst.py - 修正版
训练状态自适应多专家 PatchTST 模型（招商银行）
"""
import os
import random
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
from step45_market_state_detector import detect_market_state
from step46_state_adaptive_patchtst import StateAdaptivePatchTST
from step38_robust_multi_stock import load_individual_features, load_stock_data, robust_backtest


# ---------------------------- 辅助函数（独立定义） ----------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_sequences(df, features, seq_len=20):
    X, y = [], []
    for i in range(seq_len, len(df)):
        X.append(df[features].iloc[i - seq_len:i].values)
        y.append(df['label'].iloc[i])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y


def parse_metric_value(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        if value.endswith('%'):
            return float(value.rstrip('%')) / 100.0
        else:
            return float(value)
    return np.nan


# ---------------------------- 配置 ----------------------------
RESULT_DIR = "data/results"
os.makedirs(RESULT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

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


def rolling_predict_adaptive(df, features, window=WINDOW, stride=TEST_STRIDE):
    test_indices = list(range(window + SEQ_LEN, len(df) - 1, stride))
    if not test_indices:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    input_dim = len(features)
    dates = []
    probs = []
    uncertainties = []

    for test_idx in tqdm(test_indices, desc="Rolling Adaptive"):
        train_end = test_idx - 1
        train_start = train_end - window
        if train_start < 0:
            continue
        train_df = df.iloc[train_start:train_end].copy()

        # 计算市场状态（基于训练数据）
        state_series = detect_market_state(train_df)
        if len(state_series) != len(train_df):
            state_series = state_series.iloc[:len(train_df)]
        train_df['state'] = state_series

        # 划分训练/验证集
        split_idx = int(len(train_df) * 0.8)
        if split_idx < SEQ_LEN + 1:
            continue
        train_part = train_df.iloc[:split_idx]
        val_part = train_df.iloc[split_idx:]

        # 准备序列特征和标签
        X_train, y_train = prepare_sequences(train_part, features, SEQ_LEN)
        X_val, y_val = prepare_sequences(val_part, features, SEQ_LEN)

        # 对应的状态标签（每个序列末尾日期的状态）
        state_train = []
        for i in range(SEQ_LEN, len(train_part)):
            state_train.append(train_part['state'].iloc[i])
        state_val = []
        for i in range(SEQ_LEN, len(val_part)):
            state_val.append(val_part['state'].iloc[i])
        state_train = np.array(state_train, dtype=np.int64)
        state_val = np.array(state_val, dtype=np.int64)

        if len(X_train) == 0 or len(X_val) == 0:
            continue

        # 标准化
        scaler = StandardScaler()
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
        X_val_flat = X_val.reshape(-1, X_val.shape[-1])
        X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape)

        # 转换为张量
        X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)
        state_train_t = torch.tensor(state_train, dtype=torch.long)
        state_val_t = torch.tensor(state_val, dtype=torch.long)

        train_dataset = TensorDataset(X_train_t, y_train_t, state_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t, state_val_t)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # 初始化模型
        model = StateAdaptivePatchTST(
            input_dim=input_dim, seq_len=SEQ_LEN, patch_len=PATCH_LEN, stride=STRIDE,
            d_model=D_MODEL, n_heads=N_HEADS, num_layers=NUM_LAYERS, dropout=DROPOUT,
            n_states=4, uncertainty=True
        ).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)

        def nll_loss(mean, log_var, target):
            var = torch.exp(log_var)
            loss = 0.5 * (log_var + (target - mean) ** 2 / var)
            return loss.mean()

        best_val_loss = float('inf')
        best_state_dict = None
        for epoch in range(EPOCHS):
            model.train()
            for Xb, yb, sb in train_loader:
                Xb, yb, sb = Xb.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE)
                optimizer.zero_grad()
                mean, log_var = model(Xb, sb)
                loss = nll_loss(mean, log_var, yb)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for Xb, yb, sb in val_loader:
                    Xb, yb, sb = Xb.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE)
                    mean, log_var = model(Xb, sb)
                    loss = nll_loss(mean, log_var, yb)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state_dict)
        model.eval()

        # 预测测试点
        X_test_seq = df[features].iloc[test_idx - SEQ_LEN:test_idx].values
        X_test_seq = X_test_seq.reshape(1, SEQ_LEN, -1)
        X_test_scaled = scaler.transform(X_test_seq.reshape(-1, X_test_seq.shape[-1])).reshape(X_test_seq.shape)
        X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)

        # 获取测试点的市场状态（基于截至测试日前一天的数据）
        temp_df = df.iloc[:test_idx].copy()
        test_state_series = detect_market_state(temp_df)
        test_state = int(test_state_series.iloc[-1])
        test_state_t = torch.tensor([test_state], dtype=torch.long).to(DEVICE)

        with torch.no_grad():
            mean, log_var = model(X_test_t, test_state_t)
            prob = mean.item()
            unc = torch.exp(0.5 * log_var).item()

        dates.append(df.index[test_idx])
        probs.append(prob)
        uncertainties.append(unc)

    return pd.Series(probs, index=dates, name='prob'), pd.Series(uncertainties, index=dates, name='uncertainty')


def main():
    print("=" * 60)
    print("step47：自适应多专家 PatchTST 回测（招商银行）")
    print("=" * 60)

    stock = "A_sh.600036_招商银行"
    features = load_individual_features(stock)
    print(f"特征数量: {len(features)}")
    df = load_stock_data(stock, features)

    cache_probs = os.path.join(RESULT_DIR, "adaptive_probs_600036.csv")
    if os.path.exists(cache_probs):
        probs = pd.read_csv(cache_probs, index_col=0, parse_dates=True)['prob']
        print("从缓存加载预测概率")
    else:
        print("开始滚动预测（自适应模型）...")
        probs, unc = rolling_predict_adaptive(df, features)
        result_df = pd.DataFrame({'prob': probs, 'uncertainty': unc})
        result_df.to_csv(cache_probs)
        print(f"预测概率已保存至 {cache_probs}")

    metrics = robust_backtest(probs, df,
                              long_thresh=0.55, short_thresh=0.45,
                              vol_target=0.15, stop_loss=0.01,
                              transaction_cost=0.001)

    print("\n=== 自适应 PatchTST 策略绩效（招商银行） ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    pd.DataFrame([metrics]).to_csv(os.path.join(RESULT_DIR, "adaptive_600036_metrics.csv"), index=False)
    print("step47 完成。")


if __name__ == "__main__":
    set_seed(42)
    main()