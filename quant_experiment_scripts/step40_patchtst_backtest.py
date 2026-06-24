#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step40_patchtst_backtest.py - 稳定版本 v3
适配中文指标名和百分比字符串
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
from step38_robust_multi_stock import load_individual_features, load_stock_data, robust_backtest

# ---------------------------- 配置 ----------------------------
DATA_DIR = "data/features"
FEATURE_SEL_DIR = "data/feature_selection"
RESULT_DIR = "data/results"
os.makedirs(RESULT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# 超参数
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
N_REPEATS = 20   # 先改为1快速测试，确认有效后再改回20

# ---------------------------- 辅助函数 ----------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_sequences(df, features, seq_len=SEQ_LEN):
    X, y = [], []
    for i in range(seq_len, len(df)):
        X.append(df[features].iloc[i - seq_len:i].values)
        y.append(df['label'].iloc[i])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y

def train_patchtst(X_train, y_train, X_val, y_val, input_dim, epochs=EPOCHS):
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = PatchTST(input_dim=input_dim, seq_len=SEQ_LEN,
                     patch_len=PATCH_LEN, stride=STRIDE,
                     d_model=D_MODEL, n_heads=N_HEADS,
                     num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)
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

def rolling_predict_patchtst(df, features, seed, window=WINDOW, stride=TEST_STRIDE):
    set_seed(seed)
    test_indices = list(range(window + SEQ_LEN, len(df) - 1, stride))
    if not test_indices:
        print(f"警告: 种子 {seed} 未生成任何测试点")
        return pd.Series(dtype=float)
    input_dim = len(features)
    dates = []
    probs = []

    for test_idx in tqdm(test_indices, desc=f"Rolling (seed={seed})", leave=False):
        train_end = test_idx - 1
        train_start = train_end - window
        if train_start < 0:
            continue
        train_df = df.iloc[train_start:train_end]
        split_idx = int(len(train_df) * 0.8)
        if split_idx < SEQ_LEN + 1:
            continue
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

        model = train_patchtst(X_train_scaled, y_train, X_val_scaled, y_val, input_dim, epochs=EPOCHS)
        model.eval()
        with torch.no_grad():
            prob = model(torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)).item()
        dates.append(df.index[test_idx])
        probs.append(prob)

    if len(probs) == 0:
        return pd.Series(dtype=float)
    return pd.Series(probs, index=dates, name='prob')

def parse_metric_value(value):
    """将 robust_backtest 返回的值转为浮点数"""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # 去掉百分号并除以100
        if value.endswith('%'):
            return float(value.rstrip('%')) / 100.0
        else:
            return float(value)
    return np.nan

def run_single_repeat(stock, features, df, repeat_id):
    base_seed = 42 + repeat_id
    print(f"  >>> 重复实验 {repeat_id+1}/{N_REPEATS} (seed={base_seed})")
    probs = rolling_predict_patchtst(df, features, seed=base_seed,
                                     window=WINDOW, stride=TEST_STRIDE)
    if probs.empty:
        print(f"      警告: 种子 {base_seed} 未产生任何预测，跳过本次")
        return None

    metrics = robust_backtest(probs, df,
                              long_thresh=0.55, short_thresh=0.45,
                              vol_target=0.15, stop_loss=0.01,
                              transaction_cost=0.001)

    # 映射中文键到标准英文名
    mapped = {
        'sharpe_ratio': parse_metric_value(metrics.get('夏普比率', np.nan)),
        'annual_return': parse_metric_value(metrics.get('年化收益率', np.nan)),
        'max_drawdown': parse_metric_value(metrics.get('最大回撤', np.nan)),
        'win_rate': parse_metric_value(metrics.get('胜率', np.nan)),
        'total_trades': parse_metric_value(metrics.get('交易次数', np.nan))
    }
    mapped['repeat_id'] = repeat_id
    mapped['seed'] = base_seed
    return mapped

# ---------------------------- 主程序 ----------------------------
def main():
    print("=" * 60)
    print(f"step40：PatchTST 模型稳健性实验（重复运行 {N_REPEATS} 次）")
    print("=" * 60)

    stock = "A_sh.600036_招商银行"
    features = load_individual_features(stock)
    print(f"特征数量: {len(features)}")
    df = load_stock_data(stock, features)

    all_metrics = []
    for rep in range(N_REPEATS):
        m = run_single_repeat(stock, features, df, rep)
        if m is not None:
            all_metrics.append(m)

    if not all_metrics:
        print("错误: 没有任何一次重复实验产生有效结果，请检查数据或超参数。")
        return

    results_df = pd.DataFrame(all_metrics)

    # 提取关键指标
    key_metrics = ['sharpe_ratio', 'annual_return', 'max_drawdown', 'win_rate', 'total_trades']
    summary = {}
    for m in key_metrics:
        if m in results_df.columns:
            values = results_df[m].dropna().values
            if len(values) > 0:
                summary[f"{m}_median"] = np.median(values)
                summary[f"{m}_std"] = np.std(values)
                summary[f"{m}_min"] = np.min(values)
                summary[f"{m}_max"] = np.max(values)
            else:
                summary[f"{m}_median"] = np.nan
                summary[f"{m}_std"] = np.nan
                summary[f"{m}_min"] = np.nan
                summary[f"{m}_max"] = np.nan
        else:
            print(f"警告: 指标 {m} 不在结果中，跳过")

    print("\n" + "=" * 60)
    print(f"基于 {len(all_metrics)} 次有效重复实验的稳健绩效（中位数 ± 标准差）")
    print("=" * 60)
    for m in key_metrics:
        if f"{m}_median" in summary:
            print(f"{m:15s}: {summary[f'{m}_median']:.4f} ± {summary[f'{m}_std']:.4f}  "
                  f"(min={summary[f'{m}_min']:.4f}, max={summary[f'{m}_max']:.4f})")

    # 保存结果
    results_df.to_csv(os.path.join(RESULT_DIR, "patchtst_600036_all_repeats.csv"), index=False)
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(RESULT_DIR, "patchtst_600036_stable_metrics.csv"), index=False)

    print("\n完整重复实验数据已保存至:")
    print(f"  - {RESULT_DIR}/patchtst_600036_all_repeats.csv")
    print(f"  - {RESULT_DIR}/patchtst_600036_stable_metrics.csv")
    print("step40 完成。")

if __name__ == "__main__":
    main()