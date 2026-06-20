#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step50_hyperparameter_tuning.py
超参数优化：使用 Optuna 对自适应 PatchTST 进行调优
目标：最大化夏普比率（基于招商银行）
"""
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# 导入你的模型和工具
from step39_patchtst_model import PatchTST
from step45_market_state_detector import detect_market_state
from step46_state_adaptive_patchtst import StateAdaptivePatchTST
from step38_robust_multi_stock import load_individual_features, load_stock_data, robust_backtest

# ===== 配置 =====
RESULT_DIR = "data/results"
os.makedirs(RESULT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# 固定参数（调优过程中不变）
SEQ_LEN = 20
STRIDE = 1
NUM_LAYERS = 2
BATCH_SIZE = 32
EPOCHS = 20  # 调优时减少epoch以加速
WINDOW = 360
TEST_STRIDE = 20
STOCK = "A_sh.600036_招商银行"  # 单股票调优


# ===== 辅助函数 =====
def set_seed(seed):
    import random
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


def train_model(X_train, y_train, X_val, y_val, input_dim, params):
    """用给定超参数训练模型"""
    model = StateAdaptivePatchTST(
        input_dim=input_dim,
        seq_len=SEQ_LEN,
        patch_len=params['patch_len'],
        stride=STRIDE,
        d_model=params['d_model'],
        n_heads=params['n_heads'],
        num_layers=NUM_LAYERS,
        dropout=params['dropout'],
        n_states=4,
        uncertainty=False  # 调优时关闭不确定性输出（简化）
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    criterion = nn.BCELoss()

    train_dataset = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train),
        torch.tensor([0] * len(X_train))  # 占位状态
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val),
        torch.tensor(y_val),
        torch.tensor([0] * len(X_val))
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    best_val_loss = float('inf')
    best_state = None
    for epoch in range(EPOCHS):
        model.train()
        for Xb, yb, _ in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(Xb, None)  # 不使用状态标签（统一用0）
            loss = criterion(pred.squeeze(), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb, _ in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                pred = model(Xb, None)
                val_loss += criterion(pred.squeeze(), yb).item()
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


def rolling_evaluate(df, features, params):
    """滚动回测并返回夏普比率"""
    test_indices = list(range(WINDOW + SEQ_LEN, len(df) - 1, TEST_STRIDE))
    if not test_indices:
        return -10.0

    input_dim = len(features)
    dates = []
    probs = []

    for test_idx in tqdm(test_indices, desc="Rolling", leave=False):
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

        model = train_model(X_train_scaled, y_train, X_val_scaled, y_val, input_dim, params)
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE), None)
            prob = pred.squeeze().item()

        dates.append(df.index[test_idx])
        probs.append(prob)

    if len(probs) == 0:
        return -10.0

    probs_series = pd.Series(probs, index=dates, name='prob')
    metrics = robust_backtest(
        probs_series, df,
        long_thresh=0.55, short_thresh=0.45,
        vol_target=0.15, stop_loss=0.01,
        transaction_cost=0.001
    )
    sharpe = float(metrics.get('夏普比率', -10).replace('%', '')) if isinstance(metrics.get('夏普比率'),
                                                                                str) else metrics.get('夏普比率', -10)
    if isinstance(sharpe, str):
        sharpe = float(sharpe.replace('%', ''))
    return sharpe


def objective(trial):
    """Optuna目标函数"""
    params = {
        'patch_len': trial.suggest_int('patch_len', 3, 12, step=1),
        'd_model': trial.suggest_categorical('d_model', [32, 48, 64, 80, 96, 128]),
        'n_heads': trial.suggest_categorical('n_heads', [2, 4, 6, 8]),
        'dropout': trial.suggest_float('dropout', 0.05, 0.4, step=0.05),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True),
    }

    print(f"\n🔬 Trial {trial.number}: {params}")

    try:
        features = load_individual_features(STOCK)
        df = load_stock_data(STOCK, features)
        sharpe = rolling_evaluate(df, features, params)
        print(f"  夏普比率: {sharpe:.4f}")
        return sharpe
    except Exception as e:
        print(f"  ❌ 错误: {e}")
        return -10.0


def main():
    print("=" * 60)
    print("step50：超参数优化 (Optuna)")
    print("=" * 60)

    # 创建Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    print(f"开始调优，共 30 次试验...")
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    # 结果汇总
    print("\n" + "=" * 60)
    print("✅ 调优完成")
    print(f"最佳试验: #{study.best_trial.number}")
    print(f"最佳夏普: {study.best_value:.4f}")
    print(f"最佳参数: {study.best_params}")
    print("=" * 60)

    # 保存结果
    df_trials = study.trials_dataframe()
    df_trials.to_csv(os.path.join(RESULT_DIR, "hyperopt_trials.csv"), index=False)

    with open(os.path.join(RESULT_DIR, "hyperopt_best_params.json"), 'w') as f:
        json.dump(study.best_params, f, indent=2)

    # 可视化
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(os.path.join(RESULT_DIR, "hyperopt_history.html"))
        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.write_html(os.path.join(RESULT_DIR, "hyperopt_importance.html"))
        fig3 = optuna.visualization.plot_parallel_coordinate(study)
        fig3.write_html(os.path.join(RESULT_DIR, "hyperopt_parallel.html"))
        print("📊 可视化图表已保存至 data/results/")
    except Exception as e:
        print(f"⚠️ 可视化生成失败: {e}")

    print("\n📌 建议使用以下参数进行最终训练:")
    print(json.dumps(study.best_params, indent=2))


if __name__ == "__main__":
    main()