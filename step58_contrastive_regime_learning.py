#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step58_contrastive_regime_learning.py
使用对比学习（SimCLR风格）从市场数据中学习隐式的市场状态表征
输出：每个时间点的状态标签（0-3），以及状态转移矩阵
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ===== 配置 =====
RESULT_DIR = "data/results"
os.makedirs(RESULT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# ===== 对比学习编码器 =====
class RegimeEncoder(nn.Module):
    """将市场数据映射到隐式状态空间的编码器"""

    def __init__(self, input_dim, hidden_dim=64, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        proj = self.projection_head(z)
        return z, proj


def contrastive_loss(z1, z2, temperature=0.1):
    """
    对比损失（SimCLR风格）
    z1, z2: 同一批数据经过不同数据增强后的表征
    """
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)

    # 相似度矩阵
    sim_matrix = torch.mm(z, z.T) / temperature

    # 正样本对：z1[i] 与 z2[i] 相似
    labels = torch.cat([torch.arange(batch_size), torch.arange(batch_size)]).to(z.device)
    labels = (labels + batch_size) % (2 * batch_size)

    loss = F.cross_entropy(sim_matrix, labels)
    return loss


def load_market_data(stock_name, window=60):
    """加载单只股票的窗口数据用于状态学习"""
    from step38_robust_multi_stock import load_stock_data, load_individual_features
    features = load_individual_features(stock_name)
    df = load_stock_data(stock_name, features)

    # 使用技术指标作为状态学习的输入
    state_features = ['volatility_5', 'volatility_20', 'pct_change', 'volume_ratio',
                      'RSI', 'MACD', 'ATR', 'close_position']
    state_features = [f for f in state_features if f in df.columns]

    # 构建滑动窗口样本
    X = []
    dates = []
    for i in range(window, len(df)):
        X.append(df[state_features].iloc[i - window:i].mean().values)
        dates.append(df.index[i])

    X = np.array(X, dtype=np.float32)
    return X, dates, state_features


def augment_data(X):
    """数据增强：添加噪声和缩放扰动"""
    # 原始样本
    X1 = X

    # 增强1：添加高斯噪声
    noise = np.random.normal(0, 0.05, X.shape)
    X2 = X + noise

    # 增强2：尺度扰动
    scale = np.random.uniform(0.8, 1.2, (X.shape[0], 1))
    X3 = X * scale

    return torch.tensor(X1, dtype=torch.float32), torch.tensor(X2, dtype=torch.float32)


def learn_regime_labels(stock_names, n_states=4):
    """
    使用对比学习学习市场状态标签
    返回：每个时间点的状态标签
    """
    print("📊 学习市场状态隐式表征...")

    # 1. 收集所有股票的市场数据
    all_X = []
    all_dates = []
    all_stocks = []

    for stock in tqdm(stock_names, desc="加载股票数据"):
        try:
            X, dates, features = load_market_data(stock)
            all_X.append(X)
            all_dates.extend(dates)
            all_stocks.extend([stock] * len(X))
        except Exception as e:
            print(f"  跳过 {stock}: {e}")

    if not all_X:
        print("❌ 没有有效数据")
        return None

    X_all = np.concatenate(all_X, axis=0)
    print(f"  总样本数: {len(X_all)}")

    # 2. 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # 3. 训练对比学习编码器
    input_dim = X_scaled.shape[1]
    encoder = RegimeEncoder(input_dim, hidden_dim=64, latent_dim=16).to(DEVICE)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)

    print("  训练对比学习编码器...")
    for epoch in range(50):
        # 随机采样批次
        idx = np.random.choice(len(X_scaled), min(256, len(X_scaled)), replace=False)
        batch = X_scaled[idx]
        X1, X2 = augment_data(batch)
        X1, X2 = X1.to(DEVICE), X2.to(DEVICE)

        z1, proj1 = encoder(X1)
        z2, proj2 = encoder(X2)

        loss = contrastive_loss(proj1, proj2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 4. 提取所有样本的表征
    encoder.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
        z, _ = encoder(X_tensor)
        z = z.cpu().numpy()

    # 5. 用K-Means聚类得到状态标签
    kmeans = KMeans(n_clusters=n_states, random_state=42, n_init=10)
    labels = kmeans.fit_predict(z)

    # 6. 将标签映射到原始数据上，并将日期转换为字符串
    state_map = {}
    idx = 0
    for stock, stock_X in zip(all_stocks, all_X):
        n_samples = len(stock_X)
        # 提取对应的日期并转换为ISO格式字符串
        date_list = all_dates[idx:idx + n_samples]
        date_str = [d.strftime("%Y-%m-%d") if hasattr(d, 'strftime') else str(d) for d in date_list]
        state_map[stock] = {
            'labels': labels[idx:idx + n_samples].tolist(),
            'dates': date_str
        }
        idx += n_samples

    # 7. 分析状态分布
    print(f"\n📊 状态分布: {np.bincount(labels)}")

    # 8. 保存结果（使用可JSON序列化的数据）
    output = {
        'state_labels': state_map,
        'cluster_centers': kmeans.cluster_centers_.tolist(),
        'n_states': n_states,
        'total_samples': len(labels)
    }

    with open(os.path.join(RESULT_DIR, "contrastive_state_labels.json"), 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✅ 状态标签已保存至 contrastive_state_labels.json")

    return state_map


def main():
    print("=" * 60)
    print("step58：对比学习市场状态发现")
    print("=" * 60)

    # 获取股票列表
    from step38_robust_multi_stock import STOCKS
    stock_names = STOCKS + [
        'A_sh.600887_伊利股份', 'A_sh.601012_隆基绿能', 'A_sh.601688_华泰证券',
        'A_sz.000333_美的集团', 'A_sz.000568_泸州老窖', 'A_sz.000651_格力电器',
        'A_sz.002142_宁波银行', 'A_sz.002594_比亚迪', 'A_sz.300059_东方财富'
    ]

    state_map = learn_regime_labels(stock_names, n_states=4)

    print("\n✅ step58 完成！")
    print("💡 提示：现在可以用对比学习得到的隐式状态替代原有的规则驱动状态划分")


if __name__ == "__main__":
    main()