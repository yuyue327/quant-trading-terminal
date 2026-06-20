#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step28_tcn_attention_model.py
- 构建统一特征集（基于特征频率）
- 定义 TCN + 多头因果注意力模型
- 训练和预测函数（供滚动回测使用）
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
from collections import Counter

# ==================== 配置 ====================
FEATURE_SELECTION_DIR = "data/feature_selection"
OUTPUT_DIR = "data/models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# ==================== 构建统一特征集 ====================
def build_unified_features(min_pct=0.5):
    """
    从 step27 的 per-stock 选择结果中，统计特征频率，返回出现比例 > min_pct 的特征列表
    """
    per_stock_path = os.path.join(FEATURE_SELECTION_DIR, "selected_features_per_stock.json")
    with open(per_stock_path, 'r') as f:
        selected = json.load(f)

    counter = Counter()
    total_stocks = len(selected)
    for stock, feats in selected.items():
        counter.update(feats)

    # 计算频率
    freq = {k: v / total_stocks for k, v in counter.items()}
    unified = [feat for feat, f in freq.items() if f >= min_pct]
    # 按频率降序排列
    unified.sort(key=lambda x: freq[x], reverse=True)

    # 保存频率信息
    freq_df = pd.DataFrame([freq]).T.reset_index()
    freq_df.columns = ['feature', 'frequency']
    freq_df.to_csv(os.path.join(FEATURE_SELECTION_DIR, "feature_frequency.csv"), index=False)

    print(f"共有 {len(unified)} 个特征出现在至少 {min_pct * 100}% 的股票中")
    print(f"统一特征集: {unified}")
    return unified, freq


# ==================== 模型定义 ====================
class Chomp1d(nn.Module):
    """移除卷积后多余的时间步（用于保持序列长度）"""

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """TCN 基本块：膨胀卷积 + 残差连接"""

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """堆叠多个 TemporalBlock，实现多尺度膨胀卷积"""

    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, channels, seq_len)
        return self.network(x)


class CausalSelfAttention(nn.Module):
    """因果自注意力（只允许过去的信息）"""

    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, d_model)
        batch, seq_len, _ = x.shape
        Q = self.w_q(x).view(batch, seq_len, self.n_head, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch, seq_len, self.n_head, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch, seq_len, self.n_head, self.d_k).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is None:
            # 因果掩码（上三角为 -inf）
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(out)


class TCNAttentionModel(nn.Module):
    """TCN 编码 + 因果自注意力 + 分类头"""

    def __init__(self, input_dim, seq_len, tcn_channels=[64, 128, 128], kernel_size=3,
                 d_model=128, n_head=4, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len

        # 输入映射到 d_model
        self.input_fc = nn.Linear(input_dim, d_model)

        # TCN 在序列维度上提取特征（输入通道数为 d_model）
        self.tcn = TemporalConvNet(d_model, tcn_channels, kernel_size, dropout)
        # TCN 输出通道为 tcn_channels[-1]
        tcn_out_dim = tcn_channels[-1]

        # 注意力层
        self.attn = CausalSelfAttention(tcn_out_dim, n_head, dropout)
        self.norm = nn.LayerNorm(tcn_out_dim)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(tcn_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_fc(x)  # (batch, seq_len, d_model)
        # TCN 需要 (batch, channels, seq_len)
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.tcn(x)  # (batch, tcn_out_dim, seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, tcn_out_dim)
        # 自注意力
        attn_out = self.attn(x)  # (batch, seq_len, tcn_out_dim)
        x = self.norm(x + attn_out)
        # 全局平均池化
        pooled = x.mean(dim=1)  # (batch, tcn_out_dim)
        out = self.classifier(pooled).squeeze(-1)
        return out


# ==================== 训练函数 ====================
def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, lr=0.001, patience=10):
    """训练单个模型，返回训练后的模型和验证集最佳损失"""
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    best_val_loss = float('inf')
    best_state = None
    wait = 0

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
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_val_loss


def predict(model, X):
    """预测概率"""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        probs = model(X_tensor).cpu().numpy()
    return probs


# ==================== 主程序测试 ====================
if __name__ == "__main__":
    # 测试模型构建
    unified_features, freq = build_unified_features(min_pct=0.5)
    print(f"\n统一特征集大小: {len(unified_features)}")
    # 简单测试模型前向传播
    model = TCNAttentionModel(input_dim=len(unified_features), seq_len=20)
    dummy = torch.randn(4, 20, len(unified_features))
    out = model(dummy)
    print(f"测试输出形状: {out.shape} (应为 (4,))")
    print("step28 模型定义完成。")