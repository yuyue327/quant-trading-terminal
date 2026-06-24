#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step39_patchtst_model.py
PatchTST 时序预测模型（用于股票涨跌分类，输出上涨概率）
参考：https://github.com/yuqinie98/PatchTST
"""
from seed_manager import set_global_seed
set_global_seed(42)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


class PatchTST(nn.Module):
    """
    PatchTST 分类模型
    - input_dim: 特征维度
    - seq_len: 输入序列长度
    - patch_len: patch 长度（建议 5）
    - stride: patch 步长（建议 1）
    - d_model: Transformer 隐藏维度
    - n_heads: 多头注意力头数
    - num_layers: Transformer 层数
    - dropout: dropout 率
    """

    def __init__(self, input_dim, seq_len=20, patch_len=5, stride=1,
                 d_model=64, n_heads=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        # 计算 patch 数量
        self.num_patches = (seq_len - patch_len) // stride + 1

        # 输入映射（每个时间步的特征 -> d_model）
        self.input_proj = nn.Linear(input_dim, d_model)

        # Patch 编码：将连续 patch_len 个时间步的 d_model 维特征拼接并映射
        # 每个 patch 输入维度 = patch_len * d_model
        self.patch_proj = nn.Linear(patch_len * d_model, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.num_patches)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        """
        # 映射到 d_model
        x = self.input_proj(x)  # (batch, seq_len, d_model)

        # 分 patch
        batch_size = x.size(0)
        patches = []
        for i in range(self.num_patches):
            start = i * self.stride
            end = start + self.patch_len
            patch = x[:, start:end, :]  # (batch, patch_len, d_model)
            patch = patch.reshape(batch_size, -1)  # (batch, patch_len * d_model)
            patches.append(patch)
        patches = torch.stack(patches, dim=1)  # (batch, num_patches, patch_len*d_model)

        # 编码每个 patch
        patch_emb = self.patch_proj(patches)  # (batch, num_patches, d_model)

        # 位置编码
        patch_emb = self.pos_encoder(patch_emb)

        # Transformer
        trans_out = self.transformer(patch_emb)  # (batch, num_patches, d_model)

        # 取最后一个 patch 的输出
        last_out = trans_out[:, -1, :]  # (batch, d_model)

        # 分类
        prob = self.classifier(last_out).squeeze(-1)
        return prob


if __name__ == "__main__":
    # 测试
    model = PatchTST(input_dim=15, seq_len=20, patch_len=5, stride=1)
    dummy = torch.randn(4, 20, 15)
    out = model(dummy)
    print(f"Output shape: {out.shape} (should be (4,))")
    print("PatchTST model defined successfully.")