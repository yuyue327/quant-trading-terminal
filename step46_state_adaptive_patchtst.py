#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step46_state_adaptive_patchtst.py
定义状态自适应多专家 PatchTST 模型
包含 K=4 个子模型，根据市场状态动态选择或融合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchTSTBlock(nn.Module):
    """单个 PatchTST 编码器块（简化版，可复用 step39 中的实现）"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class StateAdaptivePatchTST(nn.Module):
    """
    状态自适应 PatchTST
    - n_states: 状态数量（默认4）
    - 每个状态对应一个独立的 PatchTST 编码器 + 分类头
    - 前向时根据状态标签选择对应的子模型
    """

    def __init__(self, input_dim, seq_len, patch_len, stride,
                 d_model=64, n_heads=4, num_layers=2, dropout=0.1, n_states=4,
                 uncertainty=True):
        super().__init__()
        self.uncertainty = uncertainty
        self.n_states = n_states

        # 将原始序列切分成 patch
        self.patch_len = patch_len
        self.stride = stride
        self.seq_len = seq_len
        # 计算 patch 数量
        self.n_patches = (seq_len - patch_len) // stride + 1
        # 线性映射到 d_model
        self.input_proj = nn.Linear(patch_len * input_dim, d_model)

        # 为每个状态创建独立的 Transformer 编码器
        self.encoders = nn.ModuleList([
            nn.ModuleList([PatchTSTBlock(d_model, n_heads, dropout) for _ in range(num_layers)])
            for _ in range(n_states)
        ])

        # 每个状态的分类头（输出概率，或概率均值+方差）
        if uncertainty:
            # 输出两个值: mean 和 log_var
            self.heads = nn.ModuleList([
                nn.Linear(d_model, 2) for _ in range(n_states)
            ])
        else:
            self.heads = nn.ModuleList([
                nn.Linear(d_model, 1) for _ in range(n_states)
            ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, state_labels=None):
        """
        x: [batch, seq_len, input_dim]
        state_labels: [batch] 每个样本的状态标签 (0..n_states-1)
        如果 state_labels 为 None，则使用所有状态的加权平均（测试时可启用）
        """
        batch, seq_len, input_dim = x.shape

        # 1. 切分为 patch
        patches = []
        for i in range(0, seq_len - self.patch_len + 1, self.stride):
            patch = x[:, i:i + self.patch_len, :]  # [batch, patch_len, input_dim]
            patch = patch.reshape(batch, -1)  # [batch, patch_len*input_dim]
            patches.append(patch)
        patches = torch.stack(patches, dim=1)  # [batch, n_patches, patch_len*input_dim]

        # 2. 线性映射到 d_model
        x = self.input_proj(patches)  # [batch, n_patches, d_model]
        x = self.dropout(x)

        # 3. 根据状态选择编码器
        outputs = []
        for i in range(batch):
            state = state_labels[i].item() if state_labels is not None else 0
            h = x[i:i + 1]  # [1, n_patches, d_model]
            for layer in self.encoders[state]:
                h = layer(h)
            # 全局池化
            h = h.mean(dim=1)  # [1, d_model]
            if self.uncertainty:
                out = self.heads[state](h)  # [1, 2]
            else:
                out = torch.sigmoid(self.heads[state](h))  # [1, 1]
            outputs.append(out)

        outputs = torch.cat(outputs, dim=0)  # [batch, 2 or 1]
        if self.uncertainty:
            mean = torch.sigmoid(outputs[:, 0])  # 限制在 (0,1)
            log_var = outputs[:, 1]
            return mean, log_var
        else:
            return outputs.squeeze(-1)