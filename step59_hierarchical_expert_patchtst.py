#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step59_hierarchical_expert_patchtst.py
分层专家PatchTST架构：共享底层编码器 + 状态自适应Adapter + 专家头
相比4个独立模型，参数效率更高，且支持状态之间的知识迁移
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HierarchicalExpertPatchTST(nn.Module):
    """
    分层专家PatchTST架构

    架构设计：
    1. 共享Patch嵌入层（所有状态共用）
    2. 共享Transformer编码器（所有状态共用）
    3. 状态自适应Adapter（每个状态有独立的适配器）
    4. 专家头（每个状态有独立的分类头）

    优点：
    - 参数效率高（共享部分占大头）
    - 状态之间可以迁移知识（共享编码器学到通用表示）
    - 每个状态仍保持一定的专用性（Adapter + Head）
    """

    def __init__(self, input_dim, seq_len, patch_len, stride,
                 d_model=64, n_heads=4, num_layers=2, dropout=0.1,
                 n_states=4, adapter_dim=32):
        super().__init__()

        self.n_states = n_states
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (seq_len - patch_len) // stride + 1

        # === 共享部分（所有状态共用） ===
        # 1. Patch嵌入层
        self.input_proj = nn.Linear(patch_len * input_dim, d_model)

        # 2. Transformer编码器层
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        # === 状态自适应部分（每个状态独立） ===
        # 3. State Adapter（轻量级适配器）
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, adapter_dim),
                nn.GELU(),
                nn.Linear(adapter_dim, d_model)
            )
            for _ in range(n_states)
        ])

        # 4. 专家头
        self.heads = nn.ModuleList([
            nn.Linear(d_model, 2)  # 输出mean和log_var
            for _ in range(n_states)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, state_labels=None):
        """
        x: [batch, seq_len, input_dim]
        state_labels: [batch] 每个样本的状态标签
        """
        batch, seq_len, input_dim = x.shape

        # === 1. Patch嵌入（共享） ===
        patches = []
        for i in range(0, seq_len - self.patch_len + 1, self.stride):
            patch = x[:, i:i + self.patch_len, :]
            patch = patch.reshape(batch, -1)
            patches.append(patch)
        patches = torch.stack(patches, dim=1)

        # 线性投影
        h = self.input_proj(patches)
        h = self.dropout(h)

        # === 2. Transformer编码器（共享） ===
        for layer in self.encoder_layers:
            h = layer(h)

        # 全局平均池化
        h = h.mean(dim=1)  # [batch, d_model]

        # === 3. 状态自适应路由 ===
        if state_labels is not None:
            # 训练/推理时使用给定的状态标签
            outputs = []
            for i in range(batch):
                state = state_labels[i].item()
                # 应用适配器
                adapted = self.adapters[state](h[i:i + 1]) + h[i:i + 1]  # 残差连接
                # 专家头
                out = self.heads[state](adapted)
                outputs.append(out)
            outputs = torch.cat(outputs, dim=0)
        else:
            # 测试时：使用所有状态的加权平均
            # 这里简化：使用状态0
            adapted = self.adapters[0](h) + h
            outputs = self.heads[0](adapted)

        # 输出mean和log_var
        mean = torch.sigmoid(outputs[:, 0])
        log_var = outputs[:, 1]
        return mean, log_var


# ===== 轻量级版本（用于消融实验对比） =====
class VanillaPatchTST(nn.Module):
    """普通PatchTST（无状态自适应）"""

    def __init__(self, input_dim, seq_len, patch_len, stride,
                 d_model=64, n_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (seq_len - patch_len) // stride + 1

        self.input_proj = nn.Linear(patch_len * input_dim, d_model)
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model * 4, dropout=dropout,
                activation='gelu', batch_first=True
            )
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, seq_len, input_dim = x.shape
        patches = []
        for i in range(0, seq_len - self.patch_len + 1, self.stride):
            patch = x[:, i:i + self.patch_len, :]
            patch = patch.reshape(batch, -1)
            patches.append(patch)
        patches = torch.stack(patches, dim=1)

        h = self.input_proj(patches)
        h = self.dropout(h)
        for layer in self.encoder_layers:
            h = layer(h)
        h = h.mean(dim=1)
        out = self.head(h)
        return torch.sigmoid(out[:, 0]), out[:, 1]


def main():
    print("=" * 60)
    print("step59：分层专家PatchTST架构定义")
    print("=" * 60)

    # 测试模型
    input_dim = 15
    seq_len = 20
    patch_len = 5
    stride = 1

    # 分层专家模型
    model = HierarchicalExpertPatchTST(
        input_dim=input_dim, seq_len=seq_len,
        patch_len=patch_len, stride=stride,
        d_model=64, n_heads=4, num_layers=2,
        n_states=4
    )

    print("✅ 分层专家PatchTST模型定义完成")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试前向传播
    x = torch.randn(2, seq_len, input_dim)
    state_labels = torch.tensor([0, 1])
    mean, log_var = model(x, state_labels)
    print(f"  输入: {x.shape} -> 输出: mean={mean.shape}, log_var={log_var.shape}")

    print("\n📌 使用说明:")
    print("  在训练循环中，使用 state_labels 参数传入每个样本的状态标签")
    print("  状态标签可以从 step58 的对比学习结果中获得")
    print("  VS 原方案: 参数效率更高，支持状态间知识迁移")
    print("\n✅ step59 完成！")


if __name__ == "__main__":
    main()