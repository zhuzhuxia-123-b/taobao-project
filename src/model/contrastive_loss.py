"""
contrastive_loss.py — 对比学习损失模块
========================================

核心思路（InfoNCE Loss）：
  - 强视图 h_strong：含 buy/cart 的完整行为序列表示  [B, d]
  - 弱视图 h_weak  ：仅 pv 行为的序列表示           [B, d]
  - 同一个用户的 (h_strong_i, h_weak_i) 是正样本对
  - batch 内其他用户是负样本
  - 目标：让模型把同一用户的强/弱视图拉近，把不同用户的推远

  from src.model.contrastive_loss import ContrastiveLoss
  loss = ContrastiveLoss(temperature=0.07)(h_strong, h_weak)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    InfoNCE 对比学习损失（双向对称版本）。

    参数
    ----
    temperature : float
        温度系数 τ，控制分布的尖锐程度。
        越小 → 对负样本惩罚越重，训练越难但上限更高。
        建议范围：0.05 ~ 0.2，默认 0.07（与 B同学 config 一致）。

    用法
    ----
    criterion = ContrastiveLoss(temperature=0.07)
    loss = criterion(h_strong, h_weak)   # 返回标量
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        h_strong: torch.Tensor,  # [B, d]  强视图：含 buy/cart 的序列表示
        h_weak:   torch.Tensor,  # [B, d]  弱视图：仅 pv 的序列表示
    ) -> torch.Tensor:
        """
        计算双向 InfoNCE Loss。

        参数
        ----
        h_strong : [B, d]  强视图序列表示（B同学的 _get_last_hidden 输出）
        h_weak   : [B, d]  弱视图序列表示（B同学的 _get_last_hidden 输出）

        返回
        ----
        loss : 标量，两个方向的平均值
        """
        # ── 1. L2 归一化，让内积等价于余弦相似度 ────────────────
        z_strong = F.normalize(h_strong, dim=-1)  # [B, d]
        z_weak   = F.normalize(h_weak,   dim=-1)  # [B, d]

        # ── 2. 计算相似度矩阵 ─────────────────────────────────────
        # sim[i, j] = z_strong[i] · z_weak[j] / τ
        # 对角线 sim[i, i] 是正样本对，其余是负样本
        sim_matrix = torch.matmul(z_strong, z_weak.T) / self.temperature
        # shape: [B, B]

        # ── 3. 构造标签：每个样本的正样本就是同索引位置 ──────────
        B = h_strong.size(0)
        labels = torch.arange(B, device=h_strong.device)  # [0, 1, 2, ..., B-1]

        # ── 4. 双向 InfoNCE ───────────────────────────────────────
        # 方向1：以 strong 为 anchor，weak 为正/负样本
        loss_s2w = F.cross_entropy(sim_matrix, labels)

        # 方向2：以 weak 为 anchor，strong 为正/负样本
        loss_w2s = F.cross_entropy(sim_matrix.T, labels)

        # ── 5. 对称平均 ───────────────────────────────────────────
        loss = (loss_s2w + loss_w2s) / 2.0

        return loss