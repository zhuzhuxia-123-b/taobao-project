"""
InfoNCE对比学习损失
强视图：含buy/cart的完整序列
弱视图：仅pv行为序列
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    InfoNCE Loss

    用法（B同学接入时）：
        cl_loss_fn = ContrastiveLoss(temperature=0.2)
        loss = cl_loss_fn(strong_view_emb, weak_view_emb)

    Args:
        temperature: 温度系数τ，默认0.2
    """

    def __init__(self, temperature: float = 0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, strong_view: torch.Tensor, weak_view: torch.Tensor) -> torch.Tensor:
        """
        Args:
            strong_view: [B, D] 含buy/cart的序列表示
            weak_view:   [B, D] 仅pv的序列表示
        Returns:
            scalar loss
        """
        z_s = F.normalize(strong_view, dim=-1)
        z_w = F.normalize(weak_view, dim=-1)

        sim = torch.matmul(z_s, z_w.T) / self.temperature

        batch_size = strong_view.size(0)
        labels = torch.arange(batch_size, device=strong_view.device)

        loss_s2w = F.cross_entropy(sim, labels)
        loss_w2s = F.cross_entropy(sim.T, labels)

        return (loss_s2w + loss_w2s) / 2