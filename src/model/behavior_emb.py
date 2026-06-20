"""
behavior_emb.py — 行为感知嵌入模块
======================================
同一件商品，用户"随便点了一下（pv）"和"真的买了（buy）"，
对推荐的参考价值完全不同。本模块让模型感知这种差异。

核心思路：
  item_embedding（基础）
      + gate × scale × behavior_offset（行为类型带来的偏置）
      × weight_final（A同学算出的五层权重，可选）
"""

import torch
import torch.nn as nn


class BehaviorAwareEmbedding(nn.Module):
    """
    行为感知物品嵌入。

    参数
    ----
    n_items     : 物品总数（不含 padding）
    embed_dim   : 嵌入维度
    n_behaviors : 行为类型数量，默认 3（pv=0, cart=1, buy=2）
    dropout     : dropout 概率
    """

    def __init__(self, n_items: int, embed_dim: int, n_behaviors: int = 3, dropout: float = 0.1):
        super().__init__()

        # 基础物品嵌入，index 0 是 padding
        self.item_emb = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)

        # 行为类型嵌入（pv / cart / buy 各自一个向量）
        self.behavior_emb = nn.Embedding(n_behaviors, embed_dim)

        # 门控网络：决定行为偏置对基础嵌入的影响强度
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )

        # 可学习的行为重要性标量，初始值体现 buy > cart > pv 先验
        # 训练过程中会根据数据自动微调
        self.behavior_scale = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout    = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_emb.weight,     std=0.02)
        nn.init.normal_(self.behavior_emb.weight, std=0.02)
        # padding 向量强制归零
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def forward(
        self,
        item_ids:       torch.LongTensor,
        behavior_types: torch.LongTensor,
        weights:        torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """
        前向传播。

        参数
        ----
        item_ids       : LongTensor  [B, L]  物品 ID 序列（padding=0 在左侧）
        behavior_types : LongTensor  [B, L]  行为类型（0=pv, 1=cart, 2=buy）
        weights        : FloatTensor [B, L]  A同学的 weight_final（可选）

        返回
        ----
        emb : FloatTensor [B, L, embed_dim]
        """
        item_e = self.item_emb(item_ids)            # [B, L, d]
        beh_e  = self.behavior_emb(behavior_types)  # [B, L, d]

        # 门控融合：item 特征 + 行为特征 → 0~1 之间的门控值
        gate  = self.gate(torch.cat([item_e, beh_e], dim=-1))  # [B, L, d]

        # 行为标量：buy 的偏置贡献更大
        scale = self.behavior_scale[behavior_types].unsqueeze(-1)  # [B, L, 1]

        # 最终嵌入 = 基础物品嵌入 + 门控后的行为偏置
        emb = item_e + gate * scale * beh_e         # [B, L, d]

        # 若提供 A同学的五层权重，作为序列强度权重乘入
        if weights is not None:
            emb = emb * weights.unsqueeze(-1)        # [B, L, d]

        emb = self.layer_norm(emb)
        emb = self.dropout(emb)
        return emb