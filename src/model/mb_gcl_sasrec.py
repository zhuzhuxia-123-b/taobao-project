"""
mb_gcl_sasrec.py — 主模型 MB-GCL-SASRec
==========================================
把四个组件串联起来：

  [序列输入]
      ↓ BehaviorAwareEmbedding（感知点击/加购/购买的强度差异）
      ↓ + GraphConv 增强的物品嵌入（借助"相似用户"补充信息）
      ↓ SASRec 自注意力层（建模序列中的兴趣演化）
      ↓ h_last（最终用户表示）
      ↓ 评分 / 损失

对外接口：
  calculate_loss(interaction) → loss dict         （训练用）
  predict(interaction)        → [B] 分数           （评估用）
  full_sort_predict(interaction) → [B, n_items]   （全量排序，RecBole 评估用）
  get_hidden_state(interaction)  → [B, d]         （D同学做漂移分析用）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

from .behavior_emb import BehaviorAwareEmbedding
from .graph_conv    import MultiBehaviorGraphConv
from .losses        import combined_loss

# RecBole 兼容：有 RecBole 就继承它的基类，没有就用普通 nn.Module
try:
    from recbole.model.abstract_recommender import SequentialRecommender as _Base
    _HAS_RECBOLE = True
except ImportError:
    _Base = nn.Module
    _HAS_RECBOLE = False


# ─────────────────────────────────────────────
# 辅助模块：SASRec 的单层注意力块
# ─────────────────────────────────────────────

class _SASRecLayer(nn.Module):
    """单层 Transformer 编码器块（Multi-Head Attention + FFN + LayerNorm）。"""

    def __init__(self, embed_dim: int, n_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.norm1   = nn.LayerNorm(embed_dim)
        self.norm2   = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        """
        x         : [B, L, d]
        attn_mask : [L, L]  因果掩码（下三角为 0，上三角为 -inf）

        返回 [B, L, d]
        """
        # 自注意力 + 残差
        residual = x
        x_norm   = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = residual + self.dropout(attn_out)

        # FFN + 残差
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ─────────────────────────────────────────────
# 主模型
# ─────────────────────────────────────────────

class MBGCLSASRec(_Base):
    """
    Multi-Behavior Graph Contrastive Learning + SASRec。

    参数（直接传 dict 或 RecBole config 均可）
    ----
    n_users          : 用户数量
    n_items          : 物品数量
    embed_dim        : 嵌入维度，默认 64
    n_layers         : SASRec 层数，默认 2
    n_heads          : 注意力头数，默认 2
    max_seq_len      : 最大序列长度，默认 50
    graph_layers     : 图卷积层数，默认 2
    lambda1          : CL 损失权重，默认 0.1
    lambda2          : 需求闭合损失权重，默认 0.1
    temperature      : CL 温度参数 τ，默认 0.07
    dropout          : Dropout 概率，默认 0.1
    """

    def __init__(self, config: dict, dataset=None):
        if _HAS_RECBOLE and dataset is not None:
            super().__init__(config, dataset)
        else:
            super().__init__()

        # ── 从 config 读超参数 ──────────────────────────
        self.n_users = config.get("n_users",    1018012)
        self.n_items     = config.get("n_items",     5163071)
        d                = config.get("embed_dim",   64)
        self.embed_dim   = d
        n_layers         = config.get("n_layers",    2)
        n_heads          = config.get("n_heads",     2)
        max_seq_len      = config.get("max_seq_len", 50)
        graph_layers     = config.get("graph_layers",2)
        self.lambda1     = config.get("lambda1",     0.1)
        self.lambda2     = config.get("lambda2",     0.1)
        self.temperature = config.get("temperature", 0.07)
        dropout          = config.get("dropout",     0.1)
        self.max_seq_len = max_seq_len

        # ── 组件 1：行为感知嵌入 ───────────────────────
        self.behavior_emb = BehaviorAwareEmbedding(
            n_items=self.n_items, embed_dim=d, dropout=dropout
        )

        # ── 组件 2：多行为图卷积 ───────────────────────
        self.graph_conv = MultiBehaviorGraphConv(
            n_users=self.n_users, n_items=self.n_items,
            embed_dim=d, n_layers=graph_layers, dropout=dropout
        )

        # 图卷积专用的用户嵌入（全量，与序列嵌入分开）
        self.user_emb_table = nn.Embedding(self.n_users + 1, d, padding_idx=0)

        # 图卷积增强后的物品嵌入融合门控
        self.graph_gate = nn.Sequential(nn.Linear(d * 2, d), nn.Sigmoid())

        # ── 组件 3：位置编码 ───────────────────────────
        self.pos_emb = nn.Embedding(max_seq_len + 1, d)

        # ── 组件 4：SASRec 注意力层 ────────────────────
        self.attn_layers = nn.ModuleList([
            _SASRecLayer(d, n_heads, d * 4, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d)

        # ── 需求闭合预测头（BCE 损失用）──────────────
        self.closure_head = nn.Linear(d, 1)

        # ── 因果掩码（下三角，防止看到未来）─────────
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.full((max_seq_len, max_seq_len), float("-inf")), diagonal=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ──────────────────────────────────────────────────────────────────
    # 公开接口：加载图（训练前调用一次）
    # ──────────────────────────────────────────────────────────────────

    def load_graphs(self, graphs: list):
        self.graph_conv.load_graphs(graphs)

    # ──────────────────────────────────────────────────────────────────
    # 内部：编码序列 → 返回所有层的隐状态
    # ──────────────────────────────────────────────────────────────────

    def _encode_sequence(
        self,
        item_seq:      torch.LongTensor,    # [B, L]
        behavior_seq:  torch.LongTensor,    # [B, L]  0=pv,1=cart,2=buy
        seq_len:       torch.LongTensor,    # [B]     实际有效长度
        weights:       torch.FloatTensor = None,  # [B, L]
        user_ids:      torch.LongTensor = None,   # [B]
    ) -> torch.Tensor:
        """
        返回 [B, L, d]：序列中每个位置的隐状态。
        """
        B, L = item_seq.shape

        # ── 行为感知嵌入 ───────────────────────────────
        seq_emb = self.behavior_emb(item_seq, behavior_seq, weights)  # [B, L, d]

        # ── 图卷积增强 ─────────────────────────────────
        if self.graph_conv._graphs_loaded and user_ids is not None:
            # 全量用户/物品嵌入送入图卷积
            all_user_emb = self.user_emb_table.weight[1:]   # [n_users, d]
            all_item_emb = self.behavior_emb.item_emb.weight[1:]  # [n_items, d]

            u_enh, i_enh = self.graph_conv(all_user_emb, all_item_emb)
            # [n_users, d]  [n_items, d]

            # 取出当前 batch 中物品的图增强嵌入
            i_enh = torch.cat(
                [torch.zeros(1, i_enh.size(-1), device=i_enh.device), i_enh], dim=0
            )  # [n_items+1, d]
            i_enh_seq = F.embedding(item_seq, i_enh)  # [B, L, d]

            # 门控融合：原始嵌入 + 图增强嵌入
            gate = self.graph_gate(
                torch.cat([seq_emb, i_enh_seq], dim=-1)
            )                                                # [B, L, d]
            seq_emb = seq_emb + gate * i_enh_seq            # [B, L, d]

        # ── 位置编码 ───────────────────────────────────
        positions = torch.arange(1, L + 1, device=item_seq.device)  # [L]
        seq_emb   = seq_emb + self.pos_emb(positions.unsqueeze(0))  # [B, L, d]

        # ── SASRec 因果自注意力 ────────────────────────
        mask = self.causal_mask[:L, :L]                     # [L, L]
        h    = seq_emb
        for layer in self.attn_layers:
            h = layer(h, attn_mask=mask)                    # [B, L, d]

        h = self.final_norm(h)
        return h  # [B, L, d]

    def _get_last_hidden(
        self,
        item_seq:     torch.LongTensor,
        behavior_seq: torch.LongTensor,
        seq_len:      torch.LongTensor,
        weights:      torch.FloatTensor = None,
        user_ids:     torch.LongTensor  = None,
    ) -> torch.Tensor:
        """取出每个样本最后一个有效位置的隐状态 → [B, d]。"""
        h = self._encode_sequence(item_seq, behavior_seq, seq_len, weights, user_ids)
        # seq_len 记录有效长度，取对应位置（1-indexed → 0-indexed）
        idx = (seq_len - 1).clamp(0, h.size(1) - 1)       # [B]
        h_last = h[torch.arange(h.size(0), device=h.device), idx]  # [B, d]
        return h_last

    # ──────────────────────────────────────────────────────────────────
    # 公开接口：D同学用于提取隐状态（漂移分析）
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def get_hidden_state(self, interaction: dict) -> torch.Tensor:
        """
        提取用户最终隐状态，供 D同学做兴趣漂移分析。
        """
        self.eval()
        return self._get_last_hidden(
            item_seq     = interaction["item_seq"],
            behavior_seq = interaction["behavior_seq"],
            seq_len      = interaction["item_seq_len"],
            weights      = interaction.get("weights"),
            user_ids     = interaction.get("user_id"),
        )

    # ──────────────────────────────────────────────────────────────────
    # 训练接口
    # ──────────────────────────────────────────────────────────────────

    def calculate_loss(self, interaction: dict) -> dict:
        """
        计算三路联合损失
        """
        item_seq     = interaction["item_seq"]
        behavior_seq = interaction["behavior_seq"]
        seq_len      = interaction["item_seq_len"]
        pos_ids      = interaction["pos_item_id"]
        neg_ids      = interaction["neg_item_id"]
        weights      = interaction.get("weights")
        user_ids     = interaction.get("user_id")

        # 强视图：完整序列（含 buy/cart），即 item_seq 本身
        h_strong = self._get_last_hidden(item_seq, behavior_seq, seq_len, weights, user_ids)

        # 弱视图：仅保留 pv 行为（behavior_type == 0），其他位置置为 padding
        pv_mask = (behavior_seq == 0) & (item_seq != 0)
        pv_seq    = item_seq * pv_mask.long()          # 非 pv 位置变为 0
        pv_beh    = torch.zeros_like(behavior_seq)
        pv_len    = pv_mask.sum(dim=1).clamp(min=1)   # 至少 1
        h_weak    = self._get_last_hidden(pv_seq, pv_beh, pv_len, None, user_ids)

        # 评分（内积）
        all_item_emb = self.behavior_emb.item_emb.weight  # [n_items+1, d]
        pos_emb      = all_item_emb[pos_ids]               # [B, d]
        neg_emb      = all_item_emb[neg_ids]               # [B, d]
        pos_scores   = (h_strong * pos_emb).sum(-1)        # [B]
        neg_scores   = (h_strong * neg_emb).sum(-1)        # [B]

        # 需求闭合 logit
        closure_logits = self.closure_head(h_strong).squeeze(-1)  # [B]
        closure_labels = interaction.get("closure_label")

        return combined_loss(
            pos_scores      = pos_scores,
            neg_scores      = neg_scores,
            closure_logits  = closure_logits,
            closure_labels  = closure_labels,
            h_strong        = h_strong,
            h_weak          = h_weak,
            lambda1         = self.lambda1,
            lambda2         = self.lambda2,
            temperature     = self.temperature,
        )

    # ──────────────────────────────────────────────────────────────────
    # 评估接口（RecBole 兼容）
    # ──────────────────────────────────────────────────────────────────

    def predict(self, interaction: dict) -> torch.Tensor:
        """
        对指定物品打分。

        返回 [B] 分数（值越高越推荐）
        """
        item_seq     = interaction["item_seq"]
        behavior_seq = interaction["behavior_seq"]
        seq_len      = interaction["item_seq_len"]
        target_ids   = interaction["item_id"]
        weights      = interaction.get("weights")
        user_ids     = interaction.get("user_id")

        h_last   = self._get_last_hidden(item_seq, behavior_seq, seq_len, weights, user_ids)
        tgt_emb  = self.behavior_emb.item_emb(target_ids)  # [B, d]
        scores   = (h_last * tgt_emb).sum(-1)              # [B]
        return scores

    def full_sort_predict(self, interaction: dict) -> torch.Tensor:
        """
        对所有物品打分（RecBole 评估 Hit/NDCG 用）。

        返回 [B, n_items] 分数矩阵
        """
        item_seq     = interaction["item_seq"]
        behavior_seq = interaction["behavior_seq"]
        seq_len      = interaction["item_seq_len"]
        weights      = interaction.get("weights")
        user_ids     = interaction.get("user_id")

        h_last       = self._get_last_hidden(item_seq, behavior_seq, seq_len, weights, user_ids)
        all_item_emb = self.behavior_emb.item_emb.weight[1:]  # [n_items, d]（去掉 padding）
        scores       = h_last @ all_item_emb.T                # [B, n_items]
        return scores