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

设备策略（自动适配，无需手动配置）：
  item_emb / user_emb_table 体积巨大（百万级物品），当其大小超过可用显存的
  40% 时自动留在 CPU；其余参数随 .to(device) 正常迁移。
  所有前向计算会在正确设备间自动搬运，对调用方透明。
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
    from recbole.utils import InputType
    _HAS_RECBOLE = True
except ImportError:
    _Base = nn.Module
    _HAS_RECBOLE = False


def _cfg_get(config, key, default):
    """
    兼容 RecBole Config 对象和普通字典的安全取值工具
    """
    try:
        # 优先尝试 [] 方式取值（支持 RecBole Config）
        val = config[key]
        # 若取到 None，返回默认值
        return default if val is None else val
    except (KeyError, AttributeError):
        # 找不到键或对象不支持 []，返回默认值
        return default

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
        residual = x
        x_norm   = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = residual + self.dropout(attn_out)
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

    input_type = InputType.PAIRWISE if _HAS_RECBOLE else None
    def __init__(self, config, dataset=None):
        if dataset is None:
        # 造空对象占位，不依赖recbole导入
            dataset = object()
        super().__init__(config, dataset)

        # ── 从 config 读超参数 ──────────────────────────
        self.n_users     = _cfg_get(config, "n_users",     987994)
        self.n_items     = _cfg_get(config, "n_items",     4162024)
        d                = _cfg_get(config, "embed_dim",   64)
        self.embed_dim   = d
        n_layers         = _cfg_get(config, "n_layers",    2)
        n_heads          = _cfg_get(config, "n_heads",     2)
        max_seq_len      = _cfg_get(config, "max_seq_len", 50 )
        graph_layers     = _cfg_get(config, "graph_layers",2)
        self.lambda1     = _cfg_get(config, "lambda1",     0.1)
        self.lambda2     = _cfg_get(config, "lambda2",     0.1)
        self.temperature = _cfg_get(config, "temperature", 0.07)
        dropout          = _cfg_get(config, "dropout",     0.1)
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

        # ── 大型嵌入表设备策略（自动检测，无需手动配置）──
        # item_emb / user_emb_table 体积 = n_items * embed_dim * 4 bytes
        # 若超过可用显存的 40%，自动留在 CPU，其余模型正常放 GPU
        # 在 CPU 上训练时此逻辑不触发，行为与原来完全一致
        emb_mb = (self.n_items + 1) * d * 4 / 1e6
        use_cpu_emb = False
        if torch.cuda.is_available():
            free_mb = torch.cuda.mem_get_info()[0] / 1e6
            if emb_mb > free_mb * 0.4:
                use_cpu_emb = True
                print(f"[MBGCLSASRec] item_emb ({emb_mb:.0f}MB) 超过可用显存40%"
                      f"（{free_mb:.0f}MB），自动留在 CPU。")
        self._large_emb_on_cpu = use_cpu_emb
        if use_cpu_emb:
            self.behavior_emb.item_emb = self.behavior_emb.item_emb.cpu()
            self.user_emb_table = self.user_emb_table.cpu()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ──────────────────────────────────────────────────────────────────
    # 内部工具：获取模型主体所在设备（attn_layers 一定在主设备上）
    # ──────────────────────────────────────────────────────────────────

    @property
    def _main_device(self) -> torch.device:
        return next(self.attn_layers.parameters()).device

    def _to_main(self, t: torch.Tensor) -> torch.Tensor:
        """把张量搬到模型主体设备。"""
        return t.to(self._main_device)

    def _item_lookup(self, ids: torch.LongTensor) -> torch.Tensor:
        """
        查 item_emb，自动处理 CPU/GPU 分离。
        ids 可以在任意设备，返回结果与模型主体同设备。
        """
        emb = self.behavior_emb.item_emb(ids.cpu() if self._large_emb_on_cpu else ids)
        return self._to_main(emb)

    # ──────────────────────────────────────────────────────────────────
    # 公开接口：加载图 + 预计算图卷积缓存
    # ──────────────────────────────────────────────────────────────────

    def load_graphs(self, graphs: list):
        """
        加载 A同学的三张图。训练前调用一次。

        示例
        ----
        import scipy.sparse as sp
        graphs = [
            sp.load_npz('data/processed/B/graph_pv.npz'),
            sp.load_npz('data/processed/B/graph_cart.npz'),
            sp.load_npz('data/processed/B/graph_buy.npz'),
        ]
        model.load_graphs(graphs)
        model.update_graph_emb()   # 紧接着调用，预计算 GCN 缓存
        """
        self.graph_conv.load_graphs(graphs)

    @torch.no_grad()
    def update_graph_emb(self):
        """
        预计算图卷积增强嵌入并缓存，避免每个 batch 重复计算。
        每个 epoch 开始前调用一次即可。

        图卷积在 CPU 上计算（图本身是 CPU sparse tensor），
        结果缓存到 CPU，forward 时按需搬到主设备。
        """
        if not self.graph_conv._graphs_loaded:
            return

        # 嵌入表和图卷积都在 CPU 上，统一在 CPU 计算
        graph_conv_cpu = self.graph_conv.cpu()
        all_user_emb   = self.user_emb_table.weight[1:].cpu()      # [n_users, d]
        all_item_emb   = self.behavior_emb.item_emb.weight[1:].cpu()  # [n_items, d]

        u_enh, i_enh = graph_conv_cpu(all_user_emb, all_item_emb)

        # 补 padding 行（index 0），对齐 item_id 从 1 开始
        pad = torch.zeros(1, i_enh.size(-1), dtype=i_enh.dtype)
        self._i_enh_cache = torch.cat([pad, i_enh], dim=0).detach()  # [n_items+1, d] CPU
        self._u_enh_cache = u_enh.detach()                            # [n_users, d]   CPU
        print("[MBGCLSASRec] 图卷积缓存已更新。")

    # ──────────────────────────────────────────────────────────────────
    # 内部：编码序列 → 返回所有位置的隐状态
    # ──────────────────────────────────────────────────────────────────

    def _encode_sequence(
        self,
        item_seq:      torch.LongTensor,          # [B, L]
        behavior_seq:  torch.LongTensor,          # [B, L]  0=pv,1=cart,2=buy
        seq_len:       torch.LongTensor,          # [B]     实际有效长度
        weights:       torch.FloatTensor = None,  # [B, L]
        user_ids:      torch.LongTensor  = None,  # [B]
    ) -> torch.Tensor:
        """返回 [B, L, d]：序列中每个位置的隐状态。"""
        B, L = item_seq.shape

        # ── 行为感知嵌入 ───────────────────────────────
        # item_emb 可能在 CPU，先在 CPU 取嵌入再搬到主设备
        if self._large_emb_on_cpu:
            seq_emb = self.behavior_emb(item_seq.cpu(), behavior_seq.cpu(),
                                        weights.cpu() if weights is not None else None)
            seq_emb = self._to_main(seq_emb)                        # [B, L, d]
        else:
            seq_emb = self.behavior_emb(item_seq, behavior_seq, weights)  # [B, L, d]

        # ── 图卷积增强（查缓存，不重新计算）─────────
        if self.graph_conv._graphs_loaded and hasattr(self, "_i_enh_cache"):
            # 缓存在 CPU，查完再搬到主设备
            i_enh_seq = F.embedding(item_seq.cpu(), self._i_enh_cache)
            i_enh_seq = self._to_main(i_enh_seq)                    # [B, L, d]

            gate    = self.graph_gate(torch.cat([seq_emb, i_enh_seq], dim=-1))
            seq_emb = seq_emb + gate * i_enh_seq                    # [B, L, d]

        # ── 位置编码 ───────────────────────────────────
        positions = torch.arange(1, L + 1, device=self._main_device)
        seq_emb   = seq_emb + self.pos_emb(positions.unsqueeze(0))  # [B, L, d]

        # ── SASRec 因果自注意力 ────────────────────────
        mask = self.causal_mask[:L, :L]
        h    = seq_emb
        for layer in self.attn_layers:
            h = layer(h, attn_mask=mask)

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
        h   = self._encode_sequence(item_seq, behavior_seq, seq_len, weights, user_ids)
        idx = (seq_len.to(self._main_device) - 1).clamp(0, h.size(1) - 1)
        return h[torch.arange(h.size(0), device=self._main_device), idx]  # [B, d]

    # ──────────────────────────────────────────────────────────────────
    # 公开接口：D同学用于提取隐状态（漂移分析）
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def get_hidden_state(self, interaction: dict) -> torch.Tensor:
        """
        提取用户最终隐状态，供 D同学做兴趣漂移分析。

        参数
        ----
        interaction : dict，至少包含：
            "item_seq"       : LongTensor [B, L]
            "behavior_seq"   : LongTensor [B, L]
            "item_seq_len"   : LongTensor [B]
            "weights"        : FloatTensor [B, L]（可选）
            "user_id"        : LongTensor [B]（可选）

        返回
        ----
        h_last : FloatTensor [B, embed_dim]

        用法示例（D同学）
        ----
        model.load_state_dict(torch.load('best_model.pth'))
        model.eval()
        h = model.get_hidden_state(interaction)   # [B, d]
        drift = torch.norm(h[1:] - h[:-1], dim=-1)  # 相邻时刻的漂移距离
        """
        self.eval()
        item_seq     = interaction["item_id_list"]
        behavior_seq = interaction["behavior_list"].long()
        seq_len      = interaction["item_length"]
        #weights      = interaction.get("weight_list")
        #user_ids     = interaction.get("user_id")
        weights      = interaction["weight_list"] if "weight_list" in interaction else None
        user_ids     = interaction["user_id"] if "user_id" in interaction else None
        return self._get_last_hidden(item_seq, behavior_seq, seq_len, weights, user_ids)

    # ──────────────────────────────────────────────────────────────────
    # 训练接口
    # ──────────────────────────────────────────────────────────────────

    def calculate_loss(self, interaction: dict) -> torch.Tensor:
        """
        计算三路联合损失。
        """
        item_seq     = interaction["item_id_list"]
        behavior_seq = interaction["behavior_list"].long()
        seq_len      = interaction["item_length"]
        pos_ids      = interaction["item_id"]
        neg_ids      = interaction["neg_item_id"]
        #weights      = interaction.get("weight_list")
        #user_ids     = interaction.get("user_id")
        weights      = interaction["weight_list"] if "weight_list" in interaction else None
        user_ids     = interaction["user_id"] if "user_id" in interaction else None
        # 强视图：完整序列（含 buy/cart）
        h_strong = self._get_last_hidden(item_seq, behavior_seq, seq_len, weights, user_ids)

        # 弱视图：仅保留 pv 行为
        pv_mask = (behavior_seq == 0) & (item_seq != 0)
        pv_seq  = item_seq * pv_mask.long()
        pv_beh  = torch.zeros_like(behavior_seq)
        pv_len  = pv_mask.sum(dim=1).clamp(min=1)
        h_weak  = self._get_last_hidden(pv_seq, pv_beh, pv_len, None, user_ids)

        # 正负样本打分
        pos_emb    = self._item_lookup(pos_ids)
        neg_emb    = self._item_lookup(neg_ids)
        pos_scores = (h_strong * pos_emb).sum(-1)
        neg_scores = (h_strong * neg_emb).sum(-1)

        # 需求闭合分支：二维 [B,L] → 取最后有效位置转为一维 [B]
        closure_logits = self.closure_head(h_strong).squeeze(-1)
        #closure_labels = interaction.get("closure_list")
        closure_labels = interaction["closure_list"] if "closure_list" in interaction else None
        if closure_labels is not None:
            # 取每条序列最后一个有效位置的值
            idx = (seq_len - 1).clamp(0, closure_labels.size(1) - 1)
            closure_labels = closure_labels[torch.arange(closure_labels.size(0)), idx]
            closure_labels = closure_labels.to(self._main_device)

        loss_dict = combined_loss(
            pos_scores     = pos_scores,
            neg_scores     = neg_scores,
            closure_logits = closure_logits,
            closure_labels = closure_labels,
            h_strong       = h_strong,
            h_weak         = h_weak,
            lambda1        = self.lambda1,
            lambda2        = self.lambda2,
            temperature    = self.temperature,
        )
# 取出字典里的总损失张量（根据你 combined_loss 里的键名，一般是 total_loss）
        final_loss = loss_dict["loss"] 
        return final_loss

    # ──────────────────────────────────────────────────────────────────
    # 评估接口（RecBole 兼容）
    # ──────────────────────────────────────────────────────────────────

    def predict(self, interaction: dict) -> torch.Tensor:
        """
        对指定物品打分。返回 [B] 分数（值越高越推荐）。
        """
        item_seq     = interaction["item_id_list"]
        behavior_seq = interaction["behavior_list"].long()
        seq_len      = interaction["item_length"]
        target_ids   = interaction["item_id"]
        #weights      = interaction.get("weight_list")
        #user_ids     = interaction.get("user_id")
        weights      = interaction["weight_list"] if "weight_list" in interaction else None
        user_ids     = interaction["user_id"] if "user_id" in interaction else None
        h_last  = self._get_last_hidden(item_seq, behavior_seq, seq_len, weights, user_ids)
        tgt_emb = self._item_lookup(target_ids)
        return (h_last * tgt_emb).sum(-1)                       # [B]

    def full_sort_predict(self, interaction: dict) -> torch.Tensor:
        """
        对所有物品打分（RecBole 评估 Hit/NDCG 用）。返回 [B, n_items]。
        """
        item_seq     = interaction["item_id_list"]
        behavior_seq = interaction["behavior_list"].long()
        seq_len      = interaction["item_length"]
        #weights      = interaction.get("weight_list")
        #user_ids     = interaction.get("user_id")
        weights      = interaction["weight_list"] if "weight_list" in interaction else None
        user_ids     = interaction["user_id"] if "user_id" in interaction else None
        h_last = self._get_last_hidden(item_seq, behavior_seq, seq_len, weights, user_ids)

        # 兼容 CPU/GPU 分离的物品嵌入表
        if self._large_emb_on_cpu:
            all_item_emb = self.behavior_emb.item_emb.weight[1:].cpu()
        else:
            all_item_emb = self.behavior_emb.item_emb.weight[1:]
        all_item_emb = self._to_main(all_item_emb)
        return h_last @ all_item_emb.T                            # [B, n_items]