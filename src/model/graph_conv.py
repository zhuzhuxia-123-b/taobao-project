"""
graph_conv.py — 多行为图卷积模块
=====================================
核心思路：
  "和你有相似购买记录的人，还买了什么" → 推荐给你

  对 pv / cart / buy 三张用户-物品图分别做 LightGCN 风格的图卷积，
  让低活跃用户也能借助"相似用户"的行为补充表示。

依赖 A同学的文件：
  graph_pv.npz / graph_cart.npz / graph_buy.npz
  格式：scipy CSR 稀疏矩阵，shape = (n_users, n_items)

用法：
  conv = MultiBehaviorGraphConv(n_users, n_items, embed_dim=64)
  graphs = [load_npz('graph_pv.npz'), load_npz('graph_cart.npz'), load_npz('graph_buy.npz')]
  conv.load_graphs(graphs)           # 训练前调用一次
  u_out, i_out = conv(user_emb, item_emb)
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


class MultiBehaviorGraphConv(nn.Module):
    """
    三行为图卷积 + 加权融合。

    参数
    ----
    n_users   : 用户数量（与图的行数一致）
    n_items   : 物品数量（与图的列数一致）
    embed_dim : 嵌入维度
    n_layers  : LightGCN 传播层数，建议 2~3
    dropout   : Dropout 概率
    """

    BEHAVIOR_NAMES = ["pv", "cart", "buy"]

    def __init__(
        self,
        n_users:   int,
        n_items:   int,
        embed_dim: int,
        n_layers:  int = 2,
        dropout:   float = 0.1,
    ):
        super().__init__()
        self.n_users   = n_users
        self.n_items   = n_items
        self.embed_dim = embed_dim
        self.n_layers  = n_layers

        # 三种行为的融合权重（可学习），初始值体现 buy > cart > pv 先验
        self.behavior_weight = nn.Parameter(torch.tensor([0.2, 0.3, 0.5]))

        # 各层输出的融合权重（LightGCN 原论文用均值，这里改为可学习）
        self.layer_weight = nn.Parameter(torch.ones(n_layers + 1) / (n_layers + 1))

        self.dropout = nn.Dropout(dropout)
        self._graphs_loaded = False

    # ------------------------------------------------------------------
    # 图加载（训练前调用一次，之后图矩阵会注册到 buffer 跟随 .to(device)）
    # ------------------------------------------------------------------

    def load_graphs(self, graphs: list):
        """
        加载三张稀疏图并完成 LightGCN 归一化。

        参数
        ----
        graphs : list of scipy.sparse.csr_matrix
                 [graph_pv, graph_cart, graph_buy]，shape = (n_users, n_items)
        """
        assert len(graphs) == 3, "需要传入 pv / cart / buy 三张图"
        for g, name in zip(graphs, self.BEHAVIOR_NAMES):
            norm = self._build_norm_graph(g.tocsr(), self.n_users, self.n_items)
            self.register_buffer(f"graph_{name}", norm)
        self._graphs_loaded = True
        print(f"[GraphConv] 图加载完成，n_users={self.n_users}, n_items={self.n_items}")

    @staticmethod
    def _to_rect(adj: sp.spmatrix, n_users: int, n_items: int) -> sp.csr_matrix:
        """
        把 A同学生成的方阵（user_id 为行，item_id 为列，共享 ID 空间）
        转成标准的 (n_users, n_items) 矩形 CSR 图。
        超出范围的边直接丢弃。
        """
        coo = adj.tocoo()
        rows, cols, data = coo.row, coo.col, coo.data

        # 保留 user_id < n_users 且 item_id < n_items 的边
        mask = (rows < n_users) & (cols < n_items)
        return sp.csr_matrix(
            (data[mask], (rows[mask], cols[mask])),
            shape=(n_users, n_items),
        )

    @staticmethod
    def _build_norm_graph(adj: sp.csr_matrix, n_users: int, n_items: int) -> torch.Tensor:
        # 统一尺寸
        if adj.shape != (n_users, n_items):
            adj = adj.tocoo()
            adj = sp.csr_matrix(
                (adj.data, (adj.row, adj.col)),
                shape=(n_users, n_items)
            )

        n = n_users + n_items
        adj = adj.tocoo().astype(np.float32)

        # 手动拼二部图，避免 sp.bmat 的内存峰值
        # 上半：user->item 边，行偏移0，列偏移n_users
        # 下半：item->user 边，行偏移n_users，列偏移0
        rows = np.concatenate([adj.row, adj.col + n_users])
        cols = np.concatenate([adj.col + n_users, adj.row])
        data = np.concatenate([adj.data, adj.data])

        # 度归一化
        deg = np.zeros(n, dtype=np.float32)
        np.add.at(deg, rows, data)
        deg[deg == 0] = 1.0
        d_inv_sqrt = deg ** -0.5

        # 归一化边权重
        data = data * d_inv_sqrt[rows] * d_inv_sqrt[cols]

        indices = torch.from_numpy(np.vstack([rows, cols]).astype(np.int64))
        values = torch.from_numpy(data)
        return torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
    # ------------------------------------------------------------------
    # 前向传播
    # ------------------------------------------------------------------

    def forward(
        self,
        user_emb: torch.Tensor,  # [n_users, d]
        item_emb: torch.Tensor,  # [n_items, d]
    ):
        """
        输入原始用户/物品嵌入，输出图增强后的嵌入。

        返回
        ----
        user_out : [n_users, d]
        item_out : [n_items, d]
        """
        if not self._graphs_loaded:
            # 图尚未加载时直接返回原始嵌入（方便单元测试和 debug）
            return user_emb, item_emb

        beh_w  = torch.softmax(self.behavior_weight, dim=0)   # [3]，和为 1
        lyr_w  = torch.softmax(self.layer_weight,    dim=0)   # [K+1]，和为 1

        user_agg = torch.zeros_like(user_emb)
        item_agg = torch.zeros_like(item_emb)

        for b_idx, name in enumerate(self.BEHAVIOR_NAMES):
            graph = getattr(self, f"graph_{name}")             # (n_u+n_i, n_u+n_i) sparse

            x = torch.cat([user_emb, item_emb], dim=0)        # [n_u+n_i, d]
            layers = [x]

            for _ in range(self.n_layers):
                x = torch.sparse.mm(graph, x)                 # [n_u+n_i, d]
                x = self.dropout(x)
                layers.append(x)

            # 层间加权求和
            stacked = torch.stack(layers, dim=0)               # [K+1, n_u+n_i, d]
            x_mean  = (stacked * lyr_w.view(-1, 1, 1)).sum(0)  # [n_u+n_i, d]

            user_agg = user_agg + beh_w[b_idx] * x_mean[: self.n_users]
            item_agg = item_agg + beh_w[b_idx] * x_mean[self.n_users :]

        return user_agg, item_agg