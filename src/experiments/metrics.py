"""
metrics.py — 自定义评估指标
=============================

包含两个指标：
  1. NeedSatisfactionRate (NSR)：需求满足率（原创指标）
  2. 标准指标包装：NDCG@K, Hit@K（方便在 run_ablation.py 里统一调用）

NSR 定义：
  推荐列表 Top-K 中，closure_label=1（用户最终购买）的商品占比。
  衡量"模型推荐的东西，用户真的会买"的比例。
  值域 [0, 1]，越高越好。

closure_label=1 代表已购买（已闭合），来自 chained_clean.csv，已确认。
"""

import torch
import numpy as np
from typing import List, Union


# ==========================================================================
# 1. NSR：需求满足率
# ==========================================================================

def need_satisfaction_rate(
    recommended_items: Union[torch.Tensor, np.ndarray, List],
    closure_labels:    Union[torch.Tensor, np.ndarray, List],
    k: int = 10,
) -> float:
    """
    计算需求满足率 NSR@K。

    参数
    ----
    recommended_items : [B, K] 或 [K]
        模型推荐的 Top-K 物品 ID 列表。
        如果是二维 [B, K]，则计算 batch 平均值。

    closure_labels : dict 或 array-like
        每个 item_id 对应的 closure_label 值。
        两种传入格式均支持：
          - dict：{item_idx: label}
          - array/tensor：下标即为 item_idx，值为 label
        closure_label=1 代表已购买（闭合）。

    k : int
        取 Top-K，默认 10。

    返回
    ----
    nsr : float，值域 [0, 1]
    """
    # ── 统一转为 numpy ────────────────────────────────────────────
    if isinstance(recommended_items, torch.Tensor):
        recommended_items = recommended_items.cpu().numpy()
    else:
        recommended_items = np.array(recommended_items)

    # ── 处理 closure_labels 的两种格式 ───────────────────────────
    if isinstance(closure_labels, dict):
        label_lookup = closure_labels
    else:
        if isinstance(closure_labels, torch.Tensor):
            closure_labels = closure_labels.cpu().numpy()
        else:
            closure_labels = np.array(closure_labels)
        label_lookup = {i: int(closure_labels[i]) for i in range(len(closure_labels))}

    # ── 计算 NSR ─────────────────────────────────────────────────
    if recommended_items.ndim == 1:
        # 单用户
        top_k_items = recommended_items[:k]
        satisfied = sum(
            1 for item_id in top_k_items
            if label_lookup.get(int(item_id), 0) == 1
        )
        return satisfied / max(len(top_k_items), 1)

    else:
        # batch [B, K]
        batch_nsr = []
        for user_recs in recommended_items:
            top_k_items = user_recs[:k]
            satisfied = sum(
                1 for item_id in top_k_items
                if label_lookup.get(int(item_id), 0) == 1
            )
            batch_nsr.append(satisfied / max(len(top_k_items), 1))
        return float(np.mean(batch_nsr))


# ==========================================================================
# 2. NDCG@K
# ==========================================================================

def ndcg_at_k(
    recommended_items: Union[torch.Tensor, np.ndarray],
    ground_truth_items: Union[torch.Tensor, np.ndarray],
    k: int = 10,
) -> float:
    """
    计算 NDCG@K。

    参数
    ----
    recommended_items  : [B, N_items] 按分数降序排列的物品 ID
    ground_truth_items : [B] 每个用户真实交互的物品 ID
    k                  : Top-K

    返回
    ----
    ndcg : float，值域 [0, 1]
    """
    if isinstance(recommended_items, torch.Tensor):
        recommended_items = recommended_items.cpu().numpy()
    if isinstance(ground_truth_items, torch.Tensor):
        ground_truth_items = ground_truth_items.cpu().numpy()

    ndcg_list = []
    for recs, gt in zip(recommended_items, ground_truth_items):
        top_k = recs[:k]
        if gt in top_k:
            rank = np.where(top_k == gt)[0][0] + 1  # 1-indexed
            ndcg_list.append(1.0 / np.log2(rank + 1))
        else:
            ndcg_list.append(0.0)

    return float(np.mean(ndcg_list))


# ==========================================================================
# 3. Hit@K
# ==========================================================================

def hit_at_k(
    recommended_items: Union[torch.Tensor, np.ndarray],
    ground_truth_items: Union[torch.Tensor, np.ndarray],
    k: int = 10,
) -> float:
    """
    计算 Hit@K（命中率）。

    参数
    ----
    recommended_items  : [B, N_items] 按分数降序排列的物品 ID
    ground_truth_items : [B] 每个用户真实交互的物品 ID
    k                  : Top-K

    返回
    ----
    hit_rate : float，值域 [0, 1]
    """
    if isinstance(recommended_items, torch.Tensor):
        recommended_items = recommended_items.cpu().numpy()
    if isinstance(ground_truth_items, torch.Tensor):
        ground_truth_items = ground_truth_items.cpu().numpy()

    hits = [
        1 if gt in recs[:k] else 0
        for recs, gt in zip(recommended_items, ground_truth_items)
    ]
    return float(np.mean(hits))


# ==========================================================================
# 4. 一次性计算所有指标（run_ablation.py 调用这个）
# ==========================================================================

def compute_all_metrics(
    recommended_items:  Union[torch.Tensor, np.ndarray],
    ground_truth_items: Union[torch.Tensor, np.ndarray],
    closure_labels:     Union[dict, torch.Tensor, np.ndarray] = None,
    k: int = 10,
) -> dict:
    """
    一次返回所有指标。

    返回
    ----
    {
        "NDCG@K"  : float,
        "Hit@K"   : float,
        "NSR@K"   : float 或 None（若未提供 closure_labels）,
        "K"       : int,
    }
    """
    results = {
        "K":         k,
        f"NDCG@{k}": ndcg_at_k(recommended_items, ground_truth_items, k),
        f"Hit@{k}":  hit_at_k(recommended_items, ground_truth_items, k),
        f"NSR@{k}":  None,
    }

    if closure_labels is not None:
        results[f"NSR@{k}"] = need_satisfaction_rate(
            recommended_items, closure_labels, k
        )

    return results