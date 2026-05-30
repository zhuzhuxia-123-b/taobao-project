"""
NSR（需求满足率）评估指标
C同学实现
输入：模型推荐的Top-K列表 + 用户的closure_label标签
输出：推荐Top-K中最终促成需求闭合（buy）的商品比例
"""

import numpy as np
import pandas as pd


def compute_nsr(
    recommended_items: list,
    closure_labels: dict,
    k: int = 10,
) -> float:
    """
    计算NSR@K

    Args:
        recommended_items: 每个用户的Top-K推荐列表 List[List[item_id]]
        closure_labels:    每个用户最终发生buy的商品集合 Dict[user_idx -> Set[item_id]]
        k:                 评估截断位置

    Returns:
        float: NSR@K均值（0-1之间）
    """
    nsr_list = []
    for user_idx, rec_list in enumerate(recommended_items):
        top_k = rec_list[:k]
        closed = closure_labels.get(user_idx, set())
        if len(top_k) == 0:
            continue
        hits = sum(1 for item in top_k if item in closed)
        nsr_list.append(hits / len(top_k))

    return float(np.mean(nsr_list)) if nsr_list else 0.0


def build_closure_labels(chain_csv_path: str) -> dict:
    """
    从chain_clean.csv构建闭合标签字典

    Args:
        chain_csv_path: chain_clean.csv的路径

    Returns:
        Dict[user_id -> Set[item_id]]
    """
    df = pd.read_csv(chain_csv_path)
    closed = df[df['closure_label'] == 1]

    labels = {}
    for row in closed.itertuples():
        uid = int(row.user_id)
        iid = int(row.item_id)
        labels.setdefault(uid, set()).add(iid)

    return labels


def compute_nsr_by_group(
    recommended_items: list,
    closure_labels: dict,
    user_groups: dict,
    k: int = 10,
) -> dict:
    """
    按用户群体分组计算NSR@K

    Args:
        user_groups: Dict[user_idx -> group_id(0-3)]

    Returns:
        Dict[group_id -> NSR@K]
    """
    group_recs = {g: [] for g in range(4)}
    group_labels = {g: {} for g in range(4)}

    for user_idx, rec_list in enumerate(recommended_items):
        g = user_groups.get(user_idx, 0)
        local_idx = len(group_recs[g])
        group_recs[g].append(rec_list)
        if user_idx in closure_labels:
            group_labels[g][local_idx] = closure_labels[user_idx]

    return {g: compute_nsr(group_recs[g], group_labels[g], k) for g in range(4)}