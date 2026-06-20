"""
losses.py — 损失函数模块
=========================
包含三路损失：

  L_total = L_BPR  (推荐损失，让模型会推荐)
          + λ₂ × L_BCE  (需求闭合损失，让模型会判断"用户买没买")
          + λ₁ × L_CL   (对比学习损失，由 C同学的 contrastive_loss.py 提供)

注意：L_CL 的内部实现在 src/model/contrastive_loss.py（C同学负责）。
      本文件只保留调用接口，B同学不修改 CL 内部逻辑。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================================
# 1. BPR 推荐损失
#    让模型对"用户实际交互的物品"打分高于随机负样本
# ==========================================================================

def bpr_loss(
    pos_scores: torch.Tensor,   # [B]  正样本得分
    neg_scores: torch.Tensor,   # [B]  负样本得分
    reduction:  str = "mean",
) -> torch.Tensor:
    """
    Bayesian Personalized Ranking loss。

    loss = -log(σ(pos_score - neg_score))

    参数
    ----
    pos_scores : 模型对正样本的评分 [B]
    neg_scores : 模型对负样本的评分 [B]
    reduction  : "mean" 或 "sum"

    返回
    ----
    标量 loss
    """
    loss = -F.logsigmoid(pos_scores - neg_scores)
    return loss.mean() if reduction == "mean" else loss.sum()


# ==========================================================================
# 2. BCE 需求闭合损失
#    让模型会判断"这条序列最终是否完成了购买"
# ==========================================================================

def bce_need_closure_loss(
    logits: torch.Tensor,   # [B]  模型输出的 logit（未经过 sigmoid）
    labels: torch.FloatTensor,  # [B]  A同学的 closure_label（0 或 1）
    pos_weight: float = 2.0,    # 正样本权重，因为"买了"比"没买"少
    reduction:  str = "mean",
) -> torch.Tensor:
    """
    二元交叉熵损失，用于需求闭合预测。

    参数
    ----
    logits     : 主模型输出的闭合概率 logit [B]
    labels     : A同学生成的 closure_label（0=未闭合，1=已闭合）[B]
    pos_weight : 正样本（买了）的权重，用于缓解类别不平衡
    reduction  : "mean" 或 "sum"

    返回
    ----
    标量 loss
    """
    weight = torch.tensor([pos_weight], device=logits.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight, reduction=reduction)
    return criterion(logits, labels)


# ==========================================================================
# 3. 对比学习 Loss 接口（实现在 C同学的 contrastive_loss.py）
#    B同学只调用，不修改内部逻辑
# ==========================================================================

def cl_loss_interface(
    h_strong:    torch.Tensor,   # [B, d]  强视图序列表示（含 buy/cart）
    h_weak:      torch.Tensor,   # [B, d]  弱视图序列表示（仅 pv）
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    对比学习损失调用接口
    """
    try:
        # C同学完成后会自动生效
        from src.model.contrastive_loss import ContrastiveLoss
        return ContrastiveLoss(temperature)(h_strong, h_weak)
    except ImportError:
        # C同学尚未实现，返回 0（训练不中断）
        return torch.tensor(0.0, device=h_strong.device, requires_grad=False)


# ==========================================================================
# 4. 联合损失（主模型调用这个）
# ==========================================================================

def combined_loss(
    pos_scores:   torch.Tensor,
    neg_scores:   torch.Tensor,
    closure_logits: torch.Tensor = None,
    closure_labels: torch.Tensor = None,
    h_strong:     torch.Tensor = None,
    h_weak:       torch.Tensor = None,
    lambda1:      float = 0.1,   # CL 损失权重
    lambda2:      float = 0.1,   # 需求闭合损失权重
    temperature:  float = 0.07,
) -> dict:
    """
    三路联合损失。

    返回
    ----
    dict，包含：
      "loss"      : 总损失（用于反向传播）
      "bpr"       : BPR 分量
      "bce"       : BCE 分量（若传入）
      "cl"        : CL 分量（若传入）
    """
    loss_bpr = bpr_loss(pos_scores, neg_scores)
    result   = {"bpr": loss_bpr}
    total    = loss_bpr

    if closure_logits is not None and closure_labels is not None:
        loss_bce = bce_need_closure_loss(closure_logits, closure_labels)
        result["bce"] = loss_bce
        total = total + lambda2 * loss_bce
    else:
        result["bce"] = torch.tensor(0.0)

    if h_strong is not None and h_weak is not None:
        loss_cl = cl_loss_interface(h_strong, h_weak, temperature)
        result["cl"] = loss_cl
        total = total + lambda1 * loss_cl
    else:
        result["cl"] = torch.tensor(0.0)

    result["loss"] = total
    return result