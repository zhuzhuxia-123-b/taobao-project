"""
compute_nsr.py — 补充计算 D/E/F/G 四组的 NSR@10
=====================================================
复用已训练好的模型权重，对测试集做一次 full_sort_predict，
取 Top-10 推荐物品，结合 closure 字段计算 NSR@10。

NSR@10 = Top-10 推荐列表中 closure_label=1 的物品占比

用法：
    python -m src.experiments.compute_nsr

会自动读取每组对应的 yaml 和权重路径（在 CHECKPOINTS 字典里配置），
跑完后把结果合并进 results/ablation_results_merged.csv 的 NSR@10 列。
"""

import sys
import types
import importlib.machinery
from pathlib import Path

import numpy as np
import torch
import pandas as pd

# ── NumPy 2.x 兼容补丁 ──────────────────────────────────────────────
import numpy as _np
if not hasattr(_np, "float_"):
    _np.float_ = _np.float64
if not hasattr(_np, "unicode_"):
    _np.unicode_ = _np.str_

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ── 每组对应的 权重文件 / yaml 配置 ──────────────────────────────────
# 路径按用户提供的时间顺序对应：E -> F -> G -> D
CHECKPOINTS = {
    "D": {
        "ckpt": "saved/MBGCLSASRec-Jun-15-2026_16-51-24.pth",
        "yaml": "src/experiments/configs/ablation_D_full_model.yaml",
    },
    "E": {
        "ckpt": "saved/MBGCLSASRec-Jun-14-2026_19-47-01.pth",
        "yaml": "src/experiments/configs/ablation_E_no_cl.yaml",
    },
    "F": {
        "ckpt": "saved/MBGCLSASRec-Jun-15-2026_02-17-16.pth",
        "yaml": "src/experiments/configs/ablation_F_no_graph.yaml",
    },
    "G": {
        "ckpt": "saved/MBGCLSASRec-Jun-15-2026_08-47-56.pth",
        "yaml": "src/experiments/configs/ablation_G_no_bce.yaml",
    },
}

TOPK = 10

def _register_fake_module():
    """注册虚拟模块，让 RecBole 的 get_model 能找到 MBGCLSASRec。"""
    from src.model.mb_gcl_sasrec import MBGCLSASRec

    fake_name = "recbole.model.sequential_recommender.mbgclsasrec"
    if fake_name not in sys.modules:
        fake_module = types.ModuleType(fake_name)
        fake_module.MBGCLSASRec = MBGCLSASRec
        fake_module.__spec__ = importlib.machinery.ModuleSpec(fake_name, loader=None)
        sys.modules[fake_name] = fake_module
    return MBGCLSASRec

def _load_model_and_data(yaml_path: str, ckpt_path: str):
    """构建 config/dataset，实例化模型并加载权重。"""
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation

    MBGCLSASRec = _register_fake_module()

    config = Config(model="MBGCLSASRec", config_file_list=[yaml_path])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = MBGCLSASRec(config, dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    # RecBole 保存格式通常是 {"state_dict": ..., "config": ..., "epoch": ...}
    # 但也可能直接就是 state_dict 本身，两种都兼容
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    return model, dataset, test_data, device

def _compute_nsr_for_group(exp_id: str, yaml_path: str, ckpt_path: str, topk: int = TOPK) -> float:
    """
    对一组模型在测试集上跑 full_sort_predict，取 Top-K，结合 closure 计算 NSR@K。

    修正后的定义：
        对每个测试用户，其测试目标物品（即该用户真实发生的下一次交互）
        带有一个 closure 标签。如果模型的 Top-K 推荐列表命中了"会促成购买闭合"
        的物品（即测试目标本身 closure=1，且模型成功推荐到了它），
        则视为该用户的需求被满足。

        NSR@K = (测试集中，Top-K命中且目标closure=1 的用户数)
                / (测试集中，目标closure=1 的用户总数)

    这避免了"按item_id全局聚合closure"导致的指标失真（热门物品历史上
    只要被任何人买过一次，就被错误地永久标记为closure=1）。
    """
    print(f"[{exp_id}] 加载模型: {ckpt_path}")
    model, dataset, test_data, device = _load_model_and_data(yaml_path, ckpt_path)

    if "closure" not in dataset.inter_feat:
        print(f"[{exp_id}] 警告：dataset 中未找到 closure 字段，NSR 无法计算，返回 None")
        return None

    hit_and_closure = 0   # Top-K命中目标物品 且 目标closure=1
    total_closure = 0     # 目标closure=1 的用户总数

    with torch.no_grad():
        for batch_idx, batched_data in enumerate(test_data):
            interaction = batched_data[0] if isinstance(batched_data, (tuple, list)) else batched_data
            interaction = interaction.to(device)

            # 测试目标物品：RecBole leave-one-out 范式下，
            # interaction["item_id"] 就是该用户本次测试的真实目标物品
            target_items = interaction["item_id"].cpu().numpy()          # [B]
            target_closure = interaction["closure"].cpu().numpy()        # [B]

            scores = model.full_sort_predict(interaction)                # [B, n_items]
            topk_items = torch.topk(scores, k=topk, dim=-1).indices      # [B, K]
            # full_sort_predict 输出去掉了 padding 行(index 0)，
            # 下标 j 对应真实 item_id = j + 1
            topk_item_ids = (topk_items + 1).cpu().numpy()               # [B, K]

            for i in range(len(target_items)):
                if target_closure[i] > 0:
                    total_closure += 1
                    if target_items[i] in topk_item_ids[i]:
                        hit_and_closure += 1

    if total_closure == 0:
        print(f"[{exp_id}] 警告：测试集中没有 closure=1 的目标用户，NSR 无法计算")
        return None

    final_nsr = hit_and_closure / total_closure
    print(f"[{exp_id}] NSR@{topk} = {final_nsr} "
          f"(命中且closure=1: {hit_and_closure} / closure=1总数: {total_closure})")
    return final_nsr

def main():
    results = {}
    for exp_id, info in CHECKPOINTS.items():
        yaml_path = str(ROOT / info["yaml"]) if not Path(info["yaml"]).is_absolute() else info["yaml"]
        ckpt_path = str(ROOT / info["ckpt"]) if not Path(info["ckpt"]).is_absolute() else info["ckpt"]

        if not Path(ckpt_path).exists():
            print(f"[{exp_id}] 权重文件不存在: {ckpt_path}，跳过")
            continue
        if not Path(yaml_path).exists():
            print(f"[{exp_id}] yaml 不存在: {yaml_path}，跳过")
            continue

        try:
            nsr = _compute_nsr_for_group(exp_id, yaml_path, ckpt_path)
            results[exp_id] = nsr
        except Exception as e:
            print(f"[{exp_id}] 计算失败: {e}")
            import traceback
            traceback.print_exc()
            results[exp_id] = None

    print("\n" + "=" * 50)
    print("NSR@10 计算结果汇总")
    print("=" * 50)
    for exp_id, nsr in results.items():
        print(f"{exp_id}: {nsr}")

    # 写回 merged CSV
    merged_path = ROOT / "results" / "ablation_results_merged.csv"
    if merged_path.exists():
        df = pd.read_csv(merged_path)
        for exp_id, nsr in results.items():
            if nsr is not None:
                df.loc[df["exp_id"] == exp_id, "NSR@10"] = nsr
        df.to_csv(merged_path, index=False)
        print(f"\n已更新: {merged_path}")
        print(df.to_string(index=False))
    else:
        print(f"\n警告：{merged_path} 不存在，结果未写回，仅打印在上面")

if __name__ == "__main__":
    main()
