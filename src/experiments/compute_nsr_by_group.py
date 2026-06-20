"""
compute_nsr_by_group.py — 按用户群体分组计算 NSR@10
========================================================
在 compute_nsr.py 的基础上，按用户活跃度(item_count)和购买力(buy_count)
交叉分组（高活跃高购买 / 高活跃低购买 / 低活跃高购买 / 低活跃低购买），
分别统计每组的 NSR@10，用于生成"NSR分群对比图"。

分组依据：data_process/processed/user_groups_custom.csv
  （按 item_count / buy_count 中位数二分，交叉得到四组）

用法：
    python -m src.experiments.compute_nsr_by_group

输出：
    results/nsr_by_group.csv  （每组 x 每个用户群体 的 NSR@10 矩阵）
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

def _register_fake_module():
    from src.model.mb_gcl_sasrec import MBGCLSASRec

    fake_name = "recbole.model.sequential_recommender.mbgclsasrec"
    if fake_name not in sys.modules:
        fake_module = types.ModuleType(fake_name)
        fake_module.MBGCLSASRec = MBGCLSASRec
        fake_module.__spec__ = importlib.machinery.ModuleSpec(fake_name, loader=None)
        sys.modules[fake_name] = fake_module
    return MBGCLSASRec

MBGCLSASRec = _register_fake_module()

#USER_GROUPS_PATH = ROOT / "data_process" / "processed" / "user_groups_custom.csv"
USER_GROUPS_PATH = ROOT / "data_process" / "processed" / "taobao" / "user_groups_custom.csv"
# 复用 compute_nsr.py 里的权重/yaml配置
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
GROUP_ORDER = ["高活跃高购买", "高活跃低购买", "低活跃高购买", "低活跃低购买"]





def _load_model_and_data(yaml_path: str, ckpt_path: str):
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
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    return model, dataset, test_data, device


def _load_user_group_map() -> dict:
    """
    读取 user_groups_custom.csv，返回 {user_idx: custom_group} 映射。

    注意：user_idx 是 A 同学数据处理阶段的原始用户索引，需要确认它和
    RecBole dataset 里的 user_id（经过 token 重映射后的内部 ID）是否
    一致。RecBole 通常会对 USER_ID_FIELD 做从 0/1 开始的重新编码，
    如果 user_idx 与原始 user_id:token 列的取值本身就是一致的
    （即 A 同学没有再做一次独立编码），这里直接用即可；如果编号体系
    不同，需要先用 id_mapping.pkl 转换。本脚本假设两者一致，
    若分组命中率异常低（大量用户查不到分组），需要检查这一点。
    """
    df = pd.read_csv(USER_GROUPS_PATH)
    return dict(zip(df["user_idx"].astype(int), df["custom_group"]))


def _compute_nsr_by_group(exp_id: str, yaml_path: str, ckpt_path: str,
                            user_group_map: dict, topk: int = TOPK) -> dict:
    """
    对一组模型在测试集上跑 full_sort_predict，按用户群体分别统计 NSR@K。

    返回: {group_name: nsr_value, ...}
    """
    print(f"[{exp_id}] 加载模型: {ckpt_path}")
    model, dataset, test_data, device = _load_model_and_data(yaml_path, ckpt_path)

    if "closure" not in dataset.inter_feat:
        print(f"[{exp_id}] 警告：未找到 closure 字段，跳过")
        return {g: None for g in GROUP_ORDER}

    # 分子分母按用户群体分别累计
    hit_and_closure = {g: 0 for g in GROUP_ORDER}
    total_closure = {g: 0 for g in GROUP_ORDER}
    unmatched_users = 0

    with torch.no_grad():
        for batched_data in test_data:
            interaction = batched_data[0] if isinstance(batched_data, (tuple, list)) else batched_data
            interaction = interaction.to(device)

            target_items = interaction["item_id"].cpu().numpy()
            target_closure = interaction["closure"].cpu().numpy()
            user_ids_batch = interaction["user_id"].cpu().numpy()

            scores = model.full_sort_predict(interaction)
            topk_items = torch.topk(scores, k=topk, dim=-1).indices
            topk_item_ids = (topk_items + 1).cpu().numpy()

            for i in range(len(target_items)):
                uid = int(user_ids_batch[i])
                group = user_group_map.get(uid)
                if group is None or group not in GROUP_ORDER:
                    unmatched_users += 1
                    continue

                if target_closure[i] > 0:
                    total_closure[group] += 1
                    if target_items[i] in topk_item_ids[i]:
                        hit_and_closure[group] += 1

    if unmatched_users > 0:
        print(f"[{exp_id}] 警告：{unmatched_users} 个用户在分组映射中找不到，"
              f"请检查 user_id 编号体系是否与 user_groups_custom.csv 一致")

    result = {}
    for g in GROUP_ORDER:
        if total_closure[g] == 0:
            print(f"[{exp_id}] [{g}] 无 closure=1 样本，NSR 无法计算")
            result[g] = None
        else:
            nsr = hit_and_closure[g] / total_closure[g]
            result[g] = nsr
            print(f"[{exp_id}] [{g}] NSR@{topk} = {nsr:.6f} "
                  f"(命中: {hit_and_closure[g]} / 总数: {total_closure[g]})")

    return result


def main():
    user_group_map = _load_user_group_map()
    print(f"已加载用户分组映射，共 {len(user_group_map)} 个用户\n")

    all_results = {}
    for exp_id, info in CHECKPOINTS.items():
        yaml_path = str(ROOT / info["yaml"])
        ckpt_path = str(ROOT / info["ckpt"])

        if not Path(ckpt_path).exists() or not Path(yaml_path).exists():
            print(f"[{exp_id}] 文件缺失，跳过")
            continue

        try:
            group_nsr = _compute_nsr_by_group(exp_id, yaml_path, ckpt_path, user_group_map)
            all_results[exp_id] = group_nsr
        except Exception as e:
            print(f"[{exp_id}] 计算失败: {e}")
            import traceback
            traceback.print_exc()
            all_results[exp_id] = {g: None for g in GROUP_ORDER}

    # 汇总成 DataFrame: 行=exp_id, 列=用户群体
    df = pd.DataFrame(all_results).T
    df = df[GROUP_ORDER]
    df.index.name = "exp_id"

    print("\n" + "=" * 60)
    print("NSR@10 分群结果汇总")
    print("=" * 60)
    print(df.to_string())

    out_path = ROOT / "results" / "nsr_by_group.csv"
    df.to_csv(out_path)
    print(f"\n已保存: {out_path}")


if __name__ == "__main__":
    main()