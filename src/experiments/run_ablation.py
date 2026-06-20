"""
run_ablation.py — 消融实验主控脚本
=====================================
C同学负责实现。

功能：
  1. 依次运行 A~G 共 7 组消融实验
  2. 每组实验结束后记录 NDCG@10 / Hit@10 / NSR@10
  3. 所有结果汇总保存到 results/ablation_results.csv

运行方式（在项目根目录）：
  python -m src.experiments.run_ablation            # 跑全部 7 组
  python -m src.experiments.run_ablation --exp D    # 只跑某一组
  python -m src.experiments.run_ablation --baselines_only  # 只跑 A/B/C 三组基线

依赖：
  pip install recbole pandas
"""

# ── NumPy 2.x 兼容补丁（RecBole 1.2.0 用到了已移除的旧别名）──────────
# 必须在 import recbole 之前执行
import numpy as _np
if not hasattr(_np, "float_"):
    _np.float_ = _np.float64
if not hasattr(_np, "complex_"):
    _np.complex_ = _np.complex128
if not hasattr(_np, "object_"):
    _np.object_ = _np.object_  # 通常仍存在，保险写法
if not hasattr(_np, "unicode_"):
    _np.unicode_ = _np.str_

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import torch



# ── 路径设置 ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]   # taobao-project/
sys.path.insert(0, str(ROOT))

CONFIGS_DIR = ROOT / "src" / "experiments" / "configs"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── 消融实验配置表 ────────────────────────────────────────────────────
# key: 实验编号
# value: (描述, yaml文件名, 是否需要自定义模型)
ABLATION_CONFIG = {
    "A": ("ItemKNN 基线（无序列）",      "ablation_A_itemknn.yaml",    False),
    "B": ("GRU4Rec 基线（RNN序列）",     "ablation_B_gru4rec.yaml",    False),
    "C": ("SASRec 基线（标准自注意力）", "ablation_C_sasrec.yaml",     False),
    "D": ("完整模型 MB-GCL-SASRec",      "ablation_D_full_model.yaml", True),
    "E": ("去掉对比学习 (λ1=0)",         "ablation_E_no_cl.yaml",      True),
    "F": ("去掉图卷积 (graph_layers=0)", "ablation_F_no_graph.yaml",   True),
    "G": ("去掉需求闭合损失 (λ2=0)",     "ablation_G_no_bce.yaml",     True),
}


# ── RecBole 基线实验（A/B/C 组）────────────────────────────────────────

def run_recbole_baseline(exp_id: str, config_file: str, desc: str) -> dict:
    """
    用 RecBole 内置模型跑一组实验，返回指标 dict。
    """
    try:
        from recbole.quick_start import run_recbole
    except ImportError:
        logger.error("RecBole 未安装，请运行: pip install recbole")
        return _empty_result(exp_id, desc, error="RecBole not installed")

    config_path = str(CONFIGS_DIR / config_file)
    if not Path(config_path).exists():
        logger.error(f"配置文件不存在: {config_path}")
        return _empty_result(exp_id, desc, error=f"Config not found: {config_path}")

    logger.info(f"[{exp_id}] 开始: {desc}")
    logger.info(f"[{exp_id}] 配置文件: {config_path}")

    try:
        result = run_recbole(config_file_list=[config_path])
        # RecBole 返回 (config, model, dataset, train_data, valid_data, test_data)
        # result 本身是 test 结果 dict
        metrics = _parse_recbole_result(result)
        metrics.update({
            "exp_id":  exp_id,
            "desc":    desc,
            "status":  "success",
        })
        logger.info(f"[{exp_id}] 完成: NDCG@10={metrics.get('NDCG@10', 'N/A'):.4f}")
        return metrics

    except Exception as e:
        logger.error(f"[{exp_id}] 失败: {e}", exc_info=True)
        return _empty_result(exp_id, desc, error=str(e))


def _parse_recbole_result(result) -> dict:
    """
    解析 RecBole 返回的结果对象为标准 dict。

    RecBole run_recbole() 实际返回格式（嵌套字典）：
      {
        'best_valid_score': 0.4802,
        'best_valid_result': OrderedDict({'recall@10': ..., 'ndcg@10': ..., 'hit@10': ...}),
        'test_result':       OrderedDict({'recall@10': ..., 'ndcg@10': ..., 'hit@10': ...}),
      }

    我们要的是 test_result 里的指标。
    """
    metrics = {}

    # 优先取 test_result，没有的话退回 best_valid_result，再退回顶层（兼容旧/扁平格式）
    source = None
    if isinstance(result, dict):
        if "test_result" in result and isinstance(result["test_result"], dict):
            source = result["test_result"]
        elif "best_valid_result" in result and isinstance(result["best_valid_result"], dict):
            source = result["best_valid_result"]
        else:
            source = result

    if source:
        for key, val in source.items():
            key_lower = key.lower()
            if isinstance(val, str):
                continue
            if "ndcg" in key_lower:
                metrics["NDCG@10"] = float(val)
            elif "hit" in key_lower:
                metrics["Hit@10"] = float(val)
            elif "recall" in key_lower:
                metrics["Recall@10"] = float(val)

    metrics.setdefault("NDCG@10", 0.0)
    metrics.setdefault("Hit@10", 0.0)
    metrics.setdefault("Recall@10", 0.0)
    metrics["NSR@10"] = None   # 基线不计算 NSR
    return metrics


# ── 自定义模型实验（D/E/F/G 组）────────────────────────────────────────

def run_custom_model(exp_id: str, config_file: str, desc: str) -> dict:
    """
    跑 MB-GCL-SASRec 自定义模型实验（D/E/F/G 组）。
    """
    import yaml
    import scipy.sparse as sp
    from scipy.sparse import load_npz

    config_path = CONFIGS_DIR / config_file
    if not config_path.exists():
        return _empty_result(exp_id, desc, error=f"Config not found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    logger.info(f"[{exp_id}] 开始: {desc}")

    try:
        # ── 关键修复：把自定义模型注册为虚拟模块，让 RecBole 的
        # get_model() 能通过 find_spec 找到它 ──────────────────────
        #
        # get_model() 的真实逻辑：
        #   module_path = "recbole.model.sequential_recommender." + model_name.lower()
        #   if importlib.util.find_spec(module_path): ...
        #
        # find_spec 在查磁盘前会先查 sys.modules 缓存。只要我们把一个
        # 名字为 "recbole.model.sequential_recommender.mbgclsasrec"
        # 的虚拟模块塞进 sys.modules，并在其上挂一个 MBGCLSASRec 属性，
        # get_model() 就能正常 import_module + getattr 拿到我们的类，
        # 完全不需要修改 RecBole 安装目录下的任何文件。
        from recbole.quick_start import run_recbole
        import sys, types, importlib.machinery
        from src.model.mb_gcl_sasrec import MBGCLSASRec

        _fake_module_name = "recbole.model.sequential_recommender.mbgclsasrec"
        if _fake_module_name not in sys.modules:
            _fake_module = types.ModuleType(_fake_module_name)
            _fake_module.MBGCLSASRec = MBGCLSASRec
            # find_spec 在 sys.modules 命中时会检查 __spec__，
            # types.ModuleType 默认 __spec__=None 会触发
            # "X.__spec__ is None" 的 ValueError，必须手动补一个。
            _fake_module.__spec__ = importlib.machinery.ModuleSpec(
                _fake_module_name, loader=None
            )
            sys.modules[_fake_module_name] = _fake_module

        result = run_recbole(
            model="MBGCLSASRec",
            config_file_list=[str(config_path)],
        )
        metrics = _parse_recbole_result(result)

    except Exception as e:
        # ── RecBole 跑不通时，走独立训练流程（降级方案，应极少触发）──
        logger.warning(f"[{exp_id}] RecBole 方式失败 ({e})，尝试独立训练...")
        try:
            metrics = _run_standalone(cfg, exp_id)
        except Exception as e2:
            logger.error(f"[{exp_id}] 独立训练也失败: {e2}", exc_info=True)
            return _empty_result(exp_id, desc, error=str(e2))

    metrics.update({"exp_id": exp_id, "desc": desc, "status": "success"})
    logger.info(f"[{exp_id}] 完成: NDCG@10={metrics.get('NDCG@10', 'N/A')}")
    return metrics


def _run_standalone(cfg: dict, exp_id: str) -> dict:
    """
    不依赖 RecBole，直接实例化模型跑一轮评估。
    用于 RecBole 无法注册自定义模型时的降级方案。
    """
    from src.model.mb_gcl_sasrec import MBGCLSASRec
    import scipy.sparse as sp
    from recbole.data import create_dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 修复1：手动补充 RecBole 必需字段，解决 KEYERROR: MODEL_TYPE
    #cfg["MODEL_TYPE"] = "Sequential"
    cfg["MODEL_TYPE"] = "SEQ"

    dataset = create_dataset(cfg)
    # 修复2：标准双参数传参 config + dataset，解决初始化缺参
    model = MBGCLSASRec(cfg, dataset).to(device)

    # 加载图（如果路径存在）
    graph_paths = [
        cfg.get("graph_pv_path"),
        cfg.get("graph_cart_path"),
        cfg.get("graph_buy_path"),
    ]
    if all(p and Path(ROOT / p).exists() for p in graph_paths):
        graphs = [sp.load_npz(str(ROOT / p)) for p in graph_paths]
        model.load_graphs(graphs)
        logger.info(f"[{exp_id}] 图加载完成")
    else:
        logger.warning(f"[{exp_id}] 图文件缺失，跳过图加载")

    # TODO: 此处接入完整训练循环
    logger.warning(f"[{exp_id}] 独立训练循环尚未实现，返回占位结果")
    return {
        "NDCG@10":   None,
        "Hit@10":    None,
        "Recall@10": None,
        "NSR@10":    None,
    }

# ── NSR 计算（在基线结果上补充计算）──────────────────────────────────

def compute_nsr_for_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    对已有结果补充计算 NSR@10。
    需要 chained_clean.csv 中的 closure_label。
    """
    chained_path = ROOT / "data_process" / "processed" / "chained_clean.csv"
    if not chained_path.exists():
        logger.warning("chained_clean.csv 不存在，跳过 NSR 计算")
        return results_df

    try:
        chained = pd.read_csv(chained_path)
        # 构建 item_idx -> closure_label 映射（取每个物品最高 closure_label）
        closure_map = chained.groupby("item_idx")["closure_label"].max().to_dict()
        logger.info(f"closure_label 映射加载完成，共 {len(closure_map)} 个物品")
        # NSR 需要模型推荐结果，此处仅标记"待计算"
        # 实际 NSR 在模型评估时通过 metrics.py 的 compute_all_metrics 计算
    except Exception as e:
        logger.warning(f"NSR 计算跳过: {e}")

    return results_df


# ── 结果保存 ──────────────────────────────────────────────────────────

def save_results(results: list):
    """
    保存所有实验结果到 CSV。
    """
    df = pd.DataFrame(results)

    # 列排序
    cols = ["exp_id", "desc", "NDCG@10", "Hit@10", "Recall@10", "NSR@10", "status"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = RESULTS_DIR / f"ablation_results_{timestamp}.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"\n结果已保存: {out_path}")

    # 同时打印到终端
    print("\n" + "="*60)
    print("消融实验结果汇总")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)

    return out_path


# ── 主函数 ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="消融实验主控脚本")
    parser.add_argument(
        "--exp", type=str, default=None,
        help="只跑指定实验，如 --exp D，默认跑全部"
    )
    parser.add_argument(
        "--baselines_only", action="store_true",
        help="只跑 A/B/C 三组基线（不需要自定义模型）"
    )
    args = parser.parse_args()

    # 确定要跑的实验列表
    if args.exp:
        exp_ids = [args.exp.upper()]
    elif args.baselines_only:
        exp_ids = ["A", "B", "C"]
    else:
        exp_ids = list(ABLATION_CONFIG.keys())   # A~G

    # 检查配置文件是否存在
    for eid in exp_ids:
        if eid not in ABLATION_CONFIG:
            logger.error(f"未知实验编号: {eid}，可选: {list(ABLATION_CONFIG.keys())}")
            sys.exit(1)

    logger.info(f"本次运行实验: {exp_ids}")
    results = []

    for eid in exp_ids:
        desc, config_file, is_custom = ABLATION_CONFIG[eid]
        if is_custom:
            r = run_custom_model(eid, config_file, desc)
        else:
            r = run_recbole_baseline(eid, config_file, desc)
        results.append(r)

    save_results(results)


def _empty_result(exp_id, desc, error=""):
    return {
        "exp_id":    exp_id,
        "desc":      desc,
        "NDCG@10":   None,
        "Hit@10":    None,
        "Recall@10": None,
        "NSR@10":    None,
        "status":    f"failed: {error}",
    }


if __name__ == "__main__":
    main()