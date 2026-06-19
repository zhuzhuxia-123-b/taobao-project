import numpy as np

if not hasattr(np, "float"):
    np.float = np.float64
if not hasattr(np, "float_"):
    np.float_ = np.float64

if not hasattr(np, "complex"):
    np.complex = np.complex128
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128

if not hasattr(np, "int"):
    np.int = np.int64
if not hasattr(np, "int_"):
    np.int_ = np.int64

if not hasattr(np, "bool_"):
    np.bool_ = np.bool8
# ==========================================================================================

import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp

# ---------------------- 项目路径配置 + 清理旧缓存 + 导入模型 ----------------------
CURR_FILE = Path(__file__)
# 定位项目根目录
PROJECT_ROOT = CURR_FILE.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 清理历史recbole模块缓存
for key in list(sys.modules.keys()):
    if key.startswith("recbole"):
        del sys.modules[key]

# 导入自定义模型、配置工具
from src.model.mb_gcl_sasrec import MBGCLSASRec, _cfg_get

# 仅引入RecBole父类依赖，不触发全局扫描
import recbole
print("当前使用的recbole路径：", recbole.__file__)

# ---------------------- 业务常量配置 ----------------------
WEIGHT_FILE = "MBGCLSASRec-Jun-15-2026_16-51-24.pth"
MODEL_WEIGHT_PATH = PROJECT_ROOT / "saved" / WEIGHT_FILE
CHAINED_CSV = PROJECT_ROOT / "data" / "chained.csv"
SAVE_FIG = PROJECT_ROOT / "results" / "figures" / "real_drift.png"
GRAPH_PV = PROJECT_ROOT / "data_process" / "processed" / "graph_pv.npz"
GRAPH_CART = PROJECT_ROOT / "data_process" / "processed" / "graph_cart.npz"
GRAPH_BUY = PROJECT_ROOT / "data_process" / "processed" / "graph_buy.npz"

# 创建文件夹
Path(PROJECT_ROOT / "saved").mkdir(exist_ok=True)
SAVE_FIG.parent.mkdir(exist_ok=True)

# 绘图中文设置
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
MAX_SEQ_LEN = 50
USE_SIMULATE_DATA = False

# 【核心修改：维度和图文件对齐】
MODEL_HPARAM = {
    "n_users": 1018012,
    "n_items": 5163071,
    "embed_dim": 64,
    "max_seq_len": MAX_SEQ_LEN,
    "n_layers": 2,
    "n_heads": 2,
    "lambda1": 0.1,
    "lambda2": 0.1,
    "graph_layers": 2,
    "device": "cpu",

    # RecBole基础字段
    "USER_ID_FIELD": "user_id",
    "ITEM_ID_FIELD": "item_id",
    "TIME_FIELD": "timestamp",
    "LIST_SUFFIX": "_list",

    # 序列相关字段
    "ITEM_SEQ_FIELD": "item_id_list",
    "ITEM_LENGTH_FIELD": "item_length",
    "ITEM_LIST_LENGTH_FIELD": "item_length",
    "MAX_ITEM_LIST_LENGTH": 50,

    # 负采样配置
    "NEG_PREFIX": "neg_",
    "DEFAULT_USER_ID": 0,
    "DEFAULT_ITEM_ID": 0
}

def load_model_with_graph():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 兼容RecBole虚拟数据集类
    class DummyRecDataset:
        def __init__(self, cfg):
            self.user_num = cfg["n_users"]
            self.item_num = cfg["n_items"]
            self.USER_ID_FIELD = cfg["USER_ID_FIELD"]
            self.ITEM_ID_FIELD = cfg["ITEM_ID_FIELD"]
            self.TIME_FIELD = cfg["TIME_FIELD"]

        def num(self, field):
            if field == self.USER_ID_FIELD:
                return self.user_num
            elif field == self.ITEM_ID_FIELD:
                return self.item_num
            return 0

    # 1. 先加载训练好的权重文件
    ckpt = torch.load(MODEL_WEIGHT_PATH, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    # 2. 用大图匹配的维度初始化模型
    dummy_dataset = DummyRecDataset(MODEL_HPARAM)
    model = MBGCLSASRec(MODEL_HPARAM, dummy_dataset)

    # 超参赋值
    model.n_users = _cfg_get(MODEL_HPARAM, "n_users", 1018012)
    model.n_items = _cfg_get(MODEL_HPARAM, "n_items", 4162026)
    d = _cfg_get(MODEL_HPARAM, "embed_dim", 64)
    model.embed_dim = d
    n_layers = _cfg_get(MODEL_HPARAM, "n_layers", 2)
    n_heads = _cfg_get(MODEL_HPARAM, "n_heads", 2)
    max_seq_len = _cfg_get(MODEL_HPARAM, "max_seq_len", 50)
    graph_layers = _cfg_get(MODEL_HPARAM, "graph_layers", 2)
    model.lambda1 = _cfg_get(MODEL_HPARAM, "lambda1", 0.1)
    model.lambda2 = _cfg_get(MODEL_HPARAM, "lambda2", 0.1)

    # 3. 手动将训练权重复制到模型嵌入层前N行（核心，规避维度报错）
    model.user_emb_table.weight.data[:987995, :] = state_dict["user_emb_table.weight"]
    model.behavior_emb.item_emb.weight.data[:4162025, :] = state_dict["behavior_emb.item_emb.weight"]

    # 4. 加载图结构
    graphs = [
        sp.load_npz(str(GRAPH_PV)),
        sp.load_npz(str(GRAPH_CART)),
        sp.load_npz(str(GRAPH_BUY)),
    ]
    model.load_graphs(graphs)
    model.update_graph_emb()
    model.to(device)
    model.eval()
    print("✅ 真实模型加载成功")
    return model, device

def build_interaction(item_full, behavior_full, step_len, dev):
    prefix_item = item_full[:step_len]
    prefix_beh = behavior_full[:step_len]
    pad = MAX_SEQ_LEN - step_len
    if pad > 0:
        padded_item = [0] * pad + prefix_item
        padded_beh = [0] * pad + prefix_beh
    else:
        padded_item = prefix_item[-MAX_SEQ_LEN:]
        padded_beh = prefix_beh[-MAX_SEQ_LEN:]
    return {
        "item_id_list": torch.tensor([padded_item], dtype=torch.long).to(dev),
        "behavior_list": torch.tensor([padded_beh], dtype=torch.long).to(dev),
        "item_length": torch.tensor([min(step_len, MAX_SEQ_LEN)], dtype=torch.long).to(dev),
    }

def calc_user_drift(model, item_seq, beh_seq, dev):
    hidden_list = []
    total_len = len(item_seq)
    for t in range(1, total_len + 1):
        inter = build_interaction(item_seq, beh_seq, t, dev)
        with torch.no_grad():
            h = model.get_hidden_state(inter)
        hidden_list.append(h.squeeze(0))
    drift_res = []
    for i in range(1, len(hidden_list)):
        dist = torch.norm(hidden_list[i] - hidden_list[i-1], p=2).item()
        drift_res.append(dist)
    return drift_res

def draw_drift_figure(user_list, df):
    model, dev = load_model_with_graph()

    fig, axes = plt.subplots(5, 1, figsize=(12, 15))
    fig.suptitle("用户真实兴趣漂移曲线（模型隐向量L2距离）", fontsize=16)
    act_color = {
        "pv": ("浏览pv", "#A23B72"),
        "cart": ("加购cart", "#F18F01"),
        "buy": ("购买buy", "#C73E1D")
    }

    for idx, uid in enumerate(user_list):
        ax = axes[idx]
        user_data = df[df["user_id"] == uid].sort_values("timestamp").reset_index(drop=True)
        act_num = len(user_data)
        if act_num < 2:
           raise Exception(f"用户{uid}行为条数不足2条，无法计算真实漂移")
        items = user_data["item_id"].tolist()
    # 字符串行为转数字编码
        beh_map = {"pv": 0, "cart": 1, "buy": 2, "fav": 0}
        behaviors = [beh_map[b] for b in user_data["behavior_type"].tolist()]

        drift_data = calc_user_drift(model, items, behaviors, dev)

        x_axis = list(range(len(drift_data)))
        ax.plot(x_axis, drift_data, c="#2E86AB", lw=2, label="真实兴趣漂移L2距离")
        for x_idx, act in enumerate(behaviors[1:]):
            if x_idx >= len(drift_data):
                continue
            lab, c = act_color.get(act, ("未知行为", "#888888"))
            ax.scatter(x_axis[x_idx], drift_data[x_idx], c=c, s=70, label=lab)

        ax.set_title(f"用户{uid} 时序漂移（{act_num}条真实行为）")
        ax.set_xlabel("行为时序步（逐前缀隐向量计算）")
        ax.set_ylabel("L2距离")
        ax.grid(alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        unique_leg = dict(zip(labels, handles))
        ax.legend(unique_leg.values(), unique_leg.keys(), loc="upper right")

    plt.tight_layout()
    plt.savefig(SAVE_FIG, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 真实模型计算图表已输出至: {SAVE_FIG}")

if __name__ == "__main__":
    print("正在加载 chained.csv 行为数据集...")
    df_total = pd.read_csv(CHAINED_CSV)
    sample_users = df_total["user_id"].unique()[:5]
    print("待分析用户ID：", sample_users)
    draw_drift_figure(sample_users, df_total)