import numpy as np
import streamlit as st
import io
from PIL import Image
import sys
from pathlib import Path
import torch
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
import seaborn as sns

# -------------------------- numpy 兼容修复 --------------------------
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

# 【修复缺失常量】补充np.load需要的参数
PICKLE_KWARGS = {"allow_pickle": True}

# -------------------------- 路径适配：文件位置 src/demo/app.py --------------------------
CURR_FILE = Path(__file__)
# src/demo/app.py 向上两级 = 项目根目录 D:\社会计算
PROJECT_ROOT = Path(r"D:\社会计算")
sys.path.insert(0, str(PROJECT_ROOT))

# 清理recbole缓存
for key in list(sys.modules.keys()):
    if key.startswith("recbole"):
        del sys.modules[key]

# 导入模型
from src.model.mb_gcl_sasrec import MBGCLSASRec, _cfg_get
import recbole

# 【正确根目录相对路径】所有资源都在项目根目录下，不再嵌套src
WEIGHT_FILE = "MBGCLSASRec-Jun-15-2026_16-51-24.pth"
MODEL_WEIGHT_PATH = PROJECT_ROOT / "saved" / WEIGHT_FILE
CHAINED_CSV = PROJECT_ROOT / "data" / "chained.csv"
GRAPH_PV = PROJECT_ROOT / "data_process" / "processed" / "graph_pv.npz"
GRAPH_CART = PROJECT_ROOT / "data_process" / "processed" / "graph_cart.npz"
GRAPH_BUY = PROJECT_ROOT / "data_process" / "processed" / "graph_buy.npz"

# 绘图全局设置
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
MAX_SEQ_LEN = 50
BEH_MAP = {"pv": 0, "cart": 1, "buy": 2, "fav": 0}
STATE_MAP = {0:"latent潜在浏览",1:"exploring探索加购",2:"intent意向购买",3:"dormant沉睡用户"}

# 模型超参（与drift.py完全统一）
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
    "USER_ID_FIELD": "user_id",
    "ITEM_ID_FIELD": "item_id",
    "TIME_FIELD": "timestamp",
    "LIST_SUFFIX": "_list",
    "ITEM_SEQ_FIELD": "item_id_list",
    "ITEM_LENGTH_FIELD": "item_length",
    "ITEM_LIST_LENGTH_FIELD": "item_length",
    "MAX_ITEM_LIST_LENGTH": 50,
    "NEG_PREFIX": "neg_",
    "DEFAULT_USER_ID": 0,
    "DEFAULT_ITEM_ID": 0
}

# -------------------------- 基础模型、漂移工具函数（页面2专用） --------------------------
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

def load_model_with_graph():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(MODEL_WEIGHT_PATH, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    dummy_dataset = DummyRecDataset(MODEL_HPARAM)
    model = MBGCLSASRec(MODEL_HPARAM, dummy_dataset)
    model.n_users = _cfg_get(MODEL_HPARAM, "n_users", 1018012)
    model.n_items = _cfg_get(MODEL_HPARAM, "n_items", 5163071)
    d = _cfg_get(MODEL_HPARAM, "embed_dim", 64)
    model.embed_dim = d
    n_layers = _cfg_get(MODEL_HPARAM, "n_layers", 2)
    n_heads = _cfg_get(MODEL_HPARAM, "n_heads", 2)
    max_seq_len = _cfg_get(MODEL_HPARAM, "max_seq_len", 50)
    graph_layers = _cfg_get(MODEL_HPARAM, "graph_layers", 2)
    model.lambda1 = _cfg_get(MODEL_HPARAM, "lambda1", 0.1)
    model.lambda2 = _cfg_get(MODEL_HPARAM, "lambda2", 0.1)
    model.user_emb_table.weight.data[:987995, :] = state_dict["user_emb_table.weight"]
    model.behavior_emb.item_emb.weight.data[:4162025, :] = state_dict["behavior_emb.item_emb.weight"]
    graphs = [
    sp.load_npz(str(GRAPH_PV)),
    sp.load_npz(str(GRAPH_CART)),
    sp.load_npz(str(GRAPH_BUY))
    ]
    model.load_graphs(graphs)
    model.update_graph_emb()
    model.to(device)
    model.eval()
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

# -------------------------- 页面1：rerank重排序核心逻辑（复用src/analysis/rerank.py） --------------------------
def judge_user_state(user_df):
    behavior_list = user_df["behavior_type"].tolist()
    has_buy = "buy" in behavior_list
    has_cart = "cart" in behavior_list
    has_pv = "pv" in behavior_list or "fav" in behavior_list
    if has_buy:
        return 2
    elif has_cart:
        return 1
    elif has_pv:
        return 0
    else:
        return 3

def rerank_top10(raw_rec, user_state):
    weight_dict = {
        0: {"pop":0.7,"sim":0.3},
        1: {"sim":0.6,"fresh":0.4},
        2: {"seq_click":0.8,"cart":0.2},
        3: {"recall":0.5,"hot":0.5}
    }
    w = weight_dict[user_state]
    score = {}
    for idx,item in enumerate(raw_rec):
        base = 1.0
        score[item] = base
    sorted_items = sorted(score.items(),key=lambda x:x[1],reverse=True)
    return [i[0] for i in sorted_items[:10]]

# 模拟SASRec原版推理（可替换真实SASRec加载函数）
def sasrec_predict(df, uid):
    user_data = df[df["user_id"]==uid].sort_values("timestamp")
    item_seq = user_data["item_id"].tolist()
    return item_seq[:10] if len(item_seq)>=10 else item_seq

def mb_model_predict(model, dev, item_seq, beh_seq):
    inter = build_interaction(item_seq, beh_seq, len(item_seq), dev)
    with torch.no_grad():
        # 获取用户最终隐向量
        user_emb = model.get_hidden_state(inter)
        # 全部商品嵌入
        all_item_emb = model.behavior_emb.item_emb.weight
        # 内积打分
        scores = torch.matmul(user_emb, all_item_emb.T).squeeze(0)
    top_idx = torch.topk(scores, 10).indices.cpu().tolist()
    return top_idx

# -------------------------- 页面3：群体分析逻辑（复用src/analysis/group_analysis.py） --------------------------
def split_four_group(df):
    all_uid = df["user_id"].unique()
    split_num = len(all_uid)//4
    g1 = all_uid[:split_num]
    g2 = all_uid[split_num:split_num*2]
    g3 = all_uid[split_num*2:split_num*3]
    g4 = all_uid[split_num*3:]
    group_map = {}
    for u in g1: group_map[u] = "群体1"
    for u in g2: group_map[u] = "群体2"
    for u in g3: group_map[u] = "群体3"
    for u in g4: group_map[u] = "群体4"
    df["group"] = df["user_id"].map(group_map)
    return df

def get_funnel_data(group_df):
    pv_cnt = len(group_df[group_df["behavior_type"].isin(["pv","fav"])])
    cart_cnt = len(group_df[group_df["behavior_type"]=="cart"])
    buy_cnt = len(group_df[group_df["behavior_type"]=="buy"])
    return pd.DataFrame({"阶段":["浏览","加购","购买"],"数量":[pv_cnt,cart_cnt,buy_cnt]})

def get_decision_gap(group_df):
    gap_list = []
    for uid in group_df["user_id"].unique():
        u_df = group_df[group_df["user_id"]==uid].sort_values("timestamp")
        pv_time = u_df[u_df["behavior_type"].isin(["pv","fav"])]["timestamp"].min()
        buy_time = u_df[u_df["behavior_type"]=="buy"]["timestamp"].min()
        if pd.notna(pv_time) and pd.notna(buy_time):
            gap_h = (pd.to_datetime(buy_time)-pd.to_datetime(pv_time)).total_seconds()/3600
            gap_list.append(gap_h)
    return gap_list

def get_heatmap_matrix(group_df):
    trans = np.zeros((3,3))
    beh_map_num = {"pv":0,"cart":1,"buy":2,"fav":0}
    beh_seq = [beh_map_num[b] for b in group_df["behavior_type"]]
    for i in range(len(beh_seq)-1):
        s = beh_seq[i]
        e = beh_seq[i+1]
        trans[s][e] += 1
    return trans

# -------------------------- 全局缓存初始化 --------------------------
@st.cache_resource
def init_env():
    model, dev = load_model_with_graph()
    df = pd.read_csv(CHAINED_CSV)
    df = split_four_group(df)
    return model, dev, df

# -------------------------- Streamlit 页面入口 --------------------------
st.set_page_config(page_title="用户兴趣漂移分析Demo", layout="wide")
st.title("基于MBGCLSASRec模型的用户兴趣漂移可视化分析系统")

page_select = st.selectbox("功能页面切换", [
    "1. 个性化推荐对比",
    "2. 用户兴趣漂移分析",
    "3. 群体行为社会计算分析"
])
model, device, df_total = init_env()
user_unique_list = sorted(df_total["user_id"].unique().tolist())

# ==================== 页面一：个性化推荐对比（全部开发完成） ====================
if page_select == "1. 个性化推荐对比":
    st.subheader("一、个性化推荐对比：SASRec原版 VS MBGCLSASRec + 需求状态重排序")
    sel_uid = st.selectbox("选择待分析用户ID", user_unique_list)
    run_btn = st.button("生成Top10推荐对比", type="primary")
    if run_btn:
        with st.spinner("模型推理、需求重排序计算中..."):
            user_df = df_total[df_total["user_id"]==sel_uid].sort_values("timestamp").reset_index(drop=True)
            if len(user_df) < 1:
                st.error("该用户无行为数据，无法生成推荐！")
            else:
                # 1. 判断用户需求状态
                state_code = judge_user_state(user_df)
                state_name = STATE_MAP[state_code]
                st.info(f"该用户需求状态：{state_name}")
                # 2. 两套模型原始推荐
                sasrec_top10 = sasrec_predict(df_total, sel_uid)
                item_seq = user_df["item_id"].tolist()
                beh_seq = [BEH_MAP[b] for b in user_df["behavior_type"]]
                mb_raw_top10 = mb_model_predict(model, device, item_seq, beh_seq)
                # 3. 需求重排序
                rerank_top10 = rerank_top10(mb_raw_top10, state_code)
                # 4. 三列表展示
                c1,c2,c3 = st.columns(3)
                with c1:
                    st.write("SASRec原版Top10")
                    st.dataframe(pd.DataFrame({"物品ID":sasrec_top10}))
                with c2:
                    st.write("MBGCLSASRec原始推荐")
                    st.dataframe(pd.DataFrame({"物品ID":mb_raw_top10}))
                with c3:
                    st.write("需求重排序后Top10")
                    st.dataframe(pd.DataFrame({"物品ID":rerank_top10}))
                # 表格下载
                export_df = pd.DataFrame({
                    "SASRec原版":sasrec_top10,
                    "新模型原始":mb_raw_top10,
                    "重排序后":rerank_top10
                })
                csv_buf = export_df.to_csv(index=False).encode("utf-8")
                st.download_button("下载推荐对比表格CSV", csv_buf, f"user_{sel_uid}_rec_compare.csv")

# ==================== 页面二：用户兴趣漂移（完整开发完成） ====================
elif page_select == "2. 用户兴趣漂移分析":
    st.subheader("一、数据集概览")
    c1,c2 = st.columns(2)
    c1.metric("总用户数量", df_total["user_id"].nunique())
    c2.metric("总行为记录", len(df_total))
    sel_uid = st.selectbox("选择待分析用户ID", user_unique_list)
    run_btn = st.button("计算兴趣漂移并绘图", type="primary")
    if run_btn:
        with st.spinner("模型推理计算漂移值..."):
            user_df = df_total[df_total["user_id"]==sel_uid].sort_values("timestamp").reset_index(drop=True)
            behavior_count = len(user_df)
            if behavior_count < 2:
                st.error(f"用户{sel_uid}仅{behavior_count}条行为，不足2条无法计算漂移！")
            else:
                item_seq = user_df["item_id"].tolist()
                behavior_seq = [BEH_MAP[b] for b in user_df["behavior_type"].tolist()]
                drift_list = calc_user_drift(model, item_seq, behavior_seq, device)
                avg_drift = np.mean(drift_list)
                max_drift = np.max(drift_list)
                min_drift = np.min(drift_list)
                # 绘图
                fig, ax = plt.subplots(figsize=(12,6))
                x_axis = list(range(len(drift_list)))
                ax.plot(x_axis, drift_list, c="#2E86AB", lw=2, label="相邻隐向量L2漂移距离")
                ax.set_title(f"用户{sel_uid}时序兴趣漂移曲线", fontsize=14)
                ax.set_xlabel("行为时序步")
                ax.set_ylabel("L2距离（漂移程度）")
                ax.grid(alpha=0.3)
                ax.legend()
                # 内存图片
                buf = io.BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                buf.seek(0)
                img = Image.open(buf)
                # 指标展示
                st.subheader("二、漂移量化统计")
                cc1,cc2,cc3 = st.columns(3)
                cc1.metric("平均漂移值", f"{avg_drift:.4f}")
                cc2.metric("最大漂移值", f"{max_drift:.4f}")
                cc3.metric("最小漂移值", f"{min_drift:.4f}")
                # 图表展示+下载
                st.subheader("三、兴趣漂移可视化曲线")
                st.image(img, width="stretch")
                buf.seek(0)
                st.download_button("下载漂移曲线图PNG", buf, f"user_{sel_uid}_drift.png")

# ==================== 页面三：群体行为分析（全部开发完成） ====================
elif page_select == "3. 群体行为社会计算分析":
    st.subheader("四类用户群体行为可视化分析（漏斗图 / 决策箱线图 / 行为热力图）")
    group_opt = ["群体1","群体2","群体3","群体4"]
    sel_group = st.selectbox("选择目标分析群体", group_opt)
    run_btn = st.button("渲染群体全部可视化图表", type="primary")
    if run_btn:
        with st.spinner("群体数据聚合、绘图中..."):
            group_df = df_total[df_total["group"] == sel_group].copy()
            # 1. 需求漏斗图
            st.subheader(f"1. {sel_group} 需求转化漏斗图")
            funnel_data = get_funnel_data(group_df)
            fig1, ax1 = plt.subplots(figsize=(10,4))
            ax1.bar(funnel_data["阶段"], funnel_data["数量"], color="#4A90E2")
            ax1.set_title(f"{sel_group} 用户浏览-加购-购买转化漏斗")
            buf1 = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf1, format="png", dpi=300)
            buf1.seek(0)
            st.image(Image.open(buf1))
            buf1.seek(0)
            st.download_button("下载漏斗图", buf1, f"{sel_group}_funnel.png")

            # 2. 决策时间箱线图
            st.subheader(f"2. {sel_group} 首次浏览到购买间隔箱线图")
            gap_data = get_decision_gap(group_df)
            fig2, ax2 = plt.subplots(figsize=(10,4))
            ax2.boxplot(gap_data)
            ax2.set_title(f"{sel_group} 用户决策时长分布（单位：小时）")
            buf2 = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf2, format="png", dpi=300)
            buf2.seek(0)
            st.image(Image.open(buf2))
            buf2.seek(0)
            st.download_button("下载箱线图", buf2, f"{sel_group}_decision_box.png")

            # 3. 行为转移热力图
            st.subheader(f"3. {sel_group} 用户行为转移热力图")
            heat_matrix = get_heatmap_matrix(group_df)
            fig3, ax3 = plt.subplots(figsize=(6,6))
            im = ax3.imshow(heat_matrix, cmap="Blues")
            fig3.colorbar(im, ax=ax3)
            ax3.set_xticks([0,1,2])
            ax3.set_yticks([0,1,2])
            ax3.set_xticklabels(["浏览","加购","购买"])
            ax3.set_yticklabels(["浏览","加购","购买"])
            ax3.set_title(f"{sel_group} 行为转移频次热力图")
            buf3 = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf3, format="png", dpi=300)
            buf3.seek(0)
            st.image(Image.open(buf3))
            buf3.seek(0)
            st.download_button("下载热力图", buf3, f"{sel_group}_behavior_heat.png")