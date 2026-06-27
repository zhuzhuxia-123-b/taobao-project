"""
visualize_b.py — 用真实用户序列重新画图1(t-SNE)和图2(注意力热图)
位置：src/analysis/visualize_b.py
用法：在项目根目录执行 python -m src.analysis.visualize_b
================================================================
item_seq / behavior_seq 来自taobao.inter 里真实用户的购物行为

"""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from src.model.mb_gcl_sasrec import MBGCLSASRec
import matplotlib.font_manager as fm

# ── 字体 ──
font_path = "C:/Windows/Fonts/msyh.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    CN = fm.FontProperties(fname=font_path)
    CN8 = fm.FontProperties(fname=font_path, size=8)
else:
    CN = CN8 = fm.FontProperties()
plt.rcParams["axes.unicode_minus"] = False

CKPT = os.path.join(ROOT, "saved", "MBGCLSASRec-Jun-15-2026_16-51-24.pth")
INTER_PATH = os.path.join(ROOT, "data", "processed", "taobao.inter")
MAX_SEQ_LEN = 50
N_USERS_TO_PLOT = 20   # 挑多少个真实用户来画图

config = {
    "n_users": 987994,
    "n_items": 4162024,
    "embed_dim": 64,
    "n_layers": 2,
    "n_heads": 2,
    "max_seq_len": 50,
    "graph_layers": 2,
    "lambda1": 0.1,
    "lambda2": 0.1,
    "dropout": 0.1,
}

# ──────────────────────────────────────────────────────────
# 第一步：从 taobao.inter 里读真实用户序列
# ──────────────────────────────────────────────────────────

def load_real_sequences(inter_path: str, max_len: int = 50, n_users: int = 12):
    """
    读取 taobao.inter，挑选若干个序列长度适中、且同时含有
    buy/cart/pv 三种行为的真实用户，构造 (item_seq, behavior_seq, seq_len)。

    返回
    ----
    item_seq      : LongTensor [n_users, max_len]
    behavior_seq  : LongTensor [n_users, max_len]
    seq_len       : LongTensor [n_users]
    user_ids      : LongTensor [n_users]
    """
    print(f"读取 {inter_path} ...")

    # 分块读取，避免一次性加载1000万+行造成内存峰值
    # 每个 chunk 单独清洗 + 降dtype，再拼接，比一次性对全表操作省内存
    chunks = []
    chunk_size = 500_000
    total_dropped = 0
    for chunk in pd.read_csv(inter_path, sep="\t", chunksize=chunk_size):
        chunk.columns = [c.split(":")[0] for c in chunk.columns]
        before = len(chunk)
        chunk = chunk.dropna(subset=["user_id", "item_id", "behavior", "timestamp"])
        total_dropped += before - len(chunk)

        chunk["user_id"]   = chunk["user_id"].astype("int32")
        chunk["item_id"]   = chunk["item_id"].astype("int32")
        chunk["behavior"]  = chunk["behavior"].astype("int8")
        chunk["timestamp"] = chunk["timestamp"].astype("int64")

        # 只保留后续真正需要的列，减少内存占用
        chunks.append(chunk[["user_id", "item_id", "behavior", "timestamp"]])

    if total_dropped > 0:
        print(f"  丢弃 {total_dropped} 行含缺失值的数据")

    df = pd.concat(chunks, ignore_index=True)
    del chunks

    # ── 选用户阶段：用 value_counts 和简单聚合，避免 groupby.get_group
    #    （get_group 第一次调用会为全表建立分组索引，1000万行时内存吃紧）
    print("统计每个用户的行为类型集合与序列长度...")
    user_stats = df.groupby("user_id", sort=False)["behavior"].agg(
        lambda s: frozenset(s.unique())
    )
    user_lens = df.groupby("user_id", sort=False).size()

    selected_users = []
    for uid in user_lens.index:
        behaviors = user_stats.loc[uid]
        seq_len_real = user_lens.loc[uid]
        # 挑选条件：序列长度在 [10, max_len] 之间，且同时含 pv(0)/cart(1)/buy(2)
        if 10 <= seq_len_real <= max_len and {0, 1, 2}.issubset(behaviors):
            selected_users.append(uid)
        if len(selected_users) >= n_users:
            break

    if len(selected_users) < n_users:
        print(f"⚠️ 只找到 {len(selected_users)} 个满足条件的用户（序列长度10~{max_len}且含pv/cart/buy），"
              f"将放宽条件继续找...")
        for uid in user_lens.index:
            if uid in selected_users:
                continue
            behaviors = user_stats.loc[uid]
            seq_len_real = user_lens.loc[uid]
            if {1, 2}.issubset(behaviors) and seq_len_real >= 5:
                selected_users.append(uid)
            if len(selected_users) >= n_users:
                break

    selected_users = selected_users[:n_users]
    print(f"选中 {len(selected_users)} 个真实用户: {selected_users}")

    # 用布尔索引直接筛选这12个用户的行，而不是用 groupby.get_group
    # （isin 在 1000万行上做一次布尔过滤，比 groupby 建索引省内存）
    mask = df["user_id"].isin(selected_users)
    df_selected = df.loc[mask].copy()
    del df, mask  # 释放大表内存，后面只需要 df_selected（行数很少）

    item_seq_list, behavior_seq_list, seq_len_list = [], [], []
    for uid in selected_users:
        g = df_selected[df_selected["user_id"] == uid].sort_values("timestamp").tail(max_len)
        items = g["item_id"].astype(int).tolist()
        behs  = g["behavior"].astype(int).tolist()
        L = len(items)

        # 左侧 padding 到 max_len（item_id=0 是 padding）
        pad_len = max_len - L
        items = [0] * pad_len + items
        behs  = [0] * pad_len + behs

        item_seq_list.append(items)
        behavior_seq_list.append(behs)
        seq_len_list.append(L)

    item_seq     = torch.tensor(item_seq_list, dtype=torch.long)
    behavior_seq = torch.tensor(behavior_seq_list, dtype=torch.long)
    seq_len      = torch.tensor(seq_len_list, dtype=torch.long)
    user_ids     = torch.tensor(selected_users, dtype=torch.long)

    return item_seq, behavior_seq, seq_len, user_ids


# ──────────────────────────────────────────────────────────
# 第二步：加载模型
# ──────────────────────────────────────────────────────────

print("加载模型...")
model = MBGCLSASRec(config)
device = torch.device("cpu")
# mmap=True：避免一次性把3.45GB权重读入内存，按需从磁盘映射，大幅降低内存峰值
ckpt = torch.load(CKPT, map_location=device, weights_only=False, mmap=True)
state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
model.load_state_dict(state, strict=False)
model.to(device)
model.eval()

# ──────────────────────────────────────────────────────────
# 第三步：读真实数据
# ──────────────────────────────────────────────────────────

item_seq, behavior_seq, seq_len, user_ids = load_real_sequences(
    INTER_PATH, max_len=MAX_SEQ_LEN, n_users=N_USERS_TO_PLOT
)
B = item_seq.size(0)
print(f"真实用户数: {B}, 序列长度: {seq_len.tolist()}")

# ──────────────────────────────────────────────────────────
# 第四步：提取隐状态（强视图 vs 弱视图）
# ──────────────────────────────────────────────────────────

print("提取隐状态...")
with torch.no_grad():
    h_strong = model._get_last_hidden(
        item_seq.to(device), behavior_seq.to(device), seq_len.to(device), None, user_ids.to(device)
    )
    pv_mask = (behavior_seq == 0) & (item_seq != 0)
    pv_seq  = item_seq * pv_mask.long()
    pv_beh  = torch.zeros_like(behavior_seq)
    pv_len  = pv_mask.sum(dim=1).clamp(min=1)
    h_weak = model._get_last_hidden(
        pv_seq.to(device), pv_beh.to(device), pv_len.to(device), None, user_ids.to(device)
    )

print(f"pv_len（弱视图真实长度）: {pv_len.tolist()}")
if (pv_len <= 1).sum() > 0:
    print("⚠️ 注意：部分用户几乎没有纯pv行为，弱视图可能不够有代表性")

# ──────────────────────────────────────────────────────────
# 图1：t-SNE（真实数据版）
# ──────────────────────────────────────────────────────────

print("运行 t-SNE...")
h_all = np.vstack([h_strong.cpu().numpy(), h_weak.cpu().numpy()])
perplexity = max(2, min(5, B - 1))  # 用户数较少时降低 perplexity
coords = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(h_all)
cs, cw = coords[:B], coords[B:]

fig, ax = plt.subplots(figsize=(9, 7))
colors = plt.cm.tab10.colors
for i in range(B):
    c = colors[i % 10]
    ax.scatter(cs[i, 0], cs[i, 1], s=60, c=[c], marker="o", edgecolors="white", linewidths=0.5,
               label=f"用户{user_ids[i].item()}" if i < 10 else "")
    ax.scatter(cw[i, 0], cw[i, 1], s=60, c=[c], marker="^", edgecolors=c, linewidths=1.0, facecolors="none")
    ax.annotate("", xy=cw[i], xytext=cs[i], arrowprops=dict(arrowstyle="->", color=c, lw=1.0, alpha=0.6))

ax.legend(loc="best", fontsize=8, ncol=2, prop=CN8)
ax.set_title("对比学习 t-SNE 聚类图（真实用户数据）\n强视图(圆) vs 弱视图(三角)", fontsize=12, fontproperties=CN)
plt.tight_layout()
OUT_DIR = os.path.join(ROOT, "results", "figures")
os.makedirs(OUT_DIR, exist_ok=True)
plt.savefig(os.path.join(OUT_DIR, "fig_tsne_real.png"))
plt.close()
print("✅ 图1 已保存: results/figures/fig_tsne_real.png")

# ──────────────────────────────────────────────────────────
# 图2：注意力热图（真实数据版，挑序列里行为类型最丰富的一个用户展示）
# ──────────────────────────────────────────────────────────

print("提取注意力权重...")
# 挑一个 pv/cart/buy 都有、且非padding长度适中的用户做展示
show_idx = 0
for i in range(B):
    behs_nonpad = behavior_seq[i, -seq_len[i]:].tolist()
    if len(set(behs_nonpad)) == 3:
        show_idx = i
        break

first_layer = model.attn_layers[0]
orig_forward = first_layer.attn.forward

def patched_forward(q, k, v, **kwargs):
    kwargs["need_weights"] = True
    kwargs["average_attn_weights"] = True
    return orig_forward(q, k, v, **kwargs)

first_layer.attn.forward = patched_forward
weights = None

def hook_fn(module, inp, out):
    global weights
    weights = out[1].detach().cpu()

handle = first_layer.attn.register_forward_hook(hook_fn)
with torch.no_grad():
    _ = model._encode_sequence(
        item_seq[show_idx:show_idx+1].to(device),
        behavior_seq[show_idx:show_idx+1].to(device),
        seq_len[show_idx:show_idx+1].to(device),
        None,
        user_ids[show_idx:show_idx+1].to(device),
    )
handle.remove()
first_layer.attn.forward = orig_forward

if weights is not None:
    w = weights[0, 0].numpy() if weights.dim() == 4 else weights[0].numpy()

    # 只展示真实非padding部分（去掉左侧padding）
    L_real = seq_len[show_idx].item()
    w = w[-L_real:, -L_real:]
    beh = behavior_seq[show_idx, -L_real:].numpy()

    beh_name = {0: "pv", 1: "cart", 2: "buy"}
    labels = [f"{i+1}\n({beh_name.get(int(b), '?')})" for i, b in enumerate(beh)]

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(w, cmap="YlOrRd", aspect="auto", vmin=0)
    plt.colorbar(im, ax=ax, label="注意力权重")
    # ── 精简坐标，同时保证所有 cart/buy 位置都能显示 ──
    # 收集所有 cart(1) 和 buy(2) 的位置
    cart_buy_indices = [i for i, b in enumerate(beh) if beh_name.get(int(b)) in ("cart", "buy")]

    # 每隔 step 个位置显示一个，但保证所有 cart/buy 都在
    step = 2  # 如果序列太长可以改成3
    all_indices = set(cart_buy_indices)  # 所有强信号位置必显示
    # 再补上每隔 step 的普通位置（避免全是强信号导致分布不均）
    for i in range(0, L_real, step):
        all_indices.add(i)

    ticks = sorted(all_indices)
    label_show = [labels[i] for i in ticks]

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(label_show, fontproperties=CN8, fontsize=7, rotation=45, ha='right')
    ax.set_yticklabels(label_show, fontproperties=CN8, fontsize=7)
    ax.set_title(f"注意力权重热图（真实用户 {user_ids[show_idx].item()}）\nbuy/cart 位置(列)权重更高",
                 fontsize=12, fontproperties=CN)
    for j, b in enumerate(beh):
        if beh_name.get(int(b)) in ("cart", "buy"):
            ax.axvline(j - 0.5, color="red", linewidth=1.5, alpha=0.6)
            ax.axvline(j + 0.5, color="red", linewidth=1.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_atten_real.png"), dpi=150)
    plt.close()
    print("✅ 图2 已保存: results/figures/fig_attn_real.png")
else:
    print("⚠️ 注意力权重提取失败")