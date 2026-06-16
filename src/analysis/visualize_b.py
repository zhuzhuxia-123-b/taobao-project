"""
visualize_b.py — 模型相关可视化（B同学负责的三张图）
======================================================

图1: 注意力权重热图   — 验证 buy/cart 位置的注意力权重更高
图2: 对比学习 t-SNE  — 验证强/弱视图的序列表示在空间上有效分离
图3: 图卷积效果对比图 — 低活跃用户加入图卷积前后推荐指标变化

用法（有真实模型时）：
    python visualize_b.py --mode real --model_path best_model.pth --data_path data/processed/B

用法（无模型时，用模拟数据生成示意图）：
    python visualize_b.py --mode demo
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互后端，避免显示问题
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import matplotlib.font_manager as fm

# ========== 中文字体（自动检测，Windows/Mac/Linux 通用）==========
_CANDIDATE_FONTS = [
    # Windows
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    # macOS
    "/System/Library/Fonts/PingFang.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    # Linux
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
_font_path = next((p for p in _CANDIDATE_FONTS if os.path.exists(p)), None)
if _font_path:
    fm.fontManager.addfont(_font_path)
    chinese_font = fm.FontProperties(fname=_font_path)
else:
    # 没有找到中文字体：用系统默认字体，中文会显示方块但不会报错
    chinese_font = fm.FontProperties()
    print("[警告] 未找到中文字体，图表中文可能显示为方块。"
          "Windows用户请确认 C:/Windows/Fonts/ 下有 msyh.ttc 或 simhei.ttf。")
plt.rcParams['axes.unicode_minus'] = False
# ===============================================================

SAVE_DIR = "results/figures"

# ========================= 图1 =========================
def plot_attention_heatmap(attn_weights=None, behavior_seq=None, save_path=None):
    L = 10
    if attn_weights is None:
        rng = np.random.default_rng(42)
        base = np.exp(np.linspace(-2, 0, L))
        attn_weights = np.outer(base, base)
        strong_pos = [3, 6, 8]
        for p in strong_pos:
            attn_weights[:, p] *= 3.5
        attn_weights += rng.uniform(0, 0.05, (L, L))
        attn_weights = np.exp(attn_weights) / np.exp(attn_weights).sum(axis=1, keepdims=True)
    if behavior_seq is None:
        behavior_seq = ["pv", "pv", "pv", "cart", "pv", "pv", "buy", "pv", "buy", "pv"]
    L = len(behavior_seq)
    behavior_color = {"pv": "#9DB8D2", "cart": "#E8A838", "buy": "#D94F3D"}
    labels = [f"t{i+1}\n({b})" for i, b in enumerate(behavior_seq)]
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(attn_weights[:L, :L], cmap="YlOrRd", aspect="auto", vmin=0, interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax, label="注意力权重", shrink=0.8)
    cbar.ax.yaxis.label.set_fontproperties(chinese_font)
    cbar.ax.tick_params(labelsize=8)
    ax.set_xticks(range(L))
    ax.set_yticks(range(L))
    fp8 = fm.FontProperties(fname=_font_path, size=8) if _font_path else fm.FontProperties(size=8)
    ax.set_xticklabels(labels, fontproperties=fp8)
    ax.set_yticklabels(labels, fontproperties=fp8)
    ax.set_xlabel("Key 位置（信息来源）", fontsize=11, fontproperties=chinese_font)
    ax.set_ylabel("Query 位置（当前预测步）", fontsize=11, fontproperties=chinese_font)
    ax.set_title("注意力权重热图\nbuy/cart 位置作为 Key 时接收到更多注意力", fontsize=12, fontproperties=chinese_font)
    for j, b in enumerate(behavior_seq):
        if b in ("cart", "buy"):
            color = behavior_color[b]
            for edge in ["bottom", "top"]:
                ax.axhline(j - 0.5, color=color, linewidth=0.6, alpha=0.6)
                ax.axhline(j + 0.5, color=color, linewidth=0.6, alpha=0.6)
            ax.axvline(j - 0.5, color=color, linewidth=1.2, alpha=0.8)
            ax.axvline(j + 0.5, color=color, linewidth=1.2, alpha=0.8)
    legend_handles = [
        mpatches.Patch(color=behavior_color["pv"],   label="pv（浏览）"),
        mpatches.Patch(color=behavior_color["cart"],  label="cart（加购）"),
        mpatches.Patch(color=behavior_color["buy"],   label="buy（购买）"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9, prop=chinese_font, framealpha=0.9)
    plt.tight_layout()
    _save(fig, save_path or os.path.join(SAVE_DIR, "fig_attention_heatmap.png"))
    plt.close(fig)
    print("  [图1] 注意力权重热图 已保存")

# ========================= 图2 =========================
def plot_cl_tsne(h_strong=None, h_weak=None, user_ids=None, n_users_show=8, save_path=None):
    N_PER_USER = 20
    if h_strong is None or h_weak is None:
        rng  = np.random.default_rng(0)
        d    = 64
        N    = n_users_show * N_PER_USER
        centers = rng.normal(0, 3, (n_users_show, d))
        h_strong_list, h_weak_list, uid_list = [], [], []
        for uid, center in enumerate(centers):
            strong = center + rng.normal(0, 0.6, (N_PER_USER, d))
            weak   = center + rng.normal(0, 1.8, (N_PER_USER, d))
            h_strong_list.append(strong)
            h_weak_list.append(weak)
            uid_list.extend([uid] * N_PER_USER)
        h_strong  = np.vstack(h_strong_list)
        h_weak    = np.vstack(h_weak_list)
        user_ids  = np.array(uid_list)
    n_users_show = min(n_users_show, len(np.unique(user_ids)))
    selected_users = np.unique(user_ids)[:n_users_show]
    mask = np.isin(user_ids, selected_users)
    h_all = np.vstack([h_strong[mask], h_weak[mask]])
    print("  [图2] 正在计算 t-SNE（可能需要几秒）…")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=800)
    coords = tsne.fit_transform(normalize(h_all))
    N_half = mask.sum()
    coords_strong = coords[:N_half]
    coords_weak   = coords[N_half:]
    uids_plot     = user_ids[mask]
    palette = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, uid in enumerate(selected_users):
        c = palette[i % 10]
        idx = uids_plot == uid
        ax.scatter(coords_strong[idx, 0], coords_strong[idx, 1],
                   s=55, c=[c], marker="o", alpha=0.85,
                   edgecolors="white", linewidths=0.4, zorder=3)
        ax.scatter(coords_weak[idx, 0], coords_weak[idx, 1],
                   s=55, marker="^", alpha=0.55,
                   edgecolors=[c], linewidths=1.0, facecolors="none", zorder=2)
        cs_center = coords_strong[idx].mean(axis=0)
        cw_center = coords_weak[idx].mean(axis=0)
        ax.annotate("", xy=cw_center, xytext=cs_center,
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.0, alpha=0.6))
    legend_items = [
        mpatches.Patch(color="gray", label="实心圆 = 强视图（含buy/cart）"),
        mpatches.Patch(color="none", label="空心三角 = 弱视图（仅pv）",
                       edgecolor="gray", linewidth=1.2),
    ] + [
        mpatches.Patch(color=palette[i % 10], label=f"用户 {uid}")
        for i, uid in enumerate(selected_users)
    ]
    ax.legend(handles=legend_items, fontsize=8, loc="upper right",
              framealpha=0.9, ncol=2, prop=chinese_font)
    ax.set_title("对比学习 t-SNE 聚类图\n同一用户的强/弱视图表示在空间中相互靠近", fontsize=12, fontproperties=chinese_font)
    ax.set_xlabel("t-SNE 维度 1", fontsize=11, fontproperties=chinese_font)
    ax.set_ylabel("t-SNE 维度 2", fontsize=11, fontproperties=chinese_font)
    plt.tight_layout()
    _save(fig, save_path or os.path.join(SAVE_DIR, "fig_cl_tsne.png"))
    plt.close(fig)
    print("  [图2] 对比学习 t-SNE 图 已保存")

# ========================= 图3 =========================
def plot_graph_conv_effect(results=None, save_path=None):
    if results is None:
        results = {
            "groups":      ["低活跃\n(≤5次)", "中活跃\n(6-20次)", "高活跃\n(>20次)"],
            "hit_no_gcn":  [0.112, 0.213, 0.318],
            "hit_gcn":     [0.187, 0.241, 0.334],
            "ndcg_no_gcn": [0.068, 0.128, 0.195],
            "ndcg_gcn":    [0.118, 0.148, 0.211],
        }
    groups      = results["groups"]
    hit_no_gcn  = np.array(results["hit_no_gcn"])
    hit_gcn     = np.array(results["hit_gcn"])
    ndcg_no_gcn = np.array(results["ndcg_no_gcn"])
    ndcg_gcn    = np.array(results["ndcg_gcn"])
    x     = np.arange(len(groups))
    width = 0.2
    colors = {
        "no_gcn_hit":  "#9DB8D2",
        "gcn_hit":     "#2171B5",
        "no_gcn_ndcg": "#FDAE6B",
        "gcn_ndcg":    "#D94801",
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - 1.5*width, hit_no_gcn,  width, label="Hit@10（无图卷积）", color=colors["no_gcn_hit"],  alpha=0.85)
    b2 = ax.bar(x - 0.5*width, hit_gcn,     width, label="Hit@10（有图卷积）", color=colors["gcn_hit"],     alpha=0.85)
    b3 = ax.bar(x + 0.5*width, ndcg_no_gcn, width, label="NDCG@10（无图卷积）",color=colors["no_gcn_ndcg"], alpha=0.85)
    b4 = ax.bar(x + 1.5*width, ndcg_gcn,    width, label="NDCG@10（有图卷积）",color=colors["gcn_ndcg"],    alpha=0.85)
    for i in range(len(groups)):
        hit_gain  = (hit_gcn[i]  - hit_no_gcn[i])  / hit_no_gcn[i]  * 100
        ndcg_gain = (ndcg_gcn[i] - ndcg_no_gcn[i]) / ndcg_no_gcn[i] * 100
        ax.annotate(f"+{hit_gain:.0f}%", xy=(x[i] - 0.5*width, hit_gcn[i]),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=8, color=colors["gcn_hit"], fontweight="bold",
                    fontproperties=chinese_font)
        ax.annotate(f"+{ndcg_gain:.0f}%", xy=(x[i] + 1.5*width, ndcg_gcn[i]),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=8, color=colors["gcn_ndcg"], fontweight="bold",
                    fontproperties=chinese_font)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11, fontproperties=chinese_font)
    ax.set_ylabel("指标值", fontsize=11, fontproperties=chinese_font)
    ax.set_title("图卷积对不同活跃度用户的推荐效果提升\n低活跃用户受益更显著（缓解数据稀疏问题）", fontsize=12, fontproperties=chinese_font)
    ax.legend(fontsize=9, loc="upper left", prop=chinese_font)
    ax.set_ylim(0, max(hit_gcn.max(), ndcg_gcn.max()) * 1.35)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    plt.tight_layout()
    _save(fig, save_path or os.path.join(SAVE_DIR, "fig_graph_conv_effect.png"))
    plt.close(fig)
    print("  [图3] 图卷积效果对比图 已保存")

def _save(fig, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)

def generate_all_demo():
    print("生成演示图（模拟数据）...")
    plot_attention_heatmap()
    plot_cl_tsne()
    plot_graph_conv_effect()
    print(f"\n全部完成，图片保存在 {SAVE_DIR}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="demo", choices=["demo", "real"])
    parser.add_argument("--model_path", default="results/best_model.pth")
    parser.add_argument("--data_path", default="data/processed/B")
    args = parser.parse_args()
    if args.mode == "demo":
        generate_all_demo()
    else:
        print("真实模型模式尚未实现，请使用 --mode demo")