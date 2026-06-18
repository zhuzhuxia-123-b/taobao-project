"""
plot_ablation_figures.py — 消融实验图表生成
=================================================
读取 results/ablation_results_merged.csv，生成两张核心图：

  图1: 基线对比柱状图（A/B/C/D 在 NDCG@10 / Hit@10 / Recall@10 上的对比）
  图2: 消融阶梯图（D/E/F/G 在 NDCG@10 上的折线，展示各创新模块的边际贡献）

用法：
    python -m src.experiments.plot_ablation_figures

输出：
    results/figures/fig_baseline_comparison.png
    results/figures/fig_ablation_ladder.png
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ========== 中文字体（自动检测，Windows/Mac/Linux 通用）==========
_CANDIDATE_FONTS = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
_font_path = next((p for p in _CANDIDATE_FONTS if os.path.exists(p)), None)
if _font_path:
    fm.fontManager.addfont(_font_path)
    chinese_font = fm.FontProperties(fname=_font_path)
else:
    chinese_font = fm.FontProperties()
    print("[警告] 未找到中文字体，图表中文可能显示为方块。")
plt.rcParams["axes.unicode_minus"] = False
# ===============================================================

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MERGED_PATH = RESULTS_DIR / "ablation_results_merged.csv"


# ========================= 图1：基线对比柱状图 =========================

def plot_baseline_comparison(df: pd.DataFrame, save_path: str = None):
    """
    A/B/C/D 四组在 NDCG@10 / Hit@10 / Recall@10 上的横向对比柱状图。
    """
    target_ids = ["A", "B", "C", "D"]
    sub = df[df["exp_id"].isin(target_ids)].copy()
    sub["exp_id"] = pd.Categorical(sub["exp_id"], categories=target_ids, ordered=True)
    sub = sub.sort_values("exp_id")

    name_map = {
        "A": "A: ItemKNN",
        "B": "B: GRU4Rec",
        "C": "C: SASRec",
        "D": "D: MB-GCL-SASRec\n(完整模型)",
    }
    labels = [name_map.get(x, x) for x in sub["exp_id"]]

    metrics = ["NDCG@10", "Hit@10", "Recall@10"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    x = np.arange(len(sub))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, m in enumerate(metrics):
        vals = sub[m].astype(float).values
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=m, color=colors[i], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.annotate(f"{v:.3f}", xy=(bar.get_x() + bar.get_width() / 2, v),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", fontsize=8, fontproperties=chinese_font)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontproperties=chinese_font, fontsize=10)
    ax.set_ylabel("指标值", fontproperties=chinese_font, fontsize=11)
    ax.set_title("基线模型与完整模型性能对比\n验证序列建模与多模块融合的整体收益",
                  fontproperties=chinese_font, fontsize=12)
    ax.legend(fontsize=9, loc="upper left", prop=chinese_font)
    ax.set_ylim(0, max(sub[metrics].astype(float).max().max() * 1.2, 0.1))
    plt.tight_layout()

    out = save_path or str(FIG_DIR / "fig_baseline_comparison.png")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[图1] 基线对比柱状图 已保存: {out}")


# ========================= 图2：消融阶梯图 =========================

def plot_ablation_ladder(df: pd.DataFrame, metric: str = "NDCG@10", save_path: str = None):
    """
    D（完整模型）与 E/F/G（分别去除对比学习/图卷积/需求闭合）在指定指标上的对比折线图。

    采用“反向消融”设计：D为完整模型（上限），E/F/G分别移除一个模块后，
    指标相对D的下降幅度即为该模块的边际贡献。
    """
    target_ids = ["D", "E", "F", "G"]
    sub = df[df["exp_id"].isin(target_ids)].copy()
    sub["exp_id"] = pd.Categorical(sub["exp_id"], categories=target_ids, ordered=True)
    sub = sub.sort_values("exp_id")

    name_map = {
        "D": "D: 完整模型\n(全部模块)",
        "E": "E: 去除\n对比学习",
        "F": "F: 去除\n图卷积",
        "G": "G: 去除\n需求闭合",
    }
    labels = [name_map.get(x, x) for x in sub["exp_id"]]
    vals = sub[metric].astype(float).values
    d_val = vals[0]  # D 组的值，作为基准

    x = np.arange(len(sub))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(x, vals, marker="o", markersize=10, linewidth=2, color="#C44E52")

    for i, v in enumerate(vals):
        diff = v - d_val
        if i == 0:
            text = f"{v:.4f}\n(基准)"
        else:
            text = f"{v:.4f}\n({diff:+.4f})"
        ax.annotate(text, xy=(x[i], v), xytext=(0, 12), textcoords="offset points",
                     ha="center", fontsize=9, fontproperties=chinese_font,
                     fontweight="bold" if i == 0 else "normal")

    # D 组基准线
    ax.axhline(d_val, color="gray", linestyle="--", linewidth=1, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontproperties=chinese_font, fontsize=10)
    ax.set_ylabel(metric, fontproperties=chinese_font, fontsize=11)
    ax.set_title(
        f"消融实验：各创新模块对 {metric} 的边际贡献\n"
        f"（相对完整模型D的下降幅度，即该模块带来的提升）",
        fontproperties=chinese_font, fontsize=12,
    )

    y_min = min(vals) * 0.95
    y_max = max(vals) * 1.08
    ax.set_ylim(y_min, y_max)
    plt.tight_layout()

    out = save_path or str(FIG_DIR / "fig_ablation_ladder.png")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[图2] 消融阶梯图 已保存: {out}")


# ========================= 主函数 =========================

def main():
    if not MERGED_PATH.exists():
        raise FileNotFoundError(
            f"未找到 {MERGED_PATH}，请先运行: python -m src.experiments.merge_results"
        )

    df = pd.read_csv(MERGED_PATH)
    print("读取到的实验结果：")
    print(df.to_string(index=False))
    print()

    # 检查所需的组是否齐全，缺失则提示但不中断（部分图可以先用现有数据生成）
    available = set(df["exp_id"])

    if {"A", "B", "C", "D"}.issubset(available):
        plot_baseline_comparison(df)
    else:
        missing = {"A", "B", "C", "D"} - available
        print(f"[跳过图1] 缺少基线对比所需的组: {missing}")

    if {"D", "E", "F", "G"}.issubset(available):
        plot_ablation_ladder(df, metric="NDCG@10")
    else:
        missing = {"D", "E", "F", "G"} - available
        print(f"[跳过图2] 缺少消融阶梯图所需的组: {missing}")


if __name__ == "__main__":
    main()