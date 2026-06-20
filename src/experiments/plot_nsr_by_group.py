"""
plot_nsr_by_group.py — NSR分群对比图
=========================================
读取 results/nsr_by_group.csv，绘制 D/E/F/G 四组模型在
四类用户群体（高活跃高购买/高活跃低购买/低活跃高购买/低活跃低购买）
上的 NSR@10 分组柱状图。


"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

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
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

GROUP_ORDER = ["高活跃高购买", "高活跃低购买", "低活跃高购买", "低活跃低购买"]
EXP_ORDER = ["D", "E", "F", "G"]
EXP_LABELS = {
    "D": "D: 完整模型",
    "E": "E: 去对比学习",
    "F": "F: 去图卷积",
    "G": "G: 去需求闭合",
}


def plot_nsr_by_group(df: pd.DataFrame, save_path: str = None):
    df = df.set_index("exp_id").loc[EXP_ORDER, GROUP_ORDER]

    x = np.arange(len(GROUP_ORDER))
    width = 0.18
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, exp_id in enumerate(EXP_ORDER):
        vals = df.loc[exp_id].astype(float).values
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=EXP_LABELS[exp_id], color=colors[i], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.annotate(f"{v:.4f}", xy=(bar.get_x() + bar.get_width() / 2, v),
                         xytext=(0, 2), textcoords="offset points",
                         ha="center", fontsize=7, fontproperties=chinese_font, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_ORDER, fontproperties=chinese_font, fontsize=11)
    ax.set_ylabel("NSR@10", fontproperties=chinese_font, fontsize=11)
    ax.set_title("各消融组在不同用户群体上的需求满足率(NSR@10)对比",
                  fontproperties=chinese_font, fontsize=12)
    ax.legend(fontsize=9, loc="upper right", prop=chinese_font)
    ax.set_ylim(0, df.values.astype(float).max() * 1.35)
    plt.tight_layout()

    out = save_path or str(FIG_DIR / "fig_nsr_by_group.png")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[图] NSR分群对比图 已保存: {out}")


def main():
    csv_path = RESULTS_DIR / "nsr_by_group.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到 {csv_path}，请先运行 compute_nsr_by_group.py")

    df = pd.read_csv(csv_path)
    plot_nsr_by_group(df)


if __name__ == "__main__":
    main()