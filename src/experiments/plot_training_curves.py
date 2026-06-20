"""
plot_training_curves.py — 训练收敛曲线绘制
=================================================
解析 RecBole 训练日志（log_E_full.txt / log_F_full.txt / log_G_full.txt /
log_D_full.txt），提取每个 epoch 的 train_loss 和 valid NDCG@10，
绘制训练收敛曲线。

日志格式示例：
    19:54:57 [INFO] epoch 0 training [time: 476.39s, train loss: 1489.8571]
    19:59:41 [INFO] epoch 0 evaluating [time: 283.59s, valid_score: 0.600600]
    ndcg@10 : 0.6006    hit@10 : 0.9725    recall@10 : 0.5701

用法：
    python -m src.experiments.plot_training_curves \
        --logs log_D_full.txt log_E_full.txt log_F_full.txt log_G_full.txt \
        --labels D E F G

输出：
    results/figures/fig_training_loss.png      （训练loss曲线，对数坐标）
    results/figures/fig_training_ndcg.png      （验证集NDCG@10曲线）
"""

import argparse
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ========== 中文字体 ==========
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
# ===============================================================

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RE = re.compile(r"epoch (\d+) training \[.*train loss:\s*([\d.]+)\]")
EVAL_RE  = re.compile(r"epoch (\d+) evaluating \[.*valid_score:\s*([\d.]+)\]")
NDCG_RE  = re.compile(r"ndcg@10\s*:\s*([\d.]+)")

# ANSI 颜色/控制转义序列，例如 \x1b[1;35m \x1b[0m
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(line: str) -> str:
    return ANSI_RE.sub("", line)


def parse_log(path: str):
    """
    解析单个日志文件，返回:
        epochs       : [0, 1, 2, ...]
        train_losses : [1489.8571, 624.9938, ...]
        valid_ndcgs  : [0.6006, 0.6338, ...]   （来自 "ndcg@10 : x.xxxx" 行，
                                                  比 valid_score 更精确，两者
                                                  通常一致，valid_score 是
                                                  valid_metric 指定的指标）
    """
    epochs, train_losses, valid_ndcgs = [], [], []
    pending_epoch = None

    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = _strip_ansi(line)
            m_train = TRAIN_RE.search(line)
            if m_train:
                ep, loss = int(m_train.group(1)), float(m_train.group(2))
                pending_epoch = ep
                # 先占位，等evaluating行/ndcg行出现再补valid_ndcg
                epochs.append(ep)
                train_losses.append(loss)
                valid_ndcgs.append(None)
                continue

            m_ndcg = NDCG_RE.search(line)
            if m_ndcg and epochs and valid_ndcgs[-1] is None:
                valid_ndcgs[-1] = float(m_ndcg.group(1))
                continue

    return epochs, train_losses, valid_ndcgs


def plot_training_loss(curves: dict, save_path: str = None):
    """
    curves: {label: (epochs, train_losses, valid_ndcgs)}
    训练loss曲线，y轴对数坐标（loss从~1500降到个位数，线性坐标看不出后期变化）
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, (epochs, losses, _) in curves.items():
        ax.plot(epochs, losses, marker="o", markersize=4, linewidth=1.5, label=label)

    ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontproperties=chinese_font, fontsize=11)
    ax.set_ylabel("训练损失 (train loss, 对数坐标)", fontproperties=chinese_font, fontsize=11)
    ax.set_title("MB-GCL-SASRec 各消融组训练损失收敛曲线",
                  fontproperties=chinese_font, fontsize=12)
    ax.legend(fontsize=10, prop=chinese_font)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    out = save_path or str(FIG_DIR / "fig_training_loss.png")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[图] 训练loss曲线 已保存: {out}")


def plot_validation_ndcg(curves: dict, save_path: str = None):
    """
    curves: {label: (epochs, train_losses, valid_ndcgs)}
    验证集NDCG@10曲线
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, (epochs, _, ndcgs) in curves.items():
        # 过滤掉None
        xs = [e for e, n in zip(epochs, ndcgs) if n is not None]
        ys = [n for n in ndcgs if n is not None]
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.5, label=label)

    ax.set_xlabel("Epoch", fontproperties=chinese_font, fontsize=11)
    ax.set_ylabel("验证集 NDCG@10", fontproperties=chinese_font, fontsize=11)
    ax.set_title("MB-GCL-SASRec 各消融组验证集NDCG@10变化曲线",
                  fontproperties=chinese_font, fontsize=12)
    ax.legend(fontsize=10, prop=chinese_font)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = save_path or str(FIG_DIR / "fig_training_ndcg.png")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[图] 验证集NDCG曲线 已保存: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", required=True, help="日志文件路径列表")
    parser.add_argument("--labels", nargs="+", required=True, help="对应的图例标签，需与logs等长")
    args = parser.parse_args()

    assert len(args.logs) == len(args.labels), "--logs 和 --labels 数量必须一致"

    curves = {}
    for log_path, label in zip(args.logs, args.labels):
        if not os.path.exists(log_path):
            print(f"[跳过] 文件不存在: {log_path}")
            continue
        epochs, losses, ndcgs = parse_log(log_path)
        if not epochs:
            print(f"[跳过] {log_path} 未解析到任何 epoch 数据，检查日志格式")
            continue
        curves[label] = (epochs, losses, ndcgs)
        print(f"[{label}] 解析到 {len(epochs)} 个epoch，"
              f"train_loss范围 [{min(losses):.4f}, {max(losses):.4f}]，"
              f"valid_ndcg范围 [{min(n for n in ndcgs if n is not None):.4f}, "
              f"{max(n for n in ndcgs if n is not None):.4f}]")

    if not curves:
        print("没有任何有效曲线数据，退出。")
        return

    plot_training_loss(curves)
    plot_validation_ndcg(curves)


if __name__ == "__main__":
    main()