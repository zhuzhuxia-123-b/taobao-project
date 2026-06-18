"""
merge_results.py — 合并所有消融实验结果
=========================================
读取 results/ 目录下所有 ablation_results_*.csv（每个实验单独跑一次会
生成一个带时间戳的文件），按 exp_id 去重合并成一张总表，
exp_id 相同时保留最新（文件名时间戳最大）的一条记录。

用法：
    python -m src.experiments.merge_results

输出：
    results/ablation_results_merged.csv
"""

import glob
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"


def merge_results() -> pd.DataFrame:
    files = sorted(glob.glob(str(RESULTS_DIR / "ablation_results_*.csv")))
    if not files:
        raise FileNotFoundError(f"未找到任何 ablation_results_*.csv，请确认 {RESULTS_DIR} 路径")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["_source_file"] = Path(f).name
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)

    # 按文件名（含时间戳）排序，同一 exp_id 保留最后出现的（即时间最新的）一条
    all_df = all_df.sort_values("_source_file")
    merged = all_df.drop_duplicates(subset="exp_id", keep="last").drop(columns="_source_file")

    # 按 A-G 固定顺序排列
    order = ["A", "B", "C", "D", "E", "F", "G"]
    merged["_order"] = merged["exp_id"].apply(lambda x: order.index(x) if x in order else 99)
    merged = merged.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    return merged


if __name__ == "__main__":
    merged = merge_results()
    out_path = RESULTS_DIR / "ablation_results_merged.csv"
    merged.to_csv(out_path, index=False)

    print(f"合并完成，共 {len(merged)} 组实验，已保存到: {out_path}\n")
    print(merged.to_string(index=False))