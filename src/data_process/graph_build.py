import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, save_npz
from tqdm import tqdm

DATA_DIR = r"D:\社会计算\taobao-project\data\processed"
SOURCE_CSV = f"{DATA_DIR}\\chained.csv"
OUT_PV_GRAPH = f"{DATA_DIR}\\graph_pv.npz"
OUT_CART_GRAPH = f"{DATA_DIR}\\graph_cart.npz"
OUT_BUY_GRAPH = f"{DATA_DIR}\\graph_buy.npz"

def build_single_graph(df, save_path, desc):
    df_graph = df[["user_idx", "item_idx"]].copy()
    df_graph = df_graph.dropna(subset=["user_idx", "item_idx"])
    df_graph = df_graph.reset_index(drop=True)

    total_lines = len(df_graph)
    if total_lines == 0:
        print(f"❌ 无有效交互数据，跳过生成：{save_path}")
        return

    print(f"✅ 有效交互行数：{total_lines:,}")

    user_arr = df_graph["user_idx"].astype(int).values
    item_arr = df_graph["item_idx"].astype(int).values

    n_users = user_arr.max() + 1
    n_items = item_arr.max() + 1
    mat_shape = (n_users, n_items)

    ones = np.ones_like(user_arr, dtype=np.float32)
    adj_mat = coo_matrix((ones, (user_arr, item_arr)), shape=mat_shape)

    save_npz(save_path, adj_mat)
    print(f"✅ 图文件已生成 | 形状: {mat_shape} | 有效交互边数：{adj_mat.nnz:,}\n")


if __name__ == "__main__":
    print("=" * 60)
    print("开始执行建图流程，读取数据源：", SOURCE_CSV)
    print("=" * 60)

    print("⏳ 读取 chained.csv...")
    df_all = pd.read_csv(SOURCE_CSV)
    print(f"✅ 数据读取完成，共 {len(df_all):,} 行")
    print("各行为类型数量统计：")
    print(df_all["behavior_type"].value_counts())
    print("-" * 60)

    df_pv = df_all[df_all["behavior_type"] == "pv"]
    df_cart = df_all[df_all["behavior_type"] == "cart"]
    df_buy = df_all[df_all["behavior_type"] == "buy"]

    print("--- 正在构建 浏览(pv) 图 ---")
    build_single_graph(df_pv, OUT_PV_GRAPH, "pv")

    print("--- 正在构建 加购(cart) 图 ---")
    build_single_graph(df_cart, OUT_CART_GRAPH, "cart")

    print("--- 正在构建 购买(buy) 图 ---")
    build_single_graph(df_buy, OUT_BUY_GRAPH, "buy")

    print("=" * 60)
    print("✅ 全部建图任务执行完毕！")
    print("=" * 60)
