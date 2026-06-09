import pandas as pd
import numpy as np
from datetime import datetime

raw_data_path = r"D:\\社会计算\\taobao-project\\data\\raw\\UserBehavior.csv"
processed_dir = r"D:\\社会计算\\taobao-project\\data\\processed"

df = pd.read_csv(
    raw_data_path,
    names=["user_id", "item_id", "category_id", "behavior_type", "timestamp"],
    header=None
)

df = df.drop_duplicates()
df = df.dropna()

df["time"] = pd.to_datetime(df["timestamp"], unit="s")
df["hour"] = df["time"].dt.hour
df["date"] = df["time"].dt.date

df["user_idx"] = df["user_id"].astype("category").cat.codes
df["item_idx"] = df["item_id"].astype("category").cat.codes
df["cate_idx"] = df["category_id"].astype("category").cat.codes

behavior_map = {"pv": 1, "cart": 2, "buy": 3, "fav": 4}
df["behavior_code"] = df["behavior_type"].map(behavior_map)

df.to_csv(f"{processed_dir}\\data_cleaned.csv", index=False)
id_map = df[["user_id", "user_idx", "item_id", "item_idx", "category_id", "cate_idx"]].drop_duplicates()
id_map.to_pickle(f"{processed_dir}\\id_mapping.pkl")

print("✅ 数据清洗完成！")