import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm

print("⏳ 读取 chained.csv...")
df = pd.read_csv(r"D:\\社会计算\\taobao-project\\data\\processed\\chained.csv")
print(f"✅ 数据读取完成，共 {len(df):,} 行")

print("⏳ 按用户聚合特征...")
user_feature = df.groupby("user_idx").agg(
    weight_final=("weight_final", "sum"),
    buy_count=("behavior_code", lambda x: (x == 3).sum()),
    item_count=("item_idx", "nunique"),
    category_count=("category_id", "nunique")
).fillna(0)

user_feature = user_feature.reset_index()
print(f"✅ 用户特征提取完成，共 {len(user_feature):,} 个用户")

print("⏳ KMeans 用户分群 (K=4)...")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
user_feature["group"] = kmeans.fit_predict(user_feature)
print("✅ KMeans 聚类完成")

user_feature.to_pickle(r"D:\\社会计算\\taobao-project\\data\\processed\\user_groups.pkl")
user_feature.to_csv(r"D:\\社会计算\\taobao-project\\data\\processed\\user_groups.csv", index=False, encoding="utf-8-sig")
print("✅ user_cluster 完成 → 给D同学")
print(f"📊 各群体用户数：")
print(user_feature["group"].value_counts().sort_index())
print()
print("各群体特征均值：")
print(user_feature.groupby("group")[["weight_final", "buy_count", "item_count", "category_count"]].mean())