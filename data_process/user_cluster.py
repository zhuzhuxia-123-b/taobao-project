import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv(r"D:\\社会计算\\taobao-project\\data\\processed\\chained.csv")

user_feature = df.groupby("user_idx").agg({
    "weight_final": "sum",
    "behavior_code": lambda x: (x == 3).sum(),
    "item_idx": "nunique",
    "category_id": "nunique"
}).fillna(0)

kmeans = KMeans(n_clusters=4, random_state=42)
user_feature["group"] = kmeans.fit_predict(user_feature)

user_feature.to_pickle(r"D:\\社会计算\\taobao-project\\data\\processed\\user_groups.pkl")
print("✅ user_cluster 完成 → 给D同学")