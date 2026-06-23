import pandas as pd
import numpy as np
from tqdm import tqdm

columns = ['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp']

print("⏳ 读取原始数据...")
chunk_size = 5000000
all_users = set()
sampled_df = []

for i, chunk in enumerate(pd.read_csv(r"D:\社会计算\taobao-project\data\raw\UserBehavior.csv", names=columns, chunksize=chunk_size)):
    print(f"  处理第 {i+1} 块...")
    chunk_users = set(chunk['user_id'].unique())
    all_users.update(chunk_users)
    sampled_df.append(chunk)
    
    if len(all_users) >= 100000:
        print(f"✅ 已收集到 {len(all_users):,} 个用户")
        break

print(f"✅ 总共收集到 {len(all_users):,} 个用户")

print("⏳ 抽取10万用户...")
sampled_users = np.random.choice(list(all_users), size=100000, replace=False)

print("⏳ 筛选抽样用户的数据...")
full_df = pd.concat(sampled_df)
filtered_df = full_df[full_df['user_id'].isin(sampled_users)]

print(f"✅ 抽样后数据量: {len(filtered_df):,} 行")

print("⏳ 保存抽样数据...")
filtered_df.to_csv(r"D:\社会计算\taobao-project\data\raw\user_behavior_sample.csv", index=False)

print("✅ 抽样完成！")
print("文件路径: data/raw/user_behavior_sample.csv")