import pandas as pd
from tqdm import tqdm

print("⏳ 读取 chained.csv...")
df = pd.read_csv(r"D:\\社会计算\\taobao-project\\data\\processed\\chained.csv")
print(f"✅ 数据读取完成，共 {len(df):,} 行")

print("⏳ 提取交互数据...")
inter = df[["user_idx", "item_idx", "behavior_code", "weight_final", "timestamp"]]
inter.columns = ["user_id", "item_id", "label", "weight", "timestamp"]

print("⏳ 保存 taobao.inter...")
inter.to_csv(
    r"D:\社会计算\taobao-project\data\processed\taobao.inter",
    sep="\t", index=False
)

print("✅ pipeline 完成 → taobao.inter 已生成 to C")
print(f"📊 交互数据统计：")
print(f"   - 用户数：{inter['user_id'].nunique():,}")
print(f"   - 商品数：{inter['item_id'].nunique():,}")
print(f"   - 交互数：{len(inter):,}")
