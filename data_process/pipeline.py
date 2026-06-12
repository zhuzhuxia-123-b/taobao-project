import pandas as pd

df = pd.read_csv(r"D:\\社会计算\\taobao-project\\data\\processed\\chained.csv")

inter_df = df[["user_id", "item_id", "behavior_type", "timestamp"]].copy()
inter_df = inter_df.rename(columns={"behavior_type": "label"})

output_path = r"D:\社会计算\taobao-project\data\processed\taobao.inter"
inter_df.to_csv(output_path, index=False)

print(f"✅ 带时间戳的 taobao.inter 已生成！文件路径：\n{output_path}")
print("文件字段：user_id, item_id, label(behavior_type), timestamp")
