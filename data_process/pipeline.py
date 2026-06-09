import pandas as pd

df = pd.read_csv(r"D:\\社会计算\\taobao-project\\data\\processed\\chained.csv")

inter = df[["user_idx", "item_idx", "behavior_code", "weight_final"]]
inter.columns = ["user_id", "item_id", "label", "weight"]
inter.to_csv(
    r"D:\\社会计算\\taobao-project\\data\\processed\\taobao.inter",
    sep="\t", index=False
)

print("✅ pipeline 完成 → taobao.inter 已生成 to C")