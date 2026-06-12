import pandas as pd
df = pd.read_csv(r"D:\\社会计算\\taobao-project\\data\\processed\\data_cleaned.csv")

def get_weight(code):
    if code == 1: return 1.0
    elif code == 2: return 3.0
    elif code == 3: return 5.0
    elif code == 4: return 2.0
    else: return 0.0

df["weight_final"] = df["behavior_code"].apply(get_weight)
df.to_csv(r"D:\\社会计算\\taobao-project\\data\\processed\\data_weighted.csv", index=False)
print("✅ 权重计算完成！")