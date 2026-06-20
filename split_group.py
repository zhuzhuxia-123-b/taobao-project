import pandas as pd

df = pd.read_csv('data_process/processed/taobao/user_groups.csv')

item_median = df['item_count'].median()
buy_median = df['buy_count'].median()

def assign_group(row):
    high_activity = row['item_count'] >= item_median
    high_buy = row['buy_count'] >= buy_median
    if high_activity and high_buy:
        return '高活跃高购买'
    elif high_activity and not high_buy:
        return '高活跃低购买'
    elif not high_activity and high_buy:
        return '低活跃高购买'
    else:
        return '低活跃低购买'

df['custom_group'] = df.apply(assign_group, axis=1)
print(df['custom_group'].value_counts())
print("\n各分组行为均值：")
print(df.groupby("custom_group")[["buy_count","item_count"]].mean())
df.to_csv("data_process/processed/taobao/user_groups_custom.csv", index=False)
print("\n✅ 新分组文件已保存")
