import pandas as pd

df = pd.read_csv(r"D:\\社会计算\\taobao-project\\data\\processed\\data_weighted.csv")

df["behavior_code"] = pd.to_numeric(df["behavior_code"], errors="coerce").fillna(0).astype(int)
df["time"] = pd.to_datetime(df["timestamp"], unit="s")
df = df.sort_values(["user_id", "time"])

df["time_diff"] = df.groupby("user_id")["time"].diff()
df["gap_6h"] = df["time_diff"] > pd.Timedelta(hours=6)
df["chain_id"] = df.groupby("user_id")["gap_6h"].cumsum()

chain_stats = df.groupby(["user_id", "chain_id"]).agg(
    has_buy=("behavior_code", lambda x: (x == 3).sum() > 0),
    has_cart=("behavior_code", lambda x: (x == 2).sum() > 0),
    action_count=("behavior_code", "count"),
    max_gap=("time_diff", "max")
).reset_index()

def get_state(row):
    if row["has_buy"]:
        return 1, "closed"
    elif row["has_cart"]:
        return 0, "intent"
    elif row["action_count"] >= 2:
        return 0, "exploring"
    else:
        return 0, "latent"

chain_stats[["closure_label", "need_state"]] = chain_stats.apply(
    lambda x: pd.Series(get_state(x)), axis=1
)
chain_stats.loc[chain_stats["max_gap"] > pd.Timedelta(hours=24), "need_state"] = "dormant"

df = df.merge(chain_stats[["user_id", "chain_id", "closure_label", "need_state"]],
              on=["user_id", "chain_id"], how="left")

df.to_csv(r"D:\\社会计算\\taobao-project\\data\\processed\\chained.csv", index=False)
print("✅ need_chain 运行完成！已输出 chained.csv")