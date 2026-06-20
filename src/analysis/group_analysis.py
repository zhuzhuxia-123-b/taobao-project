import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 中文显示配置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 路径配置
BASE_DIR = r"D:\社会计算"
USER_GROUP_CSV = os.path.join(BASE_DIR, "data", "user_groups.csv")
CHAINED_CSV_PATH = os.path.join(BASE_DIR, "data", "chained.csv")
FIG_SAVE_DIR = os.path.join(BASE_DIR, "results", "figures")
os.makedirs(FIG_SAVE_DIR, exist_ok=True)
SAVE_BOX_PATH = os.path.join(FIG_SAVE_DIR, "decision_boxplot.png")
SAVE_HEAT_PATH = os.path.join(FIG_SAVE_DIR, "transition_heatmap.png")
df_behavior = pd.read_csv(CHAINED_CSV_PATH)
df_behavior.columns = df_behavior.columns.str.strip().str.lower()
print("行为表全部列名：", df_behavior.columns.tolist())

df_behavior["action_clean"] = df_behavior["behavior_type"].str.strip().str.upper()
df_behavior["event_time"] = pd.to_datetime(df_behavior["timestamp"])

# 按用户+时间排序
df_behavior = df_behavior.sort_values(["user_idx", "event_time"])
print("行为类型取值：", df_behavior["action_clean"].unique())
df_user_group = pd.read_csv(USER_GROUP_CSV)
df_user_group.columns = df_user_group.columns.str.strip().str.lower()
print("分组表全部列名：", df_user_group.columns.tolist())

# 计算决策间隔（首次PV到首次Buy时长）
def get_decision_gap(group_df):
    pv_mask = group_df["action_clean"] == "PV"
    buy_mask = group_df["action_clean"] == "BUY"
    if not (pv_mask.any() and buy_mask.any()):
        return np.nan
    first_pv = group_df.loc[pv_mask, "event_time"].min()
    first_buy = group_df.loc[buy_mask, "event_time"].min()
    return (first_buy - first_pv).total_seconds() / 3600

user_gap = df_behavior.groupby("user_idx").apply(get_decision_gap).reset_index()
user_gap.columns = ["user_idx", "decision_hour"]
print("同时有PV和Buy行为的用户数量：", user_gap["decision_hour"].notna().sum())

# 关联用户分组
user_gap = user_gap.merge(df_user_group, on="user_idx", how="left")
user_gap_valid = user_gap.dropna(subset=["decision_hour", "group"])
print("箱线图有效绘图样本量：", len(user_gap_valid))

# 绘制箱线图
if len(user_gap_valid) > 0:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="group", y="decision_hour", data=user_gap_valid)
    plt.title("不同用户群体决策时长分布")
    plt.xlabel("用户群体")
    plt.ylabel("首次浏览到购买间隔(小时)")
    plt.tight_layout()
    plt.savefig(SAVE_BOX_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print("箱线图已成功保存")
else:
    print("提示：没有满足条件的有效数据，箱线图无法生成")

#  统计相邻行为转移、绘制热力图
def calc_trans_matrix(group_df):
    actions = group_df["action_clean"].tolist()
    trans = {"PV→Cart": 0, "Cart→Buy": 0, "PV→Buy": 0}
    for i in range(len(actions) - 1):
        curr, nxt = actions[i], actions[i+1]
        if curr == "PV" and nxt == "CART":
            trans["PV→Cart"] += 1
        elif curr == "CART" and nxt == "BUY":
            trans["Cart→Buy"] += 1
        elif curr == "PV" and nxt == "BUY":
            trans["PV→Buy"] += 1
    return pd.Series(trans)

trans_by_user = df_behavior.groupby("user_idx").apply(calc_trans_matrix).reset_index()
trans_by_user.columns = ["user_idx", "PV→Cart", "Cart→Buy", "PV→Buy"]
trans_by_user = trans_by_user.merge(df_user_group, on="user_idx", how="left")
trans_valid = trans_by_user.dropna(subset=["group"])
print("热力图有效绘图样本量：", len(trans_valid))

group_trans = trans_valid.groupby("group")[["PV→Cart", "Cart→Buy", "PV→Buy"]].mean()

plt.figure(figsize=(8, 5))
sns.heatmap(group_trans, annot=True, cmap="YlOrRd", fmt=".2f")
plt.title("各用户群体行为转移均值热力图")
plt.tight_layout()
plt.savefig(SAVE_HEAT_PATH, dpi=300, bbox_inches="tight")
plt.close()
print("热力图已成功保存，全部流程执行完毕")
