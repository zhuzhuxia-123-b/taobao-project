import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde
import warnings
import os
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
sns.set(font='DejaVu Sans')
plt.rcParams["axes.unicode_minus"] = False

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.facecolor'] = 'white'
sns.set_style("whitegrid")
sns.set_palette("husl")

BASE_PATH = r"D:\社会计算\taobao-project\data\processed"
DATA_CSV = f"{BASE_PATH}\\chained.csv"
GROUP_CSV = f"{BASE_PATH}\\user_groups.csv"

IMG_DIR = f"{BASE_PATH}\\visualize_a"
os.makedirs(IMG_DIR, exist_ok=True)

FIG_FUNNEL = f"{IMG_DIR}\\funnel.png"
FIG_CONVERSION = f"{IMG_DIR}\\conversion.png"
FIG_TIME_WINDOW = f"{IMG_DIR}\\time_window.png"
FIG_HEATMAP = f"{IMG_DIR}\\heatmap.png"

STATE_ORDER = ['latent', 'exploring', 'intent', 'closed']
BEHAVIOR_LABELS = ['pv', 'cart', 'buy', 'fav']
GROUP_NAMES = ['Low-active User', 'Browse-oriented User', 'Cart-oriented User', 'Purchase-oriented User']
CODE_CART = 2
CODE_BUY = 3

def load_main_data():
    """Load main behavior dataset"""
    try:
        df = pd.read_csv(DATA_CSV)
        df["time"] = pd.to_datetime(df["timestamp"], unit="s")
        print(f"✅ Main data loaded, total {len(df):,} rows")
        return df
    except FileNotFoundError:
        print(f"❌ Error: File not found {DATA_CSV}")
        exit()

def load_group_data():
    """Load user clustering result"""
    try:
        group_df = pd.read_csv(GROUP_CSV)
        print("✅ User group data loaded")
        return group_df
    except FileNotFoundError:
        print(f"❌ Error: File not found {GROUP_CSV}")
        exit()

# ===================== 1. Demand Funnel Chart =====================
def plot_funnel(df):
    print("\n⏳ Plotting Demand Funnel Chart...")
    state_count = df.groupby("user_id")["need_state"].last().value_counts()
    state_count = state_count.reindex(STATE_ORDER, fill_value=0)
    widths = state_count.values
    max_width = max(widths) if max(widths) > 0 else 1

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ['#E8F4FD', '#B8D4E8', '#6BA3D6', '#2E86AB']
    y_positions = [3, 2, 1, 0]

    for i, (state, count) in enumerate(state_count.items()):
        width = count / max_width * 0.8
        left = (1 - width) / 2
        rect = plt.Rectangle((left, y_positions[i] - 0.35), width, 0.7,
                            facecolor=colors[i], edgecolor='#2c3e50', linewidth=1.8)
        ax.add_patch(rect)
        ax.text(0.5, y_positions[i], f'{state}\nUser Count: {count:,}',
                ha='center', va='center', fontsize=14, fontweight='bold', color='#2c3e50')

    for i in range(len(STATE_ORDER) - 1):
        curr_w = widths[i] / max_width * 0.8
        next_w = widths[i+1] / max_width * 0.8
        ax.annotate('', xy=(0.5 - next_w/2, y_positions[i+1] + 0.35),
                    xytext=(0.5 - curr_w/2, y_positions[i] - 0.35),
                    arrowprops=dict(arrowstyle='-', color='#666', lw=1.2))
        ax.annotate('', xy=(0.5 + next_w/2, y_positions[i+1] + 0.35),
                    xytext=(0.5 + curr_w/2, y_positions[i] - 0.35),
                    arrowprops=dict(arrowstyle='-', color='#666', lw=1.2))
        if widths[i] > 0:
            conversion = widths[i+1] / widths[i] * 100
            ax.text(0.93, y_positions[i] - 0.15, f'Conversion: {conversion:.1f}%',
                    fontsize=11, color='#555', ha='right')

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 3.5)
    ax.set_title('User Demand State Funnel', fontsize=18, fontweight='bold', pad=20, color='#2c3e50')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(FIG_FUNNEL, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Funnel chart saved")

# ===================== 2. Top20 Category Cart-to-Buy Conversion Rate =====================
def plot_category_conversion(df):
    print("\n⏳ Plotting Top20 Category Conversion Chart...")
    cart_series = df[df["behavior_code"] == CODE_CART].groupby("category_id").size()
    buy_series = df[df["behavior_code"] == CODE_BUY].groupby("category_id").size()

    merge_df = pd.DataFrame({"cart": cart_series, "buy": buy_series}).fillna(0)
    merge_df = merge_df[merge_df["cart"] > 0]
    if len(merge_df) == 0:
        print("⚠️  No valid cart data, skip this chart")
        return

    merge_df["conversion"] = merge_df["buy"] / merge_df["cart"]
    conv_top20 = merge_df["conversion"].sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = sns.color_palette("RdYlGn", len(conv_top20))
    bars = ax.bar(range(len(conv_top20)), conv_top20.values,
                  color=colors, edgecolor='#2c3e50', linewidth=1)

    for idx, (bar, val) in enumerate(zip(bars, conv_top20.values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f'{val:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(range(len(conv_top20)))
    ax.set_xticklabels([f'Category {i+1}' for i in range(len(conv_top20))],
                       rotation=45, ha='right', fontsize=10)
    ax.set_xlabel('Category ID', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Cart-to-Buy Conversion Rate', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Top20 Categories: Cart-to-Buy Conversion Comparison', fontsize=16, fontweight='bold', pad=15)
    ax.set_ylim(0, conv_top20.max() * 1.18)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(FIG_CONVERSION, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Conversion rate chart saved")

# ===================== 3. Cart to Purchase Time Interval Distribution =====================
def plot_time_window(df):
    print("\n⏳ Plotting Cart-to-Buy Time Interval Chart...")
    cart_df = df[df["behavior_code"] == CODE_CART][["user_id", "time"]].rename(columns={"time": "t_cart"})
    buy_df = df[df["behavior_code"] == CODE_BUY][["user_id", "time"]].rename(columns={"time": "t_buy"})
    merge_df = pd.merge(cart_df, buy_df, on="user_id", how="inner")

    if len(merge_df) == 0:
        print("⚠️  No users with both cart & purchase behavior, skip this chart")
        return

    merge_df["delta_h"] = (merge_df["t_buy"] - merge_df["t_cart"]).dt.total_seconds() / 3600
    delta_data = merge_df["delta_h"].clip(lower=0, upper=72)

    fig, ax = plt.subplots(figsize=(12, 6))
    kde = gaussian_kde(delta_data)
    x = np.linspace(0, 72, 300)
    y = kde(x)

    ax.fill_between(x, y, alpha=0.45, color='#FF6B6B')
    ax.plot(x, y, color='#FF6B6B', linewidth=2.8, label='Probability Density Curve')

    peak_idx = np.argmax(y)
    ax.annotate(f'Density Peak: {x[peak_idx]:.1f} Hours',
                xy=(x[peak_idx], y[peak_idx]),
                xytext=(x[peak_idx] + 6, y[peak_idx] + 0.012),
                fontsize=11, fontweight='bold', color='#2c3e50',
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))

    ax.axvline(x=24, color='#4ECDC4', linestyle='--', linewidth=2, label='24 Hours')
    ax.axvline(x=48, color='#45B7D1', linestyle='--', linewidth=2, label='48 Hours')

    ax.set_xlabel('Time Interval (Hours)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Cart-to-Purchase Time Window Distribution', fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(FIG_TIME_WINDOW, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Time window chart saved")

# ===================== 4. Behavior Transition Heatmap for Four User Groups =====================
def plot_behavior_heatmap(df, group_df):
    print("\n⏳ Plotting User Behavior Transition Heatmap...")
    df_merge = df.merge(group_df, on="user_idx", how="inner")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for group_id in range(4):
        group_data = df_merge[df_merge["group"] == group_id]
        user_count = group_data["user_idx"].nunique()
        trans_matrix = np.zeros((4, 4))

        for uid in group_data["user_idx"].unique():
            user_seq = group_data[group_data["user_idx"] == uid]
            user_seq = user_seq.sort_values("timestamp")
            code_list = user_seq["behavior_code"].tolist()
            for curr, nxt in zip(code_list, code_list[1:]):
                if 1 <= curr <= 4 and 1 <= nxt <= 4:
                    trans_matrix[curr - 1][nxt - 1] += 1

        row_sum = trans_matrix.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        prob_matrix = trans_matrix / row_sum

        sns.heatmap(prob_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                    xticklabels=BEHAVIOR_LABELS, yticklabels=BEHAVIOR_LABELS,
                    ax=axes[group_id], vmin=0, vmax=1,
                    annot_kws={'fontsize': 12, 'fontweight': 'bold', 'color': '#222'},
                    cbar_kws={'label': 'Transition Probability', 'shrink': 0.8})

        axes[group_id].set_title(f'{GROUP_NAMES[group_id]}\n(Unique Users: {user_count:,})',
                                 fontsize=14, fontweight='bold', pad=12)
        axes[group_id].set_xlabel('Next Behavior', fontsize=11, fontweight='bold')
        axes[group_id].set_ylabel('Current Behavior', fontsize=11, fontweight='bold')

    plt.suptitle('Behavior Transition Probability Matrix of Four User Groups', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_HEATMAP, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Transition heatmap saved")

# ===================== Main Entry =====================
if __name__ == "__main__":
    print("=" * 60)
    print("📊 Start generating analysis charts")
    print("=" * 60)

    main_df = load_main_data()
    group_df = load_group_data()

    plot_funnel(main_df)
    plot_category_conversion(main_df)
    plot_time_window(main_df)
    plot_behavior_heatmap(main_df, group_df)

    print("\n" + "=" * 60)
    print("🎉 All charts generated successfully!")
    print("=" * 60)
    print("📁 Output file list:")
    print(f"  1. Demand Funnel Chart     → {FIG_FUNNEL}")
    print(f"  2. Category Conversion Chart   → {FIG_CONVERSION}")
    print(f"  3. Time Interval Distribution → {FIG_TIME_WINDOW}")
    print(f"  4. Behavior Transition Heatmap → {FIG_HEATMAP}")
    print("=" * 60)