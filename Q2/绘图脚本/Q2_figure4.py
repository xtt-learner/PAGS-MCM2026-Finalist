import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# ==========================================
# 0. Style Configuration (Academic)
# ==========================================
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12


def get_target_era(season):
    """Filter only the two target eras for this specific chart."""
    if season <= 2:
        return "Rank Era (S1-2)"
    elif 28 <= season <= 34:
        return "Save Era (S28-34)"
    else:
        return None  # Ignore other seasons


def plot_ims_impact_v2(csv_path):
    # 1. Path Check
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return

    print(f"📖 Loading data: {csv_path}")
    df = pd.read_csv(csv_path)

    # 2. Data Processing
    df['Target_Era'] = df['season'].apply(get_target_era)
    plot_df = df.dropna(subset=['Target_Era']).copy()
    era_order = ["Rank Era (S1-2)", "Save Era (S28-34)"]
    plot_df['Target_Era'] = pd.Categorical(plot_df['Target_Era'], categories=era_order, ordered=True)

    # 3. Calculate Means
    metrics = ['FAI_rank', 'FAI_rank_ims']
    era_means = plot_df.groupby('Target_Era', observed=True)[metrics].mean().reset_index()

    # Calculate Global Baseline
    global_fai_rank_mean = df['FAI_rank'].mean()
    print(f"🌍 Global Historical FAI_rank Mean: {global_fai_rank_mean:.4f}")

    # 4. Melt for Seaborn
    df_melted = era_means.melt(
        id_vars='Target_Era',
        value_vars=metrics,
        var_name='Condition',
        value_name='FAI_Value'
    )
    label_map = {
        'FAI_rank': 'Original Rank Bias (Before)',
        'FAI_rank_ims': 'With Judge Save (After)'
    }
    df_melted['Condition_Label'] = df_melted['Condition'].map(label_map)

    # 5. Plotting Setup (Increased width for external legend)
    fig, ax = plt.subplots(figsize=(12, 7))  # 增加宽度

    colors = ["#69b3a2", "#e9967a"]

    # Draw Bars
    sns.barplot(
        data=df_melted,
        x='Target_Era',
        y='FAI_Value',
        hue='Condition_Label',
        palette=colors,
        edgecolor='white',
        linewidth=1.5,
        alpha=1.0,
        ax=ax
    )

    # 6. Add & Annotate Global Reference Line
    line_color = '#69b3a2'
    ax.axhline(
        y=global_fai_rank_mean,
        color=line_color,
        linestyle='--',
        alpha=0.7,
        linewidth=1.5,
        # label 是给图例用的，这里保留
        label=f'Historical Avg Bias'
    )

    # 🔥 核心修改：直接在虚线旁标注数值
    # 获取当前 X 轴的范围，把文字放在最右侧外面
    x_start, x_end = ax.get_xlim()
    ax.text(
        x=x_end + 0.05,  # 稍微向右偏移
        y=global_fai_rank_mean,
        s=f'{global_fai_rank_mean:.3f}',  # 显示三位小数
        color=line_color,
        va='center',
        ha='left',
        fontweight='bold',
        fontsize=11,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2)  # 加个白底背景，防止看不清
    )

    # 7. Annotations & Styling
    ax.set_title('Impact of Idealized Meritocratic Save (IMS) on Fairness\n(Before vs. After)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Era', fontsize=14)
    ax.set_ylabel('Fan Alignment Index (FAI)\n(>0: Pro-Fan, <0: Pro-Judge)', fontsize=14)
    ax.axhline(0, color='black', linewidth=1)  # Zero line

    # Add values on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=11, fontweight='bold')

    # Add correction arrow
    s28_row = era_means[era_means['Target_Era'] == "Save Era (S28-34)"]
    if not s28_row.empty:
        val_before = s28_row['FAI_rank'].values[0]
        val_after = s28_row['FAI_rank_ims'].values[0]
        drop = val_before - val_after

        ax.annotate(
            f'Correction Effect:\nΔ = -{drop:.3f}',
            xy=(1, (val_before + val_after) / 2),
            xytext=(1.35, val_before),
            arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle="arc3,rad=.2"),
            fontsize=11, color='darkred', ha='center'
        )

    # 🔥 核心修改：将图例移到外部右侧
    # bbox_to_anchor=(1.05, 1) 表示图例左上角定位在图表(1.05, 1)的位置，即右上角外侧
    ax.legend(title="Metric Condition", loc='upper left', bbox_to_anchor=(1.05, 1), frameon=True, fontsize=11)

    plt.tight_layout()  # 自动调整布局以适应外部图例

    # Save

    save_path = os.path.join(project_root, "images", "Q2_images", 'IMS_Impact_Before_After.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # bbox_inches='tight' 确保保存时不裁剪外部元素
    print(f"✅ Chart saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    import os

    # Auto-locate path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))

    csv_path = os.path.join(project_root, "data", "Q2data", "Q2_season_summary(1).csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(project_root, "data", "Q2data", "Q2_season_summary.csv")

    plot_ims_impact_v2(csv_path)