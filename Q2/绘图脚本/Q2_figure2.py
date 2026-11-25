import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# ==========================================
# 1. 绘图风格设置 (学术论文风格)
# ==========================================
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12


def get_era(season):
    """
    根据赛季划分历史阶段
    """
    if season <= 2:
        return "Rank Era\n(S1-2)"
    elif season <= 27:
        return "Percent Era\n(S3-27)"
    else:
        return "Judges' Save Era\n(S28-34)"


def plot_era_comparison(csv_path):
    # 1. 路径检查
    if not os.path.exists(csv_path):
        print(f"❌ 找不到文件: {csv_path}")
        return

    print(f"📖 正在读取数据: {csv_path}")
    df = pd.read_csv(csv_path)

    # 2. 数据处理：划分 Era
    df['Era'] = df['season'].apply(get_era)

    # 定义 Era 的顺序
    era_order = ["Rank Era\n(S1-2)", "Percent Era\n(S3-27)", "Judges' Save Era\n(S28-34)"]
    df['Era'] = pd.Categorical(df['Era'], categories=era_order, ordered=True)

    # 3. 计算均值数据 (核心修改部分)
    metrics = ['FAI_pct', 'FAI_rank', 'rho_pct', 'rho_rank']

    # [A] 计算各 Era 的均值 (用于画柱子)
    era_means = df.groupby('Era', observed=True)[metrics].mean().reset_index()

    # [B] 计算全局均值 (用于画参考线)
    global_means = df[metrics].mean()

    print("\n📊 各阶段均值数据 (Era Means):")
    print(era_means)
    print("\n🌍 全局历史均值 (Global Means):")
    print(global_means)

    # 4. 数据变形 (Melt)
    df_melted = era_means.melt(
        id_vars='Era',
        value_vars=metrics,
        var_name='Metric',
        value_name='Value'
    )

    # 5. 优化图例文本
    label_map = {
        'FAI_pct': r'$FAI_{pct}$ (Percent Bias)',
        'FAI_rank': r'$FAI_{rank}$ (Rank Bias)',
        'rho_pct': r'$\rho_{pct}$ (Consistency)',
        'rho_rank': r'$\rho_{rank}$ (Consistency)'
    }
    df_melted['Metric Label'] = df_melted['Metric'].map(label_map)

    # 6. 开始绘图
    plt.figure(figsize=(14, 8))  # 稍微调大一点画布

    # 定义颜色 (与 metrics 的顺序一一对应)
    # 顺序: FAI_pct, FAI_rank, rho_pct, rho_rank
    custom_palette = ["#ff9999", "#cc0000", "#99ccff", "#003399"]

    # 创建颜色映射字典，方便后续画线时调用
    color_map = dict(zip(metrics, custom_palette))

    # [绘制柱状图]
    ax = sns.barplot(
        data=df_melted,
        x='Era',
        y='Value',
        hue='Metric Label',
        palette=custom_palette,
        edgecolor='black',
        linewidth=1,
        alpha=0.9
    )

    # =======================================================
    # 🔥 新增功能：绘制全局均值参考线
    # =======================================================
    # 为了防止文字重叠，我们可以把 FAI (低值) 和 rho (高值) 的标签放在不同侧，或者简单地画线

    # 获取当前 X 轴的范围，用于定位文字
    x_start, x_end = ax.get_xlim()

    for metric in metrics:
        mean_val = global_means[metric]
        color = color_map[metric]

        # 1. 画虚线
        plt.axhline(y=mean_val, color=color, linestyle='--', linewidth=1.5, alpha=0.8, zorder=0)

        # 2. 添加右侧文字标注 (显示数值)
        # 如果是 rho (值很大 > 0.8)，文字放在左边；如果是 FAI (值很小 < 0.4)，文字放在右边
        # 这样可以错开一点
        text_x = x_end + 0.02 if mean_val < 0.5 else x_start - 0.02
        ha_align = 'left' if mean_val < 0.5 else 'right'

        plt.text(
            x=x_end,  # 统一放在最右侧外边
            y=mean_val,
            s=f'{mean_val:.3f}',
            color=color,
            va='center',
            ha='left',
            fontsize=10,
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1)
        )

    # =======================================================

    # 7. 图表细节修饰
    plt.title(
        'Evolution of Fairness (FAI) and Consistency (Spearman $\\rho$) Across Eras\n(Dashed Lines Represent Global Historical Means)',
        fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Historical Era', fontsize=14, labelpad=10)
    plt.ylabel('Index Value', fontsize=14)
    plt.ylim(0, 1.15)  # 留出更多空间

    # 添加数值标签 (柱子上)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=10)

    # 优化图例
    plt.legend(
        title="Metric Definition",
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=True,
        fontsize=11
    )

    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(project_root, "images", "Q2_images", "Era_Comparison_With_Means.png")

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存至: {save_path}")
    plt.show()


if __name__ == "__main__":
    import os

    # 路径设置保持不变
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))

    # 尝试寻找文件
    csv_file_path = os.path.join(project_root, "data", "Q2data", "Q2_season_summary.csv")
    if not os.path.exists(csv_file_path):
        csv_file_path = os.path.join(project_root, "data", "Q2data", "Q2_season_summary(1).csv")

    plot_era_comparison(csv_file_path)