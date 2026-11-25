import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# ==========================================
# 0. 学术风格设置
# ==========================================
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12


def plot_spearman_violin(csv_path):
    # 1. 路径检查
    if not os.path.exists(csv_path):
        print(f"❌ 找不到文件: {csv_path}")
        return

    print(f"📖 正在读取数据: {csv_path}")
    df = pd.read_csv(csv_path)

    # 2. 数据清洗与变形 (Melt)
    # 我们只关心两个核心变量：rho_pct 和 rho_rank
    # 将宽表转换为长表，方便 Seaborn 绘图
    target_cols = ['rho_pct', 'rho_rank']

    # 简单的列名映射，让图表更易读
    label_map = {
        'rho_pct': 'Percent Mechanism',
        'rho_rank': 'Rank Mechanism'
    }

    # Melt 操作
    df_melted = df.melt(
        value_vars=target_cols,
        var_name='Mechanism_Key',
        value_name='Spearman_Rho'
    )
    df_melted['Mechanism'] = df_melted['Mechanism_Key'].map(label_map)

    # 3. 计算均值 (用于画图上的红点/红线)
    means = df[target_cols].mean()
    print("📊 Global Means:")
    print(means)

    # 4. 开始绘图
    plt.figure(figsize=(10, 8))

    # 定义配色：Percent用蓝色(冷)，Rank用橙色(暖)，对比鲜明
    palette = {"Percent Mechanism": "#4c72b0", "Rank Mechanism": "#dd8452"}

    # --- 图层 A: 小提琴图 (展示密度分布) ---
    # inner=None 表示不在小提琴内部画自带的箱线图，保持干净
    ax = sns.violinplot(
        data=df_melted,
        x='Mechanism',
        y='Spearman_Rho',
        hue='Mechanism',
        palette=palette,
        inner=None,
        linewidth=1.5,
        alpha=0.3,  # 让颜色淡一点，不要遮住散点
        legend=False  # 不需要图例，x轴已经很清楚了
    )

    # 手动调整 alpha (Seaborn 的 violinplot alpha 参数有时不生效，需手动 hack)
    for poly in ax.collections:
        poly.set_alpha(0.2)

    # --- 图层 B: 抖动散点图 (展示真实数据点) ---
    sns.stripplot(
        data=df_melted,
        x='Mechanism',
        y='Spearman_Rho',
        color='grey',
        alpha=0.4,  # 半透明，避免太乱
        size=2.5,  # 点的大小
        jitter=0.25,  # 抖动宽度
        zorder=1  # 放在底层
    )

    # --- 图层 C: 均值标记 (核心对比) ---
    # 我们在每个小提琴的中心画一个显眼的红点或横线

    # 获取 x 轴的坐标位置 (通常是 0, 1)
    x_coords = [0, 1]
    mean_values = [means['rho_pct'], means['rho_rank']]

    # 绘制均值点
    plt.scatter(
        x_coords,
        mean_values,
        color='red',
        s=100,  # 大小
        marker='D',  # 菱形
        zorder=5,  # 最上层
        label='Global Mean',
        edgecolor='white'
    )

    # 添加数值标签
    for x, y in zip(x_coords, mean_values):
        plt.text(
            x + 0.05, y,
            f'{y:.3f}',
            color='#b30000',
            fontweight='bold',
            fontsize=12,
            va='center'
        )

    # 5. 图表修饰
    plt.title('Distribution of Spearman Correlation ($\\rho$) with Fan Intent',
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Spearman Correlation ($\\rho$)\n(Higher is Better)', fontsize=14)
    plt.xlabel('Scoring Mechanism', fontsize=14)

    # 设置 Y 轴范围 (根据你的描述 0.4 - 1.0)
    plt.ylim(0.35, 1.05)

    # 添加参考线 (例如 0.9 表示非常优秀)
    plt.axhline(0.9, color='green', linestyle=':', alpha=0.5)
    plt.text(1.55, 0.9, 'High Consistency', color='green', fontsize=10, va='center')

    # 添加图例 (仅针对均值点)
    plt.legend(loc='lower right', frameon=True)

    plt.tight_layout()

    # 保存

    save_path = os.path.join(project_root, "images", "Q2_images", 'Spearman_Distribution_Violin.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存至: {save_path}")
    plt.show()


if __name__ == "__main__":
    import os

    # 自动定位路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))

    # 优先找周级明细数据 (因为要画分布，必须用 metrics 文件，不能用 summary 文件)
    # 你的数据源是 Q2_simulation_metrics(1).csv 或 Q2_simulation_metrics.csv
    csv_path = os.path.join(project_root, "data", "Q2data", "Q2_simulation_metrics.csv")

    # 如果找不到，尝试带 (1) 的
    if not os.path.exists(csv_path):
        csv_path = os.path.join(project_root, "data", "Q2data", "Q2_simulation_metrics(1).csv")

    plot_spearman_violin(csv_path)