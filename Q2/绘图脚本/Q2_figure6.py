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


def plot_spearman_violin_darker(csv_path):
    # 1. 路径检查
    if not os.path.exists(csv_path):
        print(f"❌ 找不到文件: {csv_path}")
        return

    print(f"📖 正在读取数据: {csv_path}")
    df = pd.read_csv(csv_path)

    # 2. 数据清洗与变形 (Melt)
    target_cols = ['rho_pct', 'rho_rank']
    label_map = {
        'rho_pct': 'Percent Mechanism',
        'rho_rank': 'Rank Mechanism'
    }
    df_melted = df.melt(
        value_vars=target_cols,
        var_name='Mechanism_Key',
        value_name='Spearman_Rho'
    )
    df_melted['Mechanism'] = df_melted['Mechanism_Key'].map(label_map)

    # 3. 计算均值
    means = df[target_cols].mean()
    print("📊 Global Means:")
    print(means)

    # 4. 开始绘图
    plt.figure(figsize=(10, 8))

    # ==========================================
    # 🎨 核心修改：采用更深、对比度更高的配色
    # ==========================================
    # Percent (深蓝) / Rank (深橙/赭色)
    palette = {"Percent Mechanism": "#2c5aa0", "Rank Mechanism": "#c4622d"}
    # 散点颜色 (深灰)
    strip_color = '#4a4a4a'
    # 均值标记颜色 (鲜艳深红)
    mean_color = '#a50f15'

    # --- 图层 A: 小提琴图 ---
    ax = sns.violinplot(
        data=df_melted,
        x='Mechanism',
        y='Spearman_Rho',
        hue='Mechanism',
        palette=palette,
        inner=None,
        linewidth=1.5,
        legend=False
    )

    # 手动调整 alpha (让颜色更实一点，0.2 -> 0.4)
    for poly in ax.collections:
        poly.set_alpha(0.4)

    # --- 图层 B: 抖动散点图 ---
    sns.stripplot(
        data=df_melted,
        x='Mechanism',
        y='Spearman_Rho',
        color=strip_color,  # 使用深灰色
        alpha=0.5,  # 稍微不透明一点
        size=3,  # 点稍微大一点
        jitter=0.25,
        zorder=1
    )

    # --- 图层 C: 均值标记 ---
    x_coords = [0, 1]
    mean_values = [means['rho_pct'], means['rho_rank']]

    plt.scatter(
        x_coords,
        mean_values,
        color=mean_color,
        s=120,  # 更大一点
        marker='D',
        zorder=5,
        label='Global Mean',
        edgecolor='white',
        linewidth=1.5
    )

    # 添加数值标签
    for x, y in zip(x_coords, mean_values):
        plt.text(
            x + 0.08, y,
            f'{y:.3f}',
            color=mean_color,
            fontweight='bold',
            fontsize=13,
            va='center',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)  # 加个白底背景
        )

    # 5. 图表修饰
    plt.title('Distribution of Spearman Correlation ($\\rho$) with Fan Intent',
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Spearman Correlation ($\\rho$)\n(Higher is Better)', fontsize=14)
    plt.xlabel('Scoring Mechanism', fontsize=14)
    plt.ylim(0.35, 1.05)

    # 参考线 (深绿色)
    plt.axhline(0.9, color='#006400', linestyle='--', alpha=0.6, linewidth=1.5)
    plt.text(1.55, 0.9, 'High Consistency', color='#006400', fontsize=10, va='center', fontweight='bold')

    plt.legend(loc='lower right', frameon=True, fontsize=11)
    plt.tight_layout()

    # 保存
    save_path = 'Spearman_Distribution_Violin_Darker.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存至: {save_path}")
    plt.show()


if __name__ == "__main__":
    import os

    # 自动定位路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))

    csv_path = os.path.join(project_root, "data", "Q2data", "Q2_simulation_metrics.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(project_root, "data", "Q2data", "Q2_simulation_metrics(1).csv")

    plot_spearman_violin_darker(csv_path)