import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import matplotlib.ticker as mtick

# ==========================================
# 1. 绘图风格设置 (学术论文风格)
# ==========================================
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12


def plot_pdiff_evolution(csv_path):
    # 1. 路径检查
    if not os.path.exists(csv_path):
        print(f"❌ 找不到文件: {csv_path}")
        return

    print(f"📖 正在读取数据: {csv_path}")
    df = pd.read_csv(csv_path)

    # 2. 数据预处理
    # 确保按赛季排序
    df = df.sort_values('season')

    # 将 Pdiff 转换为百分比 (0.262 -> 26.2)
    df['Pdiff_Percent'] = df['Pdiff'] * 100

    # 3. 计算全局均值 (用于画红虚线)
    global_mean_val = df['Pdiff_Percent'].mean()
    print(f"🌍 全局平均分歧概率: {global_mean_val:.2f}%")

    # 4. 开始绘图
    plt.figure(figsize=(14, 7))
    ax = plt.gca()

    # ==========================================
    # 🎨 核心步骤 A: 绘制背景区域 (Eras)
    # ==========================================
    # 定义三个阶段的区间和颜色
    # 为了美观，区间边界取 0.5 (例如 Rank Era 是 S1-S2，背景画到 2.5)
    eras = [
        {"name": "Rank Era", "start": 0.5, "end": 2.5, "color": "#fff7e6", "label": "Rank Era (S1-2)"},  # 浅橙色背景
        {"name": "Percent Era", "start": 2.5, "end": 27.5, "color": "#e6e6fa", "label": "Percent Era (S3-27)"},  # 浅紫色背景
        {"name": "Save Era", "start": 27.5, "end": 34.5, "color": "#e6ffe6", "label": "Save Era (S28-34)"}  # 浅绿色背景
    ]

    # 循环绘制背景矩形
    for era in eras:
        plt.axvspan(era["start"], era["end"], color=era["color"], alpha=0.6, label=era["label"])

    # ==========================================
    # 📈 核心步骤 B: 绘制主数据曲线
    # ==========================================
    sns.lineplot(
        data=df,
        x='season',
        y='Pdiff_Percent',
        marker='o',
        markersize=8,
        color='#000080',  # 深海军蓝
        linewidth=2,
        label=r'Disagreement Prob ($P_{diff}$)',
        zorder=10  # 确保线在背景之上
    )

    # ==========================================
    # 📏 核心步骤 C: 绘制全局均值虚线
    # ==========================================
    plt.axhline(
        y=global_mean_val,
        color='r',
        linestyle='--',
        linewidth=1.5,
        label=f'Overall Mean ({global_mean_val:.1f}%)',
        zorder=11
    )

    # 在虚线旁边添加文本说明
    plt.text(
        x=34.6,  # 放在图的最右侧外面
        y=global_mean_val,
        s=f'{global_mean_val:.1f}%',
        color='red',
        va='center',
        fontweight='bold',
        fontsize=10
    )

    # ==========================================
    # 🛠️ 核心步骤 D: 图表修饰
    # ==========================================
    plt.title('Evolution of Rule Disagreement Probability ($P_{diff}$) Across Seasons',
              fontsize=16, fontweight='bold', pad=15)

    plt.xlabel('Season', fontsize=14)
    plt.ylabel('Disagreement Probability (%)', fontsize=14)

    # 设置 X 轴刻度：显示所有整数赛季，太挤的话每隔2个显示
    plt.xticks(np.arange(1, 35, 1), fontsize=10)
    plt.xlim(0.5, 35.5)  # 留出一点边距

    # 设置 Y 轴刻度为百分比格式
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    # 图例设置 (包含 Eras 背景说明和线条说明)
    # 将图例放在图表外部或顶部，避免遮挡曲线
    plt.legend(
        loc='upper left',  # <--- 这里从 'upper right' 改为 'upper left'
        frameon=True,
        facecolor='white',
        edgecolor='gray',
        fontsize=10,
        framealpha=0.9  # 可选：增加一点不透明度，防止背景网格干扰文字
    )

    plt.tight_layout()

    # 保存图片

    save_path = os.path.join(project_root, "images", "Q2_images", "Pdiff_Evolution_Chart.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存至: {save_path}")
    plt.show()


if __name__ == "__main__":
    import os

    # 自动定位路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))

    # 优先寻找带 (1) 的文件，如果没有则找不带 (1) 的
    csv_path = os.path.join(project_root, "data", "Q2data", "Q2_season_summary(1).csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(project_root, "data", "Q2data", "Q2_season_summary.csv")

    plot_pdiff_evolution(csv_path)