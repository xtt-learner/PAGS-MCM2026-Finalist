import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 【路径配置区】(已替换为您提供的代码)
# ==========================================
try:
    # 尝试获取当前脚本路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # 如果在某些交互式环境中 __file__ 不可用，退回当前工作目录
    current_dir = os.getcwd()

project_root = os.path.dirname(os.path.dirname(current_dir))
# 注意：这里假设您的文件夹结构确实包含 sec7_outputs
PATH_convergence_CSV = os.path.join(current_dir, "sec7_outputs", "sec7_q1_mc_convergence.csv")

# ==========================================
# 【全局样式设置：清透蓝灰 (Airy Blue & Gray)】
# ==========================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],  # 论文标准字体
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (8, 5),
    'lines.linewidth': 1.8,  # 线条稍微加粗一点点，因为颜色较浅
    'lines.markersize': 6,
    'mathtext.fontset': 'stix',  # LaTeX 风格数学公式
    'axes.grid': True,
    'grid.alpha': 0.4,  # 网格线非常淡
    'grid.linestyle': ':',
    'axes.spines.top': False,  # 去掉顶部边框，更极简
    'axes.spines.right': False  # (注意：双轴图会自动处理右边框，这里先设全局)
})

# ==========================================
# 【配色定义】
# ==========================================
# 左轴 (L1 偏差): 使用清爽的中蓝色，既能看清又符合"浅蓝"基调
COLOR_L1 = '#2980B9'  # Medium Blue

# 右轴 (目标均值): 使用高级灰，作为参考基准
COLOR_TARGET = '#7F8C8D'  # Slate Gray

# 置信区间填充: 极淡的蓝色，像水彩一样
COLOR_CI_FILL = '#AED6F1'  # Very Light Blue (semi-transparent later)

# 文字颜色: 深蓝灰，比纯黑更柔和
COLOR_TEXT = '#2C3E50'  # Midnight Blueish Gray


def plot_convergence():
    # 1. 读取数据 (包含自动生成模拟数据的容错逻辑)
    try:
        df = pd.read_csv(PATH_convergence_CSV)
        print(f"Successfully loaded data from: {PATH_convergence_CSV}")
    except FileNotFoundError:
        print(f"File not found at {PATH_convergence_CSV}, generating dummy data for demonstration...")
        np.random.seed(42)
        M_vals = np.arange(100, 1550, 50)
        df = pd.DataFrame({
            'M': M_vals,
            'L1_mean_to_full': 0.2 * np.exp(-M_vals / 400) + 0.01 * np.random.rand(len(M_vals)),
            'target_mean': 0.5 + 0.1 * (1 - np.exp(-M_vals / 300)),
            'target_ci_width': 0.1 * np.exp(-M_vals / 800)
        })
        df['target_ci_low'] = df['target_mean'] - df['target_ci_width']
        df['target_ci_high'] = df['target_mean'] + df['target_ci_width']

    # 2. 创建画布
    fig, ax1 = plt.subplots()

    # --- 左轴绘制：L1 Deviation (蓝色) ---
    line1 = ax1.plot(df['M'], df['L1_mean_to_full'], color=COLOR_L1, marker='o',
                     linestyle='-', label=r'$L_1$ deviation to full posterior', zorder=10, markerfacecolor='white',
                     markeredgewidth=1.5)

    ax1.set_xlabel(r"Prefix samples $M'$", color=COLOR_TEXT)
    ax1.set_ylabel(r"$L_1$ deviation", color=COLOR_L1, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=COLOR_L1, colors=COLOR_L1)
    ax1.tick_params(axis='x', colors=COLOR_TEXT)

    # 轴线颜色定制
    ax1.spines['left'].set_color(COLOR_L1)
    ax1.spines['left'].set_linewidth(1.2)
    ax1.spines['bottom'].set_color(COLOR_TEXT)

    # --- 右轴绘制：Target Mean (灰色) ---
    ax2 = ax1.twinx()
    line2 = ax2.plot(df['M'], df['target_mean'], color=COLOR_TARGET, marker='s',
                     linestyle='--', label='Target posterior mean', markersize=5, markerfacecolor='white',
                     markeredgewidth=1.5)

    # 绘制置信区间 (淡蓝阴影)
    ax2.fill_between(df['M'], df['target_ci_low'], df['target_ci_high'],
                     color=COLOR_CI_FILL, alpha=0.3, label='95% CI')  # alpha=0.3 让它非常通透

    ax2.set_ylabel(r"Target posterior mean (95% CI)", color=COLOR_TARGET, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=COLOR_TARGET, colors=COLOR_TARGET)

    # 右轴边框定制
    ax2.spines['right'].set_visible(True)  # 显式开启右边框
    ax2.spines['right'].set_color(COLOR_TARGET)
    ax2.spines['right'].set_linewidth(1.2)
    ax2.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # 3. 合并图例 (极简风格，无边框)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    # 添加 CI 的图例块
    import matplotlib.patches as mpatches
    patch_ci = mpatches.Patch(color=COLOR_CI_FILL, alpha=0.3, label='95% CI')
    lines.append(patch_ci)
    labels.append('95% CI')

    legend = ax1.legend(lines, labels, loc='upper center', frameon=False, ncol=3)
    # 设置图例文本颜色
    plt.setp(legend.get_texts(), color=COLOR_TEXT)

    # 4. 标注 Plateau 点 (使用深灰)
    # 找到最接近 M=800 的点
    idx_800 = (df['M'] - 800).abs().idxmin()
    x_800 = df.loc[idx_800, 'M']
    y_800 = df.loc[idx_800, 'L1_mean_to_full']

    ax1.annotate(r"Plateau by $M' \approx 800$",
                 xy=(x_800, y_800),
                 xytext=(x_800 + 150, y_800 + (df['L1_mean_to_full'].max() - df['L1_mean_to_full'].min()) * 0.2),
                 arrowprops=dict(facecolor=COLOR_TARGET, arrowstyle='->', lw=1.0, color=COLOR_TARGET),
                 fontsize=11, ha='center', color=COLOR_TARGET, fontweight='normal',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLOR_TARGET, alpha=0.8))  # 加个白色底色防遮挡

    # 5. 标题与保存
    plt.title("Convergence Audit of Monte Carlo Approx.", pad=20, color=COLOR_TEXT, fontweight='bold')

    plt.tight_layout()
    # 保存为 PNG
    save_path = os.path.join(current_dir, 'fig7_mc_convergence_blue_gray.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Figure saved as {save_path}")


if __name__ == "__main__":
    plot_convergence()