import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 【全局样式设置：美赛 LaTeX 风格】
# ==========================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (8, 5),
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'mathtext.fontset': 'stix',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': ':',
    'axes.spines.top': False
})


def get_plot_data():
    """获取或生成演示数据"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, "sec7_outputs", "sec7_q1_prior_grid_summary.csv")
        if not os.path.exists(csv_path):
            csv_path = "sec7_q1_prior_grid_summary.csv"
    except NameError:
        csv_path = "sec7_q1_prior_grid_summary.csv"

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Data file not found, generating dummy data...")
        np.random.seed(42)
        betas = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])

        # 模拟数据逻辑：
        # vote_mean 随 beta 变化轻微波动
        vote_mean = 0.15 + 0.01 * np.sin(np.log(betas))
        # ci_width 随 beta 增大而减小（先验越强越确定）
        ci_width = 0.02 * np.sqrt(1 / betas)
        # L1 shift 随 beta 偏离 1.0 而增大
        l1_shift = 0.05 * np.abs(np.log(betas) - np.log(1.0))
        # accept_rate 随 beta 偏离最优值下降
        acc_rate = 0.3 * np.exp(-0.1 * np.abs(betas - 1.0))

        df = pd.DataFrame({
            'beta': betas,
            'target_vote_mean': vote_mean,
            'target_ci_low': vote_mean - ci_width,
            'target_ci_high': vote_mean + ci_width,
            'L1_mean_to_baseline': l1_shift,
            'accept_rate': acc_rate
        })

    return df.sort_values('beta')


def plot_prior_sensitivity(df, colors, filename):
    """
    通用绘图函数
    colors: 字典，包含 'left' (左轴/误差棒), 'right' (右轴/折线), 'text' (文字)
    """
    fig, ax1 = plt.subplots()

    # --- 左轴：Posterior Mean (误差棒) ---
    yerr_low = df['target_vote_mean'] - df['target_ci_low']
    yerr_high = df['target_ci_high'] - df['target_vote_mean']
    x_pos = np.arange(len(df))

    # 绘制误差棒
    # capsize=4 设置误差棒两端的横线宽度
    # markerfacecolor='white' 让点看起来更通透（仅对某些风格）
    mfc = 'white' if colors['left'] != 'black' else colors['left']

    eb = ax1.errorbar(x_pos, df['target_vote_mean'], yerr=[yerr_low, yerr_high],
                      fmt='-o', color=colors['left'], ecolor=colors['left'],
                      capsize=4, label='Posterior mean (95% CI)', zorder=10,
                      markerfacecolor=mfc, markeredgewidth=1.5)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(df['beta'].astype(str))
    ax1.set_xlabel(r"Prior strength $\beta$", color=colors['text'])
    ax1.set_ylabel("Posterior mean (95% CI)", color=colors['left'], fontweight='bold')

    # 左轴颜色定制
    ax1.tick_params(axis='y', labelcolor=colors['left'], colors=colors['left'])
    ax1.tick_params(axis='x', labelcolor=colors['text'], colors=colors['text'])
    ax1.spines['left'].set_color(colors['left'])
    ax1.spines['left'].set_linewidth(1.2)
    ax1.spines['bottom'].set_color(colors['text'])

    # --- 右轴：L1 Shift (折线) ---
    ax2 = ax1.twinx()
    ln2 = ax2.plot(x_pos, df['L1_mean_to_baseline'],
                   color=colors['right'], marker='^', linestyle='--',
                   label='L1 shift to baseline', markersize=6,
                   markerfacecolor='white', markeredgewidth=1.5)

    ax2.set_ylabel("L1 shift to baseline", color=colors['right'], fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=colors['right'], colors=colors['right'])

    # 右轴颜色定制
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color(colors['right'])
    ax2.spines['right'].set_linewidth(1.2)
    ax2.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # 扩大右轴范围避免重叠
    ymin, ymax = ax2.get_ylim()
    ax2.set_ylim(0, ymax * 1.5)

    # --- 标注 Accept Rate ---
    n_points = len(df)
    indices_to_label = range(n_points) if n_points <= 6 else [0, n_points // 2, n_points - 1]

    for i in indices_to_label:
        acc = df.iloc[i]['accept_rate']
        y_pt = df.iloc[i]['target_ci_high']
        # 标注颜色使用 text 颜色
        ax1.text(x_pos[i], y_pt + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.02,
                 f"acc={acc:.2f}", ha='center', va='bottom', fontsize=9,
                 color=colors['text'], alpha=0.9)

    # --- 图例 ---
    lines = [eb[0]] + ln2
    labels = [l.get_label() for l in lines]
    legend = ax1.legend(lines, labels, loc='upper left', frameon=False)
    plt.setp(legend.get_texts(), color=colors['text'])

    # --- 标题与保存 ---
    plt.title("Prior Sensitivity Analysis (Bayesian Defense)", pad=15, color=colors['text'], fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"Generated: {filename}")
    plt.close()


if __name__ == "__main__":
    df = get_plot_data()

    # ============================
    # 1. 黑白配色 (Black & White)
    # ============================
    colors_bw = {
        'left': 'black',
        'right': '#666666',  # 深灰
        'text': 'black'
    }
    plot_prior_sensitivity(df, colors_bw, 'fig7_prior_sensitivity_bw.png')

    # ============================
    # 2. 蓝白/蓝灰配色 (Airy Blue)
    # ============================
    colors_blue_gray = {
        'left': '#2980B9',  # 清透蓝 (Posterior)
        'right': '#7F8C8D',  # 高级灰 (L1 Shift)
        'text': '#2C3E50'  # 深蓝灰
    }
    plot_prior_sensitivity(df, colors_blue_gray, 'fig7_prior_sensitivity_blue_gray.png')

    # ============================
    # 3. 蓝橙配色 (Blue & Orange)
    # ============================
    colors_blue_orange = {
        'left': '#0B3D91',  # 深宝石蓝 (Posterior)
        'right': '#D35400',  # 活力橙 (L1 Shift)
        'text': '#333333'  # 深炭灰
    }
    plot_prior_sensitivity(df, colors_blue_orange, 'fig7_prior_sensitivity_blue_orange.png')

    print("\nAll 3 sensitivity figures generated successfully.")