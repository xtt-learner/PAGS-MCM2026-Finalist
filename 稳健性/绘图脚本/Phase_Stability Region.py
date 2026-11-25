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
    'lines.linewidth': 1.8,
    'lines.markersize': 6,
    'mathtext.fontset': 'stix',
    'axes.grid': True,
    'grid.alpha': 0.4,
    'grid.linestyle': ':',
    'axes.spines.top': False,
    'axes.spines.right': False
})


def get_plot_data():
    # 1. 路径与数据读取 (含容错)
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, "sec7_outputs", "sec7_q4_gamma_empirical_raw.csv")
        if not os.path.exists(csv_path):
            csv_path = "sec7_q4_gamma_empirical_raw.csv"
    except NameError:
        csv_path = "sec7_q4_gamma_empirical_raw.csv"

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Data file not found, generating dummy data...")
        np.random.seed(42)
        n = 1000
        gamma = np.random.uniform(0.5, 1.1, n)
        prob_activation = 1 / (1 + np.exp(20 * (gamma - 0.95)))
        guardrail_on = np.random.rand(n) < prob_activation
        D = np.random.normal(0.05, 0.01, n) + 0.02 * (1 - gamma) * guardrail_on
        df = pd.DataFrame({'gamma': gamma, 'guardrail_on': guardrail_on, 'D': D})

    # 2. 数据分箱处理
    df['gamma_bin'] = pd.qcut(df['gamma'], q=8)
    grouped = df.groupby('gamma_bin', observed=False)
    plot_data = pd.DataFrame({
        'gamma_center': grouped['gamma'].median(),
        'activation_rate': grouped['guardrail_on'].mean(),
        'D_mean': grouped['D'].mean()
    }).reset_index(drop=True)

    return plot_data.sort_values('gamma_center')


def plot_phase_diagram(data, colors, filename):
    """
    通用绘图函数
    colors: 字典，包含 'act' (左轴), 'd' (右轴), 'text' (文字), 'line' (辅助线)
    """
    fig, ax1 = plt.subplots()

    # --- 左轴：激活率 ---
    ln1 = ax1.plot(data['gamma_center'], data['activation_rate'],
                   color=colors['act'], marker='o', linestyle='-',
                   label='Guardrail activation rate', zorder=10,
                   markerfacecolor='white', markeredgewidth=1.5)

    ax1.set_xlabel(r"Gamma ($\gamma$)", color=colors['text'])
    ax1.set_ylabel("Activation rate", color=colors['act'], fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=colors['act'], colors=colors['act'])
    ax1.tick_params(axis='x', colors=colors['text'])
    ax1.set_ylim(-0.05, 1.05)

    ax1.spines['left'].set_color(colors['act'])
    ax1.spines['left'].set_linewidth(1.2)
    ax1.spines['bottom'].set_color(colors['text'])

    # --- 右轴：平均畸变 D ---
    ax2 = ax1.twinx()
    ln2 = ax2.plot(data['gamma_center'], data['D_mean'],
                   color=colors['d'], marker='s', linestyle='--',
                   label='Mean distortion $D$', markersize=5,
                   markerfacecolor='white', markeredgewidth=1.5)

    ax2.set_ylabel(r"Mean distortion $D$", color=colors['d'], fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=colors['d'], colors=colors['d'])

    # 右轴边框定制
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color(colors['d'])
    ax2.spines['right'].set_linewidth(1.2)
    ax2.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    d_max = data['D_mean'].max()
    ax2.set_ylim(0, max(0.15, d_max * 2.0))

    # --- 辅助线与标注 ---
    # 阈值线
    ax1.axvline(x=0.95, color=colors['text'], linestyle=':', linewidth=1.2, alpha=0.8)

    # 阈值文本 (靠近线条底部，避免遮挡曲线)
    ax1.text(0.955, 0.05, r"Onset near $\gamma \approx 0.95$",
             transform=ax1.get_xaxis_transform(), ha='left', va='bottom',
             fontsize=10, style='italic', color=colors['text'],
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))

    # 图例 (放在左中或左下空旷处)
    lines = ln1 + ln2
    labels = [l.get_label() for l in lines]
    legend = ax1.legend(lines, labels, loc='lower left', frameon=False)
    plt.setp(legend.get_texts(), color=colors['text'])


    plt.title("Phase Diagram: Guardrail Stability vs. Damping Factor", pad=15, color=colors['text'], fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"Generated: {filename}")
    plt.close()


if __name__ == "__main__":
    # 获取数据
    plot_data = get_plot_data()

    # ============================
    # 1. 黑白配色 (Black & White)
    # ============================
    colors_bw = {
        'act': 'black',
        'd': '#666666',  # 深灰
        'text': 'black',
        'line': 'black'
    }
    plot_phase_diagram(plot_data, colors_bw, 'fig7_gamma_phase_bw.png')

    # ============================
    # 2. 蓝白/蓝灰配色 (Airy Blue)
    # ============================
    colors_blue_gray = {
        'act': '#2980B9',  # 清透蓝
        'd': '#7F8C8D',  # 高级灰
        'text': '#2C3E50',  # 深蓝灰
        'line': '#34495E'
    }
    plot_phase_diagram(plot_data, colors_blue_gray, 'fig7_gamma_phase_blue_gray.png')

    # ============================
    # 3. 蓝橙配色 (Blue & Orange)
    # ============================
    colors_blue_orange = {
        'act': '#0B3D91',  # 深宝石蓝
        'd': '#D35400',  # 活力橙
        'text': '#333333',  # 深炭灰
        'line': '#333333'
    }
    plot_phase_diagram(plot_data, colors_blue_orange, 'fig7_gamma_phase_blue_orange.png')

    print("\nAll 3 figures generated successfully.")