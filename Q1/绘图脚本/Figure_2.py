import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# 设置学术风格
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'


# ==========================================
# 1. 数据加载与中间指标计算 (Data Processing)
# ==========================================
def process_data(file_summary, file_diag):
    print("正在处理中间数据...")

    # 加载数据
    df_summary = pd.read_csv(file_summary)
    df_diag = pd.read_csv(file_diag)

    # --- A. 计算每周的标准化熵 (H_norm) ---
    # 归一化得票率
    df_summary['vote_mean_norm'] = df_summary.groupby(['season', 'week'])['vote_mean'].transform(lambda x: x / x.sum())

    # 计算 p * log(p)
    df_summary['entropy_comp'] = df_summary['vote_mean_norm'] * np.log(df_summary['vote_mean_norm'] + 1e-10)

    # 按周聚合
    df_entropy = df_summary.groupby(['season', 'week']).agg(
        abs_entropy=('entropy_comp', lambda x: -x.sum()),
        n_candidates=('contestant_id', 'count')
    ).reset_index()

    # 标准化
    df_entropy['h_norm'] = df_entropy['abs_entropy'] / np.log(df_entropy['n_candidates'])

    # 划分时代
    def get_regime(season):
        if season <= 2:
            return 'Rank Era\n(S1-2)'
        elif season <= 27:
            return 'Percent Era\n(S3-27)'
        else:
            return 'Save Era\n(S28-34)'

    df_entropy['Regime'] = df_entropy['season'].apply(get_regime)

    # --- B. 计算每季平均 PPC ---
    df_ppc_valid = df_diag[df_diag['n_elim'] > 0].copy()
    if 'ppc_exact' in df_ppc_valid.columns:
        df_ppc_valid['ppc_final'] = df_ppc_valid['ppc_exact'].fillna(df_ppc_valid.get('ppc_relaxed', 0))
    else:
        df_ppc_valid['ppc_final'] = df_ppc_valid['ppc_relaxed']

    df_season_ppc = df_ppc_valid.groupby('season')['ppc_final'].mean().reset_index()

    # --- C. 提取案例周数据 ---
    case_s2 = df_summary[(df_summary['season'] == 2) & (df_summary['week'] == 3)].copy()
    case_s27 = df_summary[(df_summary['season'] == 27) & (df_summary['week'] == 8)].copy()

    return df_entropy, df_season_ppc, case_s2, case_s27





# ==========================================
# 绘图函数: Figure 1 Pipeline (紧凑型 + 显性约束版)
# ==========================================
def draw_figure_1_compact():
    # 设置画布：更扁平，更紧凑
    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # --- 样式定义 ---
    # 统一字体和线宽
    lw_box = 1.5
    fs_main = 10
    fs_small = 9

    # 颜色定义
    c_input = "#f5f5f5"  # 浅灰
    c_prep = "#e1f5fe"  # 浅蓝
    c_algo = "#e8f5e9"  # 浅绿
    c_check = "#fff9c4"  # 浅黄 (决策点)
    c_const = "#ffebee"  # 浅红 (约束定义)
    c_border = "#333333"

    # --- 坐标系统 (紧凑排列) ---
    y_line = 3.0  # 主流水线高度

    # 定义模块宽度和间距
    w_box = 2.2
    h_box = 1.2
    gap = 0.8  # 模块间距缩小

    # 计算各模块中心 x 坐标
    x1 = 1.5  # Input
    x2 = x1 + w_box + gap  # Prep
    x3 = x2 + w_box + gap  # Prior
    x4 = x3 + w_box + gap  # Sampling (Engine)
    x5 = x4 + w_box + gap  # Check (Diamond) - 稍微拉开一点给回路留空间
    x6 = x5 + w_box + 1.0  # Output

    # 辅助函数：绘制紧凑的框
    def draw_node(x, y, text, color, style='round,pad=0.3', w=None, h=None):
        # 使用 bbox 绘制文本框
        ax.text(x, y, text, ha="center", va="center", fontsize=fs_main,
                bbox=dict(boxstyle=style, fc=color, ec=c_border, lw=lw_box))

    # 辅助函数：绘制连接线 (实实在在的连接)
    def connect(x_start, x_end, y, label=None):
        # 计算框的边缘 (近似)
        # 这里用 annotate 画短箭头
        ax.annotate("", xy=(x_end - w_box / 2 + 0.1, y), xytext=(x_start + w_box / 2 - 0.1, y),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color=c_border))

    # ==================== 1. 主流水线 (Main Pipeline) ====================

    # Node 1: Input
    draw_node(x1, y_line, "Input $(s,t)$\n$C_{s,t}, E_{s,t}, J_{s,t}$", c_input)
    connect(x1, x2, y_line)

    # Node 2: Prep
    draw_node(x2, y_line, "Normalize\n$p^J = J / \sum J$", c_prep)
    connect(x2, x3, y_line)

    # Node 3: Prior
    draw_node(x3, y_line, "Prior\n$v \sim Dir(\\alpha \cdot p^J)$", c_prep)
    connect(x3, x4, y_line)

    # Node 4: Sampling Engine (核心)
    # 使用加粗边框强调核心
    ax.text(x4, y_line, "Sampler\nDraw $v^{(m)} \sim Prior$", ha="center", va="center", fontsize=fs_main,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", fc=c_algo, ec="#2e7d32", lw=2.0))

    # 连接 Sampler 到 Check
    ax.annotate("", xy=(x5 - 0.7, y_line), xytext=(x4 + 1.0, y_line),
                arrowprops=dict(arrowstyle="->", lw=1.5, color=c_border))

    # Node 5: The Gate (Constraint Check) - 菱形
    # 菱形很难用 bbox 完美画出文字，这里用 patch 画菱形，再放文字
    diamond = patches.RegularPolygon((x5, y_line), numVertices=4, radius=0.9, orientation=0,
                                     fc=c_check, ec="#fbc02d", lw=1.5)
    ax.add_patch(diamond)
    ax.text(x5, y_line, "Check\nFeasibility?", ha="center", va="center", fontsize=fs_small, fontweight='bold')

    # ==================== 2. 约束模块 (The Red Module) ====================

    # 约束定义框 (放在 Check 下方)
    y_const = 1.0
    const_text = (
        "Constraint Definitions (The Rules)\n"
        "----------------------------------\n"
        "Rank: $E = Worst(R^J + R^v)$\n"
        "Percent: $E = Bottom(p^J + v)$\n"
        "Save: $E \subseteq BottomTwo(...)$"
    )
    ax.text(x5, y_const, const_text, ha="center", va="center", fontsize=9, family='monospace',
            bbox=dict(boxstyle="round,pad=0.4", fc=c_const, ec="#c62828", lw=1.5))

    # 红色粗箭头：定义 -> 检查点 (表示"依据这些规则检查")
    ax.annotate("", xy=(x5, y_line - 0.8), xytext=(x5, y_const + 0.8),
                arrowprops=dict(arrowstyle="simple,tail_width=0.5,head_width=1.5,head_length=1.0", color="#c62828"))
    ax.text(x5 + 0.2, (y_line + y_const) / 2, "Defines", ha="left", va="center", color="#c62828", fontsize=9,
            fontweight='bold')

    # ==================== 3. 循环与输出 (Loop & Output) ====================

    # A. 拒绝回路 (Reject Loop): Check -> Sampler
    # 从菱形顶部出来，回到 Sampler 顶部
    path_style = "arc3,rad=0.5"  # 弯曲路径
    ax.annotate("", xy=(x4, y_line + 0.7), xytext=(x5, y_line + 0.8),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#d32f2f", ls="--", connectionstyle=path_style))
    ax.text((x4 + x5) / 2, y_line + 1.3, "Reject & Retry", ha="center", va="bottom", fontsize=9, color="#d32f2f")

    # B. 接受路径 (Accept Path): Check -> Output
    ax.annotate("", xy=(x6 - 0.8, y_line), xytext=(x5 + 0.8, y_line),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#2e7d32"))
    ax.text((x5 + x6) / 2 - 0.4, y_line + 0.2, "Pass", ha="center", va="bottom", fontsize=9, color="#2e7d32",
            fontweight='bold')

    # Node 6: Output Cluster (右侧分叉)
    # 用一个大括号或者分开两个框
    y_out_1 = y_line + 0.6
    y_out_2 = y_line - 0.6

    # Output 1: Data
    draw_node(x6, y_out_1, "Posterior Samples\n$\{v^{(m)}_{s,t}\}$", "#f5f5f5")
    # Output 2: Stats
    draw_node(x6, y_out_2, "Diagnostics\n$PPC$ & $H_{norm}$", "#f5f5f5")

    # 连接 Output
    # 这里画一个小圆点作为分叉点
    x_fork = x5 + 1.5
    ax.plot([x_fork, x6 - 1.1], [y_line, y_out_1], color=c_border, lw=1.5)  # 上分叉
    ax.plot([x_fork, x6 - 1.1], [y_line, y_out_2], color=c_border, lw=1.5)  # 下分叉

    # 标题 (可选)
    # ax.set_title("Figure 1: Compact Regime-Aware Inverse Reconstruction Pipeline", loc='left', pad=15, fontweight='bold')

    plt.tight_layout()
    plt.savefig('Figure1_Pipeline_Compact.png', dpi=300, bbox_inches='tight')
    plt.show()


# 运行
if __name__ == "__main__":
    draw_figure_1_compact()

# ==========================================
# 3. 绘图函数: Figure 2 Diagnostics
# ==========================================
def draw_figure_2(df_entropy, df_season_ppc, case_s2, case_s27):
    # 关闭 constrained_layout 以便手动控制
    fig = plt.figure(figsize=(14, 10))

    # [关键调整 1 & 2]
    # height_ratios: 保持原样
    # width_ratios=[0.85, 1.25]: 拉大右侧 (c) 图的宽度 (原为默认1:1)
    # hspace=0.2: 减小上下行间距，使 (b)(c) 向上移动
    # wspace=0.25: 保持左右间距适中
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[0.75, 1.25], width_ratios=[0.85, 1.25],
                           wspace=0.2, hspace=0.25)

    # --- Panel (a): PPC ---
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df_season_ppc['season'], df_season_ppc['ppc_final'], 'o-', color='#424242', lw=1.5, markersize=5,
             label='Avg PPC')
    ax1.axvspan(0.5, 2.5, color='#e3f2fd', alpha=0.5, label='Rank Era')
    ax1.axvspan(2.5, 27.5, color='#fff3e0', alpha=0.5, label='Percent Era')
    ax1.axvspan(27.5, 34.5, color='#e8f5e9', alpha=0.5, label='Save Era')
    ax1.axhline(0.9, ls='--', color='red', alpha=0.6, label='Target (0.9)')
    ax1.set_title('(a) Season-wise Posterior Predictive Consistency (PPC)', loc='left', fontsize=12, fontweight='bold')
    ax1.set_xlim(0.5, 34.5)
    y_min, y_max = df_season_ppc['ppc_final'].min(), df_season_ppc['ppc_final'].max()
    ax1.set_ylim(max(0, y_min - 0.1), min(1.05, y_max + 0.1))
    ax1.set_xlabel('Season', fontsize=10)
    ax1.set_ylabel('Consistency Rate', fontsize=10)
    ax1.legend(loc='lower right', frameon=True, ncol=5, fontsize=9)

    # --- Panel (b): Entropy ---
    ax2 = fig.add_subplot(gs[1, 0])
    order = ['Rank Era\n(S1-2)', 'Percent Era\n(S3-27)', 'Save Era\n(S28-34)']
    sns.violinplot(data=df_entropy, x='Regime', y='h_norm', hue='Regime', order=order, ax=ax2,
                   palette=['#90caf9', '#ffcc80', '#a5d6a7'], linewidth=1, legend=False)
    ax2.set_title('(b) Uncertainty ($H_{norm}$) Distribution by Regime', loc='left', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Normalized Entropy', fontsize=10)
    ax2.set_xlabel('')
    ax2.grid(axis='y', linestyle=':', alpha=0.5)

    # --- Panel (c): Cases ---
    # [关键调整 3]
    # hspace=0.25: 减小 c-1 和 c-2 之间的垂直空隙 (原自动布局可能很大)
    # subplot_spec=gs[1, 1]: 放入拉宽后的右下角网格
    gs_sub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1], hspace=0.25)

    def plot_case(ax, df_case, title, judge_shares_manual=None):
        if df_case.empty:
            ax.text(0.5, 0.5, "Data Missing", ha='center', va='center');
            return
        names = df_case['celebrity_name'].tolist()
        v_mean = df_case['vote_mean'].tolist()
        v_std = df_case['vote_std'].tolist()
        x = np.arange(len(names))
        width = 0.35
        if judge_shares_manual and len(judge_shares_manual) == len(names):
            pj = judge_shares_manual
        else:
            pj = [0.1] * len(names)
        ax.bar(x - width / 2, pj, width, label='Judge Share $p^J$', color='gray', alpha=0.6)
        ax.errorbar(x + width / 2, v_mean, yerr=v_std, fmt='o', color='#d32f2f', label='Est. Fan Share $v$', capsize=4)
        ax.set_xticks(x)
        short_names = [str(n).split(' ')[0] if len(str(n)) > 10 else str(n) for n in names]
        ax.set_xticklabels(short_names, fontsize=9, rotation=15)
        ax.set_title(title, loc='left', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')

    # Case 1
    n_s2 = len(case_s2)
    pj_s2 = [0.125] * n_s2
    ax3a = fig.add_subplot(gs_sub[0])
    plot_case(ax3a, case_s2, '(c-1) S2 W3: Rank Era Inversion (Jerry Rice)', pj_s2)

    # Case 2
    n_s27 = len(case_s27)
    pj_s27 = [0.16] * n_s27
    ax3b = fig.add_subplot(gs_sub[1])
    plot_case(ax3b, case_s27, '(c-2) S27 W8: Percent Era Anomaly (Bobby Bones)', pj_s27)

    # 手动调整边距，替代 tight_layout，防止 gridspec 参数失效
    plt.subplots_adjust(top=0.95, bottom=0.08, left=0.08, right=0.95)
    plt.savefig('Figure2_Diagnostics.png', dpi=300)
    print("Figure 2 saved (Adjusted Layout).")
# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # 文件路径 (请根据实际情况修改)

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. 获取项目根目录 (根据你的代码，往上回退两级)
    project_root = os.path.dirname(os.path.dirname(current_dir))

    # 3. 定义数据文件夹的路径 (避免重复写路径)
    # 假设数据都在 project_root/data/Q2data 下
    f_summary = os.path.join(project_root, "data", "Q1data", "contestant_posterior_summary.csv")
    f_diag = os.path.join(project_root, "data", "Q1data", "week_diagnostics.csv")
    # 1. 计算中间数据
    try:
        df_entropy, df_ppc, case_s2, case_s27 = process_data(f_summary, f_diag)
        print("中间数据计算完成。")

        # 2. 绘图
        draw_figure_1_horizontal()
        draw_figure_2(df_entropy, df_ppc, case_s2, case_s27)
        print("所有图表绘制完成！")

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"发生错误: {e}")
