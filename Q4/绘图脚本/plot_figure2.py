import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import json
from sklearn.utils import resample
import os

# ==========================================
# 【路径配置区】请根据实际情况修改
# ==========================================
# 建议使用绝对路径或确保文件在当前目录下
# 如果使用相对路径，请确保 cmd/terminal 的运行目录正确

# 这里使用了 os.path 动态获取当前脚本所在目录的上一级/数据目录
# 你可以简单改为: PATH_SUMMARY_CSV = 'q4_week_summary.csv'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
PATH_SUMMARY_CSV = os.path.join(project_root, "data", "Q4data", "q4_week_summary.csv")
PATH_BASELINES_CSV = os.path.join(project_root, "data", "Q4data", "q4_week_baselines_PAD.csv")
PATH_ELIM_PROB_CSV = os.path.join(project_root, "data", "Q4data", "q4_contestant_elim_prob.csv")
PATH_CONFIG_JSON = os.path.join(project_root, "data", "Q4data", "q4_run_config.json")

# ==========================================
# 【配色方案配置】Modern Academic Focus Palette
# ==========================================
# 1. 主角 (PAGS, 重点选手, 理论线): 亮眼但不刺眼的专业蓝
COLOR_PRIMARY = '#2E86C1'

# 2. 配角 (Percent, 背景选手, 实际散点): 中性灰，退后一步
COLOR_SECONDARY = '#BDC3C7'
COLOR_SECONDARY_DARK = '#7F8C8D'  # 用于边框或深色背景元素

# 3. 第三人 (Rank): 紫色，区分度高
COLOR_TERTIARY = '#884EA0'

# 4. 辅助 (界限, 警告): 柔和红
COLOR_ACCENT = '#E74C3C'

# 5. 基础 (文本, 轴线, 填充): 深蓝灰 & 极淡蓝
COLOR_TEXT = '#2C3E50'
COLOR_BG_LIGHT = '#F4F6F7'  # 极淡的灰蓝色背景，比纯白更有质感

# 全局样式设置
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei']  # 加上SimHei防中文乱码(如有)
plt.rcParams['text.color'] = COLOR_TEXT
plt.rcParams['axes.labelcolor'] = COLOR_TEXT
plt.rcParams['xtick.color'] = COLOR_TEXT
plt.rcParams['ytick.color'] = COLOR_TEXT
plt.rcParams['axes.edgecolor'] = COLOR_TEXT
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


# ==============================================================================
# Figure 6-1: PAGS Flowchart (Modern Look)
# ==============================================================================
def draw_compact_flowchart_colored():
    # Style Config
    OSTYLE = {
        'box_edge': COLOR_PRIMARY,  # 蓝色边框，强调模块感
        'box_fill': '#EBF5FB',  # 非常淡的蓝色填充
        'text_main': COLOR_TEXT,
        'line': COLOR_TEXT,
        'line_width': 1.5,
        'sub_font_color': COLOR_PRIMARY  # 关键公式用主色
    }

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10);
    ax.set_ylim(0, 7)
    ax.axis('off')

    def draw_node(cx, cy, w, h, text_lines, sub_text=None):
        x, y = cx - w / 2, cy - h / 2
        # 使用圆角矩形，增加一点投影感(zorder)
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.1,rounding_size=0.15",
                                     fc=OSTYLE['box_fill'], ec=OSTYLE['box_edge'],
                                     lw=OSTYLE['line_width'], zorder=10)
        ax.add_patch(box)

        for i, line in enumerate(text_lines):
            y_pos = cy + (len(text_lines) / 2.0 - 0.5 - i) * 0.25
            weight = 'bold' if i == 0 else 'normal'
            size = 10 if i == 0 else 8.5
            ax.text(cx, y_pos, line, ha='center', va='center', zorder=11,
                    fontsize=size, weight=weight, color=OSTYLE['text_main'])

        if sub_text:
            ax.text(x + w + 0.15, cy, sub_text, ha='left', va='center', zorder=11,
                    fontsize=8, style='italic', color=OSTYLE['sub_font_color'])
        return (cx, cy + h / 2), (cx, cy - h / 2), (cx - w / 2, cy), (cx + w / 2, cy)

    def draw_arrow(start, end, rad=0.0):
        arrow = patches.FancyArrowPatch(start, end, arrowstyle="->,head_width=6,head_length=8",
                                        connectionstyle=f"arc3,rad={rad}",
                                        color=OSTYLE['line'], lw=OSTYLE['line_width'], zorder=5)
        ax.add_patch(arrow)

    ax.text(5.0, 6.7, "Figure 6-1. PAGS weekly elimination pipeline",
            ha='center', va='center', fontsize=14, weight='bold', color=COLOR_TEXT)

    # Drawing Nodes
    i1 = draw_node(2.5, 5.5, 2.8, 0.8, [r"Input 1: Judge Scores", r"$J_{i,s,t} \to p_{i,s,t}(J)$"])
    i2 = draw_node(7.5, 5.5, 2.8, 0.8, [r"Input 2: Fan Shares", r"$\{v_{s,t}^{(m)}\}_{m=1}^{M}$"])
    ma = draw_node(7.5, 4.0, 2.8, 0.8, [r"Mod A: Damping ($\gamma_{s,t}$)", r"$v^{(m)}\to \tilde v^{(m)}$"],
                   "auditable")
    mb = draw_node(5.0, 2.5, 3.2, 1.0, [r"Mod B: Weighting ($w_{s,t}$)", r"$S^{(m)}=w p(J)+(1-w)\tilde v$"],
                   "auditable")
    mc = draw_node(5.0, 1.0, 2.8, 0.8, [r"Mod C: Guardrail", r"Protect Top-K by $p(J)$"], "auditable")
    out = draw_node(5.0, -0.5, 3.2, 0.8, [r"Output: Elimination Set", r"$E^{PAGS}$ & $P_{elim}$ (Monte Carlo)$"])

    # Drawing Connections
    draw_arrow(i2[1], ma[0])
    draw_arrow(ma[1], (mb[3][0] - 0.2, mb[0][1]), rad=-0.2)
    draw_arrow(i1[1], (mb[2][0] + 0.2, mb[0][1]), rad=0.2)
    draw_arrow(mb[1], mc[0])
    draw_arrow(mc[1], out[0])

    ax.set_ylim(-1.0, 7.0)
    plt.tight_layout()
    plt.savefig("Figure_6_1_PAGS_pipeline_colored.png", dpi=300, bbox_inches='tight')
    print("Generated: Figure 6-1")


# ==============================================================================
# Table 6-1: Configuration Table (Nice Header)
# ==============================================================================
def generate_pags_config_table_colored():
    try:
        with open(PATH_CONFIG_JSON, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Err Fig Table: {e}"); return

    param_metadata = [
        {'key': 'M', 'symbol': '$M$', 'role': 'Monte Carlo iterations'},
        {'key': 'tau', 'symbol': '$\\tau$', 'role': 'Weight transition midpoint'},
        {'key': 'kappa', 'symbol': '$\\kappa$', 'role': 'Logistic growth rate'},
        {'key': 'gamma_min', 'symbol': '$\\gamma_{min}$', 'role': 'Damping floor'},
        {'key': 'w_min', 'symbol': '$w_{min}$', 'role': 'Weight lower bound'},
        {'key': 'w_max', 'symbol': '$w_{max}$', 'role': 'Weight upper bound'},
        {'key': 'use_guardrail', 'symbol': '-', 'role': 'Enable judge guardrail'},
        {'key': 'guardrail_last_weeks', 'symbol': '$W_{GR}$', 'role': 'Guardrail window size'},
        {'key': 'guardrail_topK', 'symbol': '$K_{GR}$', 'role': 'Top-K protected'},
        {'key': 'eps_vote', 'symbol': '$\\epsilon_{v}$', 'role': 'Numerical stability epsilon'},
    ]
    table_data = []
    for item in param_metadata:
        val = config.get(item['key'], 'N/A')
        if isinstance(val, bool):
            val_str = "True" if val else "False"
        elif isinstance(val, float) and (abs(val) < 1e-4 and val != 0):
            val_str = f"{val:.1e}"
        elif isinstance(val, float):
            val_str = f"{val:.4g}"
        else:
            val_str = str(val)
        table_data.append([item['key'].replace('_', ' ').title(), item['symbol'], val_str, item['role']])

    fig, ax = plt.subplots(figsize=(7.5, 1.2 + len(table_data) * 0.4))
    ax.axis('off')

    # 颜色配置
    header_color = '#D6EAF8'  # 淡蓝色表头
    row_colors = ['#FFFFFF', '#F8F9F9']  # 斑马纹行

    table = ax.table(cellText=table_data, colLabels=["Parameter", "Symbol", "Value", "Role"],
                     colWidths=[0.25, 0.1, 0.15, 0.5], loc='center', cellLoc='left', edges='horizontal')
    table.auto_set_font_size(False);
    table.set_fontsize(10)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#BDC3C7')
        cell.set_linewidth(0.5)
        cell.set_height(0.06)
        if row == 0:
            cell.set_text_props(weight='bold', color=COLOR_TEXT)
            cell.set_facecolor(header_color)
            cell.set_linewidth(1.2)
        else:
            cell.set_facecolor(row_colors[row % 2])  # 斑马纹
            cell.set_text_props(color=COLOR_TEXT)
        if col == 1 and row > 0: cell.set_text_props(size=11)  # Symbol slightly larger

    plt.title("Table 6-1. PAGS Configuration (Reproducibility)", fontsize=11, weight='bold', pad=10, loc='left',
              color=COLOR_TEXT)
    plt.tight_layout()
    plt.savefig('Table_6-1_PAGS_Config_colored.png', dpi=300, bbox_inches='tight')
    print("Generated: Table 6-1")


# ==============================================================================
# Figure 6-2: P-A-D Comparison (High Contrast)
# ==============================================================================
def plot_pad_comparison_colored():
    try:
        summary_df = pd.read_csv(PATH_SUMMARY_CSV)
        baselines_df = pd.read_csv(PATH_BASELINES_CSV)
        merged_df = pd.merge(summary_df, baselines_df, on=['season', 'week'], how='inner')
    except Exception as e:
        print(f"Err Fig 6-2: {e}"); return

    metrics = ['P', 'A', 'D']
    # 这里的颜色策略：PAGS 用主色突显，Percent 用浅灰退后，Rank 用紫色区分
    mechanisms = {
        'Percent': {'col': ['P_pct', 'A_pct', 'D_pct'], 'color': COLOR_SECONDARY, 'edge': COLOR_SECONDARY_DARK},
        'Rank': {'col': ['P_rank', 'A_rank', 'D_rank'], 'color': COLOR_TERTIARY, 'edge': COLOR_TEXT},
        'PAGS': {'col': ['P', 'A', 'D'], 'color': COLOR_PRIMARY, 'edge': COLOR_TEXT}
    }

    def bootstrap_ci(data, n=1000):
        means = [resample(data).mean() for _ in range(n)]
        return np.percentile(means, [2.5, 97.5])

    plot_data = {}
    for mech, info in mechanisms.items():
        plot_data[mech] = {'means': [], 'yerrs': [[], []]}
        for metric_col in info['col']:
            data = merged_df[metric_col].dropna()
            mean = data.mean()
            ci_low, ci_high = bootstrap_ci(data)
            plot_data[mech]['means'].append(mean)
            plot_data[mech]['yerrs'][0].append(mean - ci_low)
            plot_data[mech]['yerrs'][1].append(ci_high - mean)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics));
    width = 0.25

    for i, (mech, info) in enumerate(mechanisms.items()):
        pos = x + (i - 1) * width
        # 增加透明度 alpha=0.9 让颜色不那么死板
        bars = ax.bar(pos, plot_data[mech]['means'], width, label=mech, yerr=plot_data[mech]['yerrs'],
                      capsize=4, color=info['color'], edgecolor=info['edge'], linewidth=1.0, alpha=0.9, zorder=3)

        # 数值标注颜色与柱子匹配或用深色
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9, color=COLOR_TEXT, weight='bold')

    ax.set_ylabel('Metric Mean Value', fontsize=11, weight='bold')
    ax.set_title('Figure 6-2. Macro P–A–D comparison (Bootstrap CIs)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x);
    ax.set_xticklabels(['Popularity (P)', 'Accuracy (A)', 'Decisiveness (D)'], fontsize=11)

    # 图例优化
    ax.legend(title='Mechanism', frameon=False, loc='upper right')

    ax.set_ylim(bottom=0)
    ax.grid(axis='y', linestyle='--', color='#E5E7E9', alpha=0.8, zorder=0)
    ax.spines['top'].set_visible(False);
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('Figure_6-2_PAD_Comparison_colored.png', dpi=300, bbox_inches='tight')
    print("Generated: Figure 6-2")


# ==============================================================================
# Figure 6-3: Bobby Bones (Target Highlight)
# ==============================================================================
def plot_bobby_bones_colored():
    try:
        elim_df = pd.read_csv(PATH_ELIM_PROB_CSV)
        summary_df = pd.read_csv(PATH_SUMMARY_CSV)
        s, w = 27, 8;
        target = "Bobby Bones"
        week_elim = elim_df[(elim_df['season'] == s) & (elim_df['week'] == w)].copy()
        week_sum = summary_df[(summary_df['season'] == s) & (summary_df['week'] == w)].iloc[0]
        bobby_prob = week_elim[week_elim['pair_id'].str.contains(target, case=False)]['p_elim_pags'].values[0]
        sorted_cont = week_elim.sort_values('p_elim_pags', ascending=False)
    except Exception as e:
        print(f"Err Fig 6-3: {e}"); return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 2]})
    fig.suptitle('Figure 6-3. Counterfactual correction: Bobby Bones (S27W8)', fontsize=14, fontweight='bold',
                 color=COLOR_TEXT)

    # Panel A: 对比
    # Percent 用灰色，PAGS 用亮蓝色，形成鲜明对比
    bars1 = ax1.bar(['Percent', 'PAGS'], [0.0, bobby_prob],
                    color=[COLOR_SECONDARY, COLOR_PRIMARY], edgecolor=COLOR_TEXT, linewidth=1.0, width=0.6)
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h * 100:.1f}%", ha='center', va='bottom',
                 weight='bold', color=COLOR_PRIMARY)

    ax1.set_ylabel('Elimination Probability ($P_{elim}$)', fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.set_title('Panel A: Target Comparison', fontsize=12, color=COLOR_SECONDARY_DARK)
    ax1.grid(axis='y', linestyle='--', color='#E5E7E9')

    # Panel B: 全局视角
    y_pos = np.arange(len(sorted_cont))
    # 颜色逻辑：目标人物用主色，其他人用浅灰色
    colors = [COLOR_PRIMARY if target in c else '#E5E7E9' for c in sorted_cont['pair_id']]
    edgecolors = [COLOR_TEXT if target in c else '#BDC3C7' for c in sorted_cont['pair_id']]

    bars2 = ax2.barh(y_pos, sorted_cont['p_elim_pags'], color=colors, edgecolor=edgecolors, height=0.7)

    # 标注目标人物数值
    for i, (c, p) in enumerate(zip(sorted_cont['pair_id'], sorted_cont['p_elim_pags'])):
        if target in c:
            ax2.text(p + 0.02, i, f"{p * 100:.1f}%", va='center', weight='bold', color=COLOR_PRIMARY, fontsize=11)
            break

    ax2.set_yticks(y_pos)
    # 简化名字显示，只保留人名
    short_names = [n.split('||')[1] if '||' in n else n for n in sorted_cont['pair_id']]
    ax2.set_yticklabels(short_names, fontsize=10)
    ax2.invert_yaxis()
    ax2.set_xlabel('PAGS Elimination Probability', fontsize=11)
    ax2.set_xlim(0, 1.05)
    ax2.set_title('Panel B: All Contestants (Ranked)', fontsize=12, color=COLOR_SECONDARY_DARK)
    ax2.grid(axis='x', linestyle='--', color='#E5E7E9')

    caption = f"Note: Params $w \\approx {week_sum['w']:.3f}$, $\\gamma = {week_sum['gamma']:.1f}$. High PAGS prob reflects correction of vote/score imbalance."
    fig.text(0.5, 0.02, caption, ha='center', fontsize=9, style='italic', color=COLOR_SECONDARY_DARK)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('Figure_6-3_BobbyBones_colored.png', dpi=300, bbox_inches='tight')
    print("Generated: Figure 6-3")


# ==============================================================================
# Figure 6-4: Jerry Rice (Moderation Highlight)
# ==============================================================================
def plot_jerry_rice_colored():
    try:
        elim_df = pd.read_csv(PATH_ELIM_PROB_CSV)
        summary_df = pd.read_csv(PATH_SUMMARY_CSV)
        s, w = 2, 3;
        target = "Jerry Rice"
        week_elim = elim_df[(elim_df['season'] == s) & (elim_df['week'] == w)].copy()
        week_sum = summary_df[(summary_df['season'] == s) & (summary_df['week'] == w)].iloc[0]
        sorted_cont = week_elim.sort_values('p_elim_pags', ascending=True)  # Ascending for display
    except Exception as e:
        print(f"Err Fig 6-4: {e}"); return

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(sorted_cont))

    # 颜色策略：Jerry Rice 是重点(蓝色)，已经被淘汰的高概率者用(红色警示)，安全的用(绿色)?
    # 不，保持简约：Jerry 蓝色，其他人灰色。
    colors = [COLOR_PRIMARY if target in c else '#E5E7E9' for c in sorted_cont['pair_id']]
    edgecolors = [COLOR_TEXT if target in c else '#BDC3C7' for c in sorted_cont['pair_id']]

    bars = ax.barh(y_pos, sorted_cont['p_elim_pags'], color=colors, edgecolor=edgecolors, height=0.7)

    for i, (c, p) in enumerate(zip(sorted_cont['pair_id'], sorted_cont['p_elim_pags'])):
        if target in c:
            ax.text(p + 0.01, i, f"{p * 100:.1f}% (Moderate Risk)", va='center', weight='bold', color=COLOR_PRIMARY)
        elif p > 0.8:  # 极高风险者
            ax.text(p + 0.01, i, f"{p * 100:.0f}%", va='center', color=COLOR_SECONDARY_DARK, fontsize=9)

    short_names = [n.split('||')[1] if '||' in n else n for n in sorted_cont['pair_id']]
    ax.set_yticks(y_pos);
    ax.set_yticklabels(short_names, color=COLOR_TEXT)
    ax.set_xlabel('PAGS Elimination Probability', color=COLOR_TEXT, weight='bold')
    ax.set_xlim(0, 1.05)
    ax.set_title('Figure 6-4. Moderation Case: Jerry Rice (S2W3)', fontsize=14, fontweight='bold', pad=15)

    ax.grid(axis='x', linestyle='--', color='#E5E7E9')
    ax.spines['top'].set_visible(False);
    ax.spines['right'].set_visible(False)

    caption = f"Note: Despite low scores, Jerry is not guaranteed elimination ($24.7\\%$). Params: $w \\approx {week_sum['w']:.3f}$, $\\gamma = {week_sum['gamma']:.1f}$."
    fig.text(0.5, 0.01, caption, ha='center', fontsize=9, style='italic', color=COLOR_SECONDARY_DARK,
             bbox=dict(facecolor=COLOR_BG_LIGHT, edgecolor='none', pad=5.0))
    plt.tight_layout();
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('Figure_6-4_JerryRice_colored.png', dpi=300, bbox_inches='tight')
    print("Generated: Figure 6-4")


# ==============================================================================
# Figure 6-5: Gamma Distribution (Activation Highlight)
# ==============================================================================
def plot_gamma_dist_colored():
    try:
        summary_df = pd.read_csv(PATH_SUMMARY_CSV)
        with open(PATH_CONFIG_JSON, 'r') as f:
            config = json.load(f)
        gamma_min = config.get('gamma_min', 0.5)
    except Exception as e:
        print(f"Err Fig 6-5: {e}"); return

    gammas = summary_df['gamma']
    rate = (gammas < 1.0).mean() * 100

    fig, ax = plt.subplots(figsize=(8, 6))

    # 直方图用次要色(灰)，表示"大多数时候是安静的"
    n, bins, patches = ax.hist(gammas, bins=30, color=COLOR_SECONDARY, edgecolor='white', linewidth=0.5, alpha=0.8)

    # 强调 gamma=1.0 的柱子 (通常是最后一个bin)
    # 这里简单画条线，或者如果有 bin 恰好在 1.0，可以变色。
    # 我们用线来表示界限

    # 辅助线：Min Limit (红色)
    ax.axvline(x=gamma_min, color=COLOR_ACCENT, linestyle=':', linewidth=2.0,
               label=f'Floor $\gamma_{{min}}={gamma_min}$')

    # 辅助线：Uncompressed (主色虚线)
    ax.axvline(x=1.0, color=COLOR_TEXT, linestyle='--', linewidth=2.0, label=r'No Damping ($\gamma=1.0$)')

    # 文字标注框
    text = f"Damping Active\n($\gamma < 1$): {rate:.1f}%"
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=11, fontweight='bold',
            va='top', color=COLOR_PRIMARY,
            bbox=dict(facecolor='white', edgecolor=COLOR_PRIMARY, boxstyle='round,pad=0.5', alpha=0.9))

    ax.set_title('Figure 6-5. Damping Activation Frequency', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel(r'Damping Factor ($\gamma$)', fontsize=11)
    ax.set_ylabel('Frequency (Weeks)', fontsize=11)
    ax.set_xlim(0, 1.1)

    ax.legend(loc='upper right', frameon=False)
    ax.grid(axis='y', linestyle='--', color='#E5E7E9')
    ax.spines['top'].set_visible(False);
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('Figure_6-5_Gamma_Dist_colored.png', dpi=300, bbox_inches='tight')
    print("Generated: Figure 6-5")


# ==============================================================================
# Figure S-1: Stage Weight (Real Data & Clean Look)
# ==============================================================================
def plot_stage_weight_colored():
    try:
        summary_df = pd.read_csv(PATH_SUMMARY_CSV)
        with open(PATH_CONFIG_JSON, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Err Fig S-1: {e}"); return

    # 计算理论值 (用于对比)
    t = summary_df['week']
    w_min, w_max, kappa, tau = config['w_min'], config['w_max'], config['kappa'], config['tau']
    summary_df['w_theoretical'] = w_min + (w_max - w_min) / (1 + np.exp(-kappa * (t - tau)))
    plot_df = summary_df.sort_values(by='n_contestants', ascending=False)

    fig, ax = plt.subplots(figsize=(8, 6))

    # 1. 实际散点：用灰色，设为背景
    ax.scatter(summary_df['n_contestants'], summary_df['w'],
               color=COLOR_SECONDARY, alpha=0.5, s=40, zorder=2, label='Actual Data')

    # 2. 理论轨迹：用主色(蓝)，线条加粗
    ax.plot(plot_df['n_contestants'], plot_df['w_theoretical'],
            color=COLOR_PRIMARY, linewidth=2.5, linestyle='--', zorder=3, label='Theoretical Rule')

    # 3. 上下界限：用红色
    ax.axhline(w_min, color=COLOR_ACCENT, ls=':', lw=1.5, zorder=1, label='Bounds')
    ax.axhline(w_max, color=COLOR_ACCENT, ls=':', lw=1.5, zorder=1)

    ax.invert_xaxis()
    ax.set_xlabel('Number of Contestants ($N_{s,t}$)', fontsize=11, weight='bold')
    ax.set_ylabel('Stage Weight ($w_{s,t}$)', fontsize=11, weight='bold')
    ax.set_title('Figure S-1. Stage Weight Trajectory (Empirical)', fontsize=14, fontweight='bold', pad=15)

    ax.grid(True, linestyle='--', color='#E5E7E9', alpha=0.6)
    ax.legend(frameon=True, facecolor='white', edgecolor='none', loc='lower left')
    ax.spines['top'].set_visible(False);
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('Figure_S-1_Stage_Weight_colored.png', dpi=300, bbox_inches='tight')
    print("Generated: Figure S-1")


# ==============================================================================
# Figure S-2: Boxplots (Aesthetic Fix)
# ==============================================================================
def plot_pad_boxplots_colored():
    try:
        s_df = pd.read_csv(PATH_SUMMARY_CSV)
        b_df = pd.read_csv(PATH_BASELINES_CSV)
        m_df = pd.merge(s_df, b_df, on=['season', 'week'], how='inner')
    except Exception as e:
        print(f"Err Fig S-2: {e}"); return

    data = [m_df['P_pct'].dropna(), m_df['P'].dropna(),
            m_df['A_pct'].dropna(), m_df['A'].dropna(),
            m_df['D_pct'].dropna(), m_df['D'].dropna()]

    fig, ax = plt.subplots(figsize=(10, 6))
    pos = [1, 1.8, 3.5, 4.3, 6, 6.8]

    # 这里的颜色策略：Baseline 填充浅灰，边框深灰；PAGS 填充淡蓝，边框深蓝

    bp = ax.boxplot(data, positions=pos, widths=0.6, patch_artist=True, showmeans=True,
                    meanprops={"marker": "o", "mfc": COLOR_TEXT, "mec": "none", "ms": 5},
                    medianprops={'color': COLOR_TEXT, 'lw': 1.5},
                    flierprops={'marker': 'x', 'mec': COLOR_SECONDARY_DARK, 'alpha': 0.5})

    # 分别设置颜色
    # 偶数索引是 Baseline (Percent)，奇数是 PAGS
    colors_fill = [COLOR_BG_LIGHT, '#D6EAF8'] * 3  # 浅灰 vs 浅蓝背景
    colors_edge = [COLOR_SECONDARY_DARK, COLOR_PRIMARY] * 3  # 深灰 vs 深蓝边框

    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors_fill[i])
        patch.set_edgecolor(colors_edge[i])
        patch.set_linewidth(1.2)

    # 设置须和盖的颜色统一为文本色，或者跟随箱子颜色？为了整洁，统一用深灰色
    plt.setp(bp['whiskers'], color=COLOR_SECONDARY_DARK, linewidth=1.0)
    plt.setp(bp['caps'], color=COLOR_SECONDARY_DARK, linewidth=1.0)

    ax.set_xticks([1.4, 3.9, 6.4])
    ax.set_xticklabels(['Popularity', 'Accuracy', 'Decisiveness'], fontsize=12, weight='bold')

    # 自定义图例
    legend_patches = [
        patches.Rectangle((0, 0), 1, 1, fc=COLOR_BG_LIGHT, ec=COLOR_SECONDARY_DARK, lw=1.2),
        patches.Rectangle((0, 0), 1, 1, fc='#D6EAF8', ec=COLOR_PRIMARY, lw=1.2)
    ]
    ax.legend(legend_patches, ['Percent (Baseline)', 'PAGS (Ours)'], loc='upper left', frameon=False)

    ax.set_ylabel('Metric Distribution', fontsize=11)
    ax.set_title('Figure S-2. Metric Consistency Analysis', fontsize=14, fontweight='bold', pad=15)
    ax.yaxis.grid(True, linestyle='--', color='#E5E7E9')
    ax.set_ylim(0, 1.05)
    ax.spines['top'].set_visible(False);
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('Figure_S-2_PAD_Boxplots_colored.png', dpi=300, bbox_inches='tight')
    print("Generated: Figure S-2")


# --- Execution ---
if __name__ == "__main__":
    print("Starting aesthetic update...")
    draw_compact_flowchart_colored()
    generate_pags_config_table_colored()
    plot_pad_comparison_colored()
    plot_bobby_bones_colored()
    plot_jerry_rice_colored()
    plot_gamma_dist_colored()
    plot_stage_weight_colored()
    plot_pad_boxplots_colored()
    print("\nAll figures updated with Modern Academic Palette.")