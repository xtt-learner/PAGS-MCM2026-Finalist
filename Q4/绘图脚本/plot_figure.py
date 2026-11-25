import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import json
import os
from sklearn.utils import resample

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
PATH_SUMMARY_CSV = os.path.join(project_root, "data", "Q4data", "q4_week_summary.csv")
PATH_BASELINES_CSV = os.path.join(project_root, "data", "Q4data", "q4_week_baselines_PAD.csv")
PATH_ELIM_PROB_CSV = os.path.join(project_root, "data", "Q4data", "q4_contestant_elim_prob.csv")
PATH_CONFIG_JSON = os.path.join(project_root, "data", "Q4data", "q4_run_config.json")
# --- Global Color Palette for Minimalist Aesthetic ---
COLOR_MAIN_BLUE = '#005b96'  # For PAGS, main highlights
COLOR_MUTED_GRAY = '#95a5a6'  # For Baseline/Percent, neutral elements
COLOR_DARK_GRAY = '#333333'  # For text, lines, edges
COLOR_LIGHT_GRAY = '#f8f9fa'  # Very light gray for box fills/backgrounds
COLOR_HIGHLIGHT_RED = '#c0392b'  # For limits, specific callouts
COLOR_RANK_GRAY = '#607d8b'  # A distinct gray-blue for 'Rank' method

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['text.color'] = COLOR_DARK_GRAY
plt.rcParams['axes.labelcolor'] = COLOR_DARK_GRAY
plt.rcParams['xtick.color'] = COLOR_DARK_GRAY
plt.rcParams['ytick.color'] = COLOR_DARK_GRAY
plt.rcParams['axes.edgecolor'] = COLOR_DARK_GRAY


# ==============================================================================
# Figure 6-1: PAGS Flowchart (Minimalist Color)
# ==============================================================================
def draw_compact_flowchart_colored():
    OSTYLE = {
        'bg': '#FFFFFF',
        'box_edge': COLOR_DARK_GRAY, 'box_fill': COLOR_LIGHT_GRAY,
        'text_main': COLOR_DARK_GRAY, 'text_sub': COLOR_MUTED_GRAY,
        'line': COLOR_DARK_GRAY, 'line_width': 1.5,
        'title_font': {'size': 14, 'weight': 'bold', 'color': COLOR_DARK_GRAY},
        'main_font': {'size': 10, 'weight': 'bold', 'color': COLOR_DARK_GRAY},
        'detail_font': {'size': 8, 'weight': 'normal', 'color': COLOR_DARK_GRAY},
        'sub_font': {'size': 8, 'style': 'italic', 'color': COLOR_MAIN_BLUE},  # Highlight sub-text with blue
        'box_padding': 0.1,
        'box_h': 0.8, 'box_w_small': 2.8, 'box_w_med': 3.2
    }

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10);
    ax.set_ylim(0, 7)
    ax.set_facecolor(OSTYLE['bg']);
    ax.axis('off')

    def draw_node(cx, cy, w, h, text_lines, sub_text=None):
        x, y = cx - w / 2, cy - h / 2
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={OSTYLE['box_padding']},rounding_size=0.1",
                                     fc=OSTYLE['box_fill'], ec=OSTYLE['box_edge'], lw=OSTYLE['line_width'], zorder=10)
        ax.add_patch(box)
        num_lines = len(text_lines)
        for i, line in enumerate(text_lines):
            y_pos = cy + (num_lines / 2.0 - 0.5 - i) * 0.25
            font_style = OSTYLE['main_font'] if i == 0 else OSTYLE['detail_font']
            ax.text(cx, y_pos, line, ha='center', va='center', zorder=11, **font_style)
        if sub_text:
            ax.text(x + w + OSTYLE['box_padding'] + 0.05, cy, sub_text, ha='left', va='center', zorder=11,
                    **OSTYLE['sub_font'])
        return (cx, cy + h / 2), (cx, cy - h / 2), (cx - w / 2, cy), (cx + w / 2, cy)

    def draw_arrow(start_xy, end_xy, style="->", rad=0.0, color=None):
        if color is None: color = OSTYLE['line']
        arrow = patches.FancyArrowPatch(start_xy, end_xy, arrowstyle=f"{style},head_width=8,head_length=10",
                                        connectionstyle=f"arc3,rad={rad}", color=color, lw=OSTYLE['line_width'],
                                        zorder=5)
        ax.add_patch(arrow)

    ax.text(5.0, 6.7, "Figure 6-1. PAGS weekly elimination pipeline", ha='center', va='center', **OSTYLE['title_font'])

    i1_t, i1_b, i1_l, i1_r = draw_node(2.5, 5.5, OSTYLE['box_w_small'], OSTYLE['box_h'],
                                       [r"Input 1: Judge Scores", r"$J_{i,s,t} \to p_{i,s,t}(J)$"])
    i2_t, i2_b, i2_l, i2_r = draw_node(7.5, 5.5, OSTYLE['box_w_small'], OSTYLE['box_h'],
                                       [r"Input 2: Fan Shares", r"$\{v_{s,t}^{(m)}\}_{m=1}^{M}$"])
    ma_t, ma_b, ma_l, ma_r = draw_node(7.5, 4.0, OSTYLE['box_w_small'], OSTYLE['box_h'],
                                       [r"Mod A: Damping ($\gamma_{s,t}$)", r"$v^{(m)}\to \tilde v^{(m)}$"],
                                       "auditable")
    mb_t, mb_b, mb_l, mb_r = draw_node(5.0, 2.5, OSTYLE['box_w_med'], OSTYLE['box_h'] + 0.2,
                                       [r"Mod B: Weighting ($w_{s,t}$)", r"$S^{(m)}=w p(J)+(1-w)\tilde v$"],
                                       "auditable")
    mc_t, mc_b, mc_l, mc_r = draw_node(5.0, 1.0, OSTYLE['box_w_small'], OSTYLE['box_h'],
                                       [r"Mod C: Guardrail", r"Protect Top-K by $p(J)$"], "auditable")
    o_t, o_b, o_l, o_r = draw_node(5.0, -0.5, OSTYLE['box_w_med'], OSTYLE['box_h'],
                                   [r"Output: Elimination Set", r"$E^{PAGS}$ & $P_{elim}$ (Monte Carlo)$"])

    draw_arrow(i2_b, ma_t);
    draw_arrow(ma_b, (mb_r[0] - 0.2, mb_t[1]), rad=-0.2)
    draw_arrow(i1_b, (mb_l[0] + 0.2, mb_t[1]), rad=0.2);
    draw_arrow(mb_b, mc_t);
    draw_arrow(mc_b, o_t)

    ax.set_ylim(-1.0, 7.0)
    plt.tight_layout();
    plt.subplots_adjust(top=0.95, bottom=0.05)
    plt.savefig("Figure_6_1_PAGS_pipeline_colored.png", dpi=300, bbox_inches='tight')
    print("Figure 6-1 generated.")


# ==============================================================================
# Table 6-1: Configuration Table (Minimalist Color)
# ==============================================================================
def generate_pags_config_table_colored():
    try:
        with open(PATH_CONFIG_JSON, 'r') as f: config = json.load(f)
    except:
        print("Config file not found."); return

    param_metadata = [
        {'key': 'M', 'symbol': '$M$', 'role': 'Monte Carlo iterations'},
        {'key': 'tau', 'symbol': '$\\tau$', 'role': 'Weight transition midpoint (week)'},
        {'key': 'kappa', 'symbol': '$\\kappa$', 'role': 'Logistic growth rate'},
        {'key': 'gamma_min', 'symbol': '$\\gamma_{min}$', 'role': 'Damping floor'},
        {'key': 'w_min', 'symbol': '$w_{min}$', 'role': 'Weight lower bound (early)'},
        {'key': 'w_max', 'symbol': '$w_{max}$', 'role': 'Weight upper bound (late)'},
        {'key': 'use_guardrail', 'symbol': '-', 'role': 'Enable judge guardrail'},
        {'key': 'guardrail_last_weeks', 'symbol': '$W_{GR}$', 'role': 'Guardrail window size'},
        {'key': 'guardrail_topK', 'symbol': '$K_{GR}$', 'role': 'Top-K protected'},
        {'key': 'finalists_default', 'symbol': '-', 'role': 'Default finalists'},
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

    fig, ax = plt.subplots(figsize=(7.5, 1.0 + len(table_data) * 0.4))
    ax.axis('off')
    table = ax.table(cellText=table_data, colLabels=["Parameter", "Symbol", "Value", "Role"],
                     colWidths=[0.2, 0.1, 0.15, 0.55], loc='center', cellLoc='left', edges='horizontal')
    table.auto_set_font_size(False);
    table.set_fontsize(10)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(COLOR_DARK_GRAY);
        cell.set_linewidth(0.5)
        cell.set_text_props(color=COLOR_DARK_GRAY)
        if row == 0:
            cell.set_text_props(weight='bold', color=COLOR_DARK_GRAY)
            cell.set_facecolor('#e9ecef')  # A very light, professional gray-blue for header
            cell.set_linewidth(1.0)
        cell.set_height(0.05)
        if col == 1 and row > 0: cell.set_text_props(size=11)

    plt.title("Table 6-1. PAGS Configuration (Reproducibility)", fontsize=11, weight='bold', pad=15, loc='left',
              color=COLOR_DARK_GRAY)
    plt.tight_layout(pad=0.5)
    plt.savefig('Table_6-1_PAGS_Config_colored.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    print("Table 6-1 generated.")


# ==============================================================================
# Figure 6-2: P-A-D Comparison (Minimalist Color)
# ==============================================================================
def plot_pad_comparison_colored():
    try:
        summary_df = pd.read_csv(PATH_SUMMARY_CSV)
        baselines_df = pd.read_csv(PATH_BASELINES_CSV)
        merged_df = pd.merge(summary_df, baselines_df, on=['season', 'week'], how='inner')
    except:
        print("Data for Fig 6-2 not found."); return

    metrics = ['P', 'A', 'D']
    mechanisms = {
        'Percent': {'col': ['P_pct', 'A_pct', 'D_pct'], 'color': COLOR_MUTED_GRAY, 'hatch': '///'},
        'Rank': {'col': ['P_rank', 'A_rank', 'D_rank'], 'color': COLOR_RANK_GRAY, 'hatch': '...'},
        'PAGS': {'col': ['P', 'A', 'D'], 'color': COLOR_MAIN_BLUE, 'hatch': ''}  # Solid for main method
    }

    def bootstrap_ci(data, n=1000, ci=95):
        means = [resample(data).mean() for _ in range(n)]
        return np.percentile(means, [(100 - ci) / 2, 100 - (100 - ci) / 2])

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
        bars = ax.bar(pos, plot_data[mech]['means'], width, label=mech, yerr=plot_data[mech]['yerrs'],
                      capsize=5, color=info['color'], hatch=info['hatch'], edgecolor=COLOR_DARK_GRAY, linewidth=1.0)
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9, color=COLOR_DARK_GRAY)

    ax.set_ylabel('Metric Mean Value', color=COLOR_DARK_GRAY)
    ax.set_title('Figure 6-2. Macro P–A–D comparison (with bootstrap CIs)', fontsize=12, fontweight='bold', pad=15,
                 color=COLOR_DARK_GRAY)
    ax.set_xticks(x);
    ax.set_xticklabels(metrics, color=COLOR_DARK_GRAY)
    ax.legend(title='Mechanism', frameon=False, labelcolor=COLOR_DARK_GRAY)
    ax.set_ylim(bottom=0);
    ax.yaxis.grid(True, linestyle='--', color='#cccccc', alpha=0.7);
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False);
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('Figure_6-2_PAD_Comparison_colored.png', dpi=300, bbox_inches='tight')
    print("Figure 6-2 generated.")


# ==============================================================================
# Figure 6-3: Bobby Bones (Minimalist Color)
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
    except:
        print("Data for Fig 6-3 not found."); return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Figure 6-3. Counterfactual correction in S27W8', fontsize=14, fontweight='bold',
                 color=COLOR_DARK_GRAY)

    # Panel A
    bars1 = ax1.bar(['Percent', 'PAGS'], [0.0, bobby_prob], color=[COLOR_MUTED_GRAY, COLOR_MAIN_BLUE],
                    edgecolor=COLOR_DARK_GRAY, linewidth=1.2)
    for bar, p in zip(bars1, [0.0, bobby_prob]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{p * 100:.1f}%", ha='center',
                 va='bottom', weight='bold', color=COLOR_DARK_GRAY)
    ax1.set_ylabel('$P_{elim}$', color=COLOR_DARK_GRAY);
    ax1.set_ylim(0, 1.0)
    ax1.set_title('Panel A: Bobby Bones $P_{elim}$', color=COLOR_DARK_GRAY)
    ax1.grid(axis='y', linestyle='--', color='#cccccc', alpha=0.7);
    ax1.spines['top'].set_visible(False);
    ax1.spines['right'].set_visible(False)

    # Panel B
    y_pos = np.arange(len(sorted_cont))
    colors = [COLOR_MAIN_BLUE if target in c else COLOR_MUTED_GRAY for c in sorted_cont['pair_id']]
    linewidths = [2.0 if target in c else 1.0 for c in sorted_cont['pair_id']]
    bars2 = ax2.barh(y_pos, sorted_cont['p_elim_pags'], color=colors, edgecolor=COLOR_DARK_GRAY, linewidth=linewidths)
    ax2.set_yticks(y_pos);
    ax2.set_yticklabels(sorted_cont['pair_id'], color=COLOR_DARK_GRAY);
    ax2.invert_yaxis()
    ax2.set_xlabel('$P_{elim}^{PAGS}$', color=COLOR_DARK_GRAY);
    ax2.set_xlim(0, 1.0)
    ax2.set_title('Panel B: All Contestants $P_{elim}^{PAGS}$', color=COLOR_DARK_GRAY)
    ax2.grid(axis='x', linestyle='--', color='#cccccc', alpha=0.7);
    ax2.spines['top'].set_visible(False);
    ax2.spines['right'].set_visible(False)

    caption = f"Note: S27W8 PAGS params: $w \\approx {week_sum['w']:.3f}$, $\\gamma = {week_sum['gamma']:.1f}$, Guardrail={bool(week_sum['guardrail_on'])}."
    fig.text(0.5, 0.02, caption, ha='center', fontsize=9, style='italic', color=COLOR_DARK_GRAY)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('Figure_6-3_BobbyBones_colored.png', dpi=300, bbox_inches='tight')
    print("Figure 6-3 generated.")


# ==============================================================================
# Figure 6-4: Jerry Rice (Minimalist Color)
# ==============================================================================
def plot_jerry_rice_colored():
    try:
        elim_df = pd.read_csv(PATH_ELIM_PROB_CSV)
        summary_df = pd.read_csv(PATH_SUMMARY_CSV)
        s, w = 2, 3;
        target = "Jerry Rice"
        week_elim = elim_df[(elim_df['season'] == s) & (elim_df['week'] == w)].copy()
        week_sum = summary_df[(summary_df['season'] == s) & (summary_df['week'] == w)].iloc[0]
        sorted_cont = week_elim.sort_values('p_elim_pags', ascending=True)
    except:
        print("Data for Fig 6-4 not found."); return

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(sorted_cont))
    colors = [COLOR_MAIN_BLUE if target in c else COLOR_MUTED_GRAY for c in sorted_cont['pair_id']]
    linewidths = [2.0 if target in c else 1.0 for c in sorted_cont['pair_id']]

    bars = ax.barh(y_pos, sorted_cont['p_elim_pags'], color=colors, edgecolor=COLOR_DARK_GRAY, linewidth=linewidths)

    for i, (c, p) in enumerate(zip(sorted_cont['pair_id'], sorted_cont['p_elim_pags'])):
        if target in c:
            ax.text(p + 0.01, i, f"{p * 100:.1f}%", va='center', weight='bold', color=COLOR_MAIN_BLUE)
            break

    ax.set_yticks(y_pos);
    ax.set_yticklabels(sorted_cont['pair_id'], color=COLOR_DARK_GRAY)
    ax.set_xlabel('$P_{elim}^{PAGS}$', color=COLOR_DARK_GRAY);
    ax.set_xlim(0, 1.0)
    ax.set_title('Figure 6-4. Moderation in S2W3 (Jerry Rice)', fontsize=14, fontweight='bold', pad=20,
                 color=COLOR_DARK_GRAY)
    ax.grid(axis='x', linestyle='--', color='#cccccc', alpha=0.7)
    ax.spines['top'].set_visible(False);
    ax.spines['right'].set_visible(False)

    caption = f"Note: S2W3 params: $w \\approx {week_sum['w']:.3f}$, $\\gamma = {week_sum['gamma']:.1f}$, Guardrail={bool(week_sum['guardrail_on'])}."
    fig.text(0.5, 0.01, caption, ha='center', fontsize=9, style='italic', color=COLOR_DARK_GRAY,
             bbox=dict(facecolor=COLOR_LIGHT_GRAY, edgecolor='none', pad=5.0))
    plt.tight_layout();
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('Figure_6-4_JerryRice_colored.png', dpi=300, bbox_inches='tight')
    print("Figure 6-4 generated.")


# ==============================================================================
# Figure 6-5: Gamma Distribution (Minimalist Color)
# ==============================================================================
def plot_gamma_dist_colored():
    try:
        summary_df = pd.read_csv(PATH_SUMMARY_CSV)
        with open(PATH_CONFIG_JSON, 'r') as f:
            config = json.load(f)
        gamma_min = config.get('gamma_min', 0.5)
    except:
        print("Data for Fig 6-5 not found."); return

    gammas = summary_df['gamma']
    rate = (gammas < 1.0).mean() * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(gammas, bins=30, color=COLOR_MUTED_GRAY, edgecolor=COLOR_DARK_GRAY, linewidth=1.0)
    ax.axvline(x=1.0, color=COLOR_DARK_GRAY, linestyle='--', linewidth=2.0, label=r'$\gamma=1.0$ (Uncompressed)')
    ax.axvline(x=gamma_min, color=COLOR_HIGHLIGHT_RED, linestyle=':', linewidth=2.0,
               label=f'$\gamma_{{min}}={gamma_min}$')

    ax.text(0.05, 0.95, f"Activated ($\gamma < 1$): {rate:.1f}%", transform=ax.transAxes, fontsize=11,
            fontweight='bold',
            va='top', color=COLOR_MAIN_BLUE,
            bbox=dict(facecolor='white', edgecolor=COLOR_MAIN_BLUE, boxstyle='round,pad=0.5'))

    ax.set_title('Figure 6-5. Damping activation ($\gamma$ distribution)', fontsize=14, fontweight='bold', pad=20,
                 color=COLOR_DARK_GRAY)
    ax.set_xlabel(r'Damping Factor ($\gamma$)', color=COLOR_DARK_GRAY);
    ax.set_ylabel('Frequency', color=COLOR_DARK_GRAY)
    ax.set_xlim(0, 1.1);
    ax.grid(axis='y', linestyle='--', color='#cccccc', alpha=0.7)
    ax.legend(loc='upper right', frameon=False, labelcolor=COLOR_DARK_GRAY)
    ax.spines['top'].set_visible(False);
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('Figure_6-5_Gamma_Dist_colored.png', dpi=300, bbox_inches='tight')
    print("Figure 6-5 generated.")


# ==============================================================================
# Figure S-1: Stage Weight Trajectory (Minimalist Color)
# ==============================================================================
def plot_stage_weight_colored():
    try:
        # 读取数据
        summary_df = pd.read_csv(PATH_SUMMARY_CSV)
        with open(PATH_CONFIG_JSON, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Data for Fig S-1 not found: {e}")
        return

    # --- 核心修复：恢复原版的计算逻辑 ---
    # 1. 理论权重 w 是由 'week' (时间) 决定的，而不是直接由人数决定的
    # 公式：w(t) = w_min + (w_max - w_min) / (1 + exp(-kappa * (t - tau)))
    # 注意：这里用的是 summary_df['week']，保证了每个点的理论值是准确对应那一周的
    summary_df['w_theoretical'] = config['w_min'] + (config['w_max'] - config['w_min']) / \
                                  (1 + np.exp(-config['kappa'] * (summary_df['week'] - config['tau'])))

    # 2. 为了画出连线，我们需要按人数(N)排序，因为x轴是N
    # 这样画出来的线才是真实的“轨迹”，而不是人工合成的平滑曲线
    theoretical_curve = summary_df.sort_values(by='n_contestants', ascending=False)

    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=(8, 6))

    # 1. 画实际散点 (Actual Weekly Points)
    ax.scatter(summary_df['n_contestants'], summary_df['w'],
               color=COLOR_MUTED_GRAY, alpha=0.6, s=40, zorder=3, label='Actual Weeks')

    # 2. 画理论轨迹 (Theoretical Trajectory) - 恢复原有特征
    ax.plot(theoretical_curve['n_contestants'], theoretical_curve['w_theoretical'],
            color=COLOR_MAIN_BLUE, linewidth=2.5, linestyle='--', zorder=2, label='Theoretical Logistic')

    # 3. 画上下界限 (Limits)
    ax.axhline(config['w_min'], color=COLOR_HIGHLIGHT_RED, ls=':', lw=1.5, zorder=1, label=f'$w_{{min}}$')
    ax.axhline(config['w_max'], color=COLOR_HIGHLIGHT_RED, ls=':', lw=1.5, zorder=1, label=f'$w_{{max}}$')

    # --- 样式调整 ---
    ax.invert_xaxis()  # 关键特征：x轴反转，从多人到少人
    ax.set_xlabel('Number of Contestants ($N_{s,t}$)', color=COLOR_DARK_GRAY, fontsize=11, weight='bold')
    ax.set_ylabel('Stage Weight ($w_{s,t}$)', color=COLOR_DARK_GRAY, fontsize=11, weight='bold')
    ax.set_title('Figure S-1. Stage weight trajectory (Actual vs. Theory)',
                 fontsize=14, fontweight='bold', pad=15, color=COLOR_DARK_GRAY)

    ax.grid(True, linestyle='--', color='#cccccc', alpha=0.6)
    ax.legend(loc='lower right',frameon=False, labelcolor=COLOR_DARK_GRAY)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('Figure_S-1_Stage_Weight_colored.png', dpi=300, bbox_inches='tight')
    print("Figure S-1 generated (Corrected).")

# ==============================================================================
# Figure S-2: P-A-D Boxplots (Minimalist Color)
# ==============================================================================
def plot_pad_boxplots_colored():
    try:
        s_df = pd.read_csv(PATH_SUMMARY_CSV)  # 使用全局变量路径
        b_df = pd.read_csv(PATH_BASELINES_CSV)  # 使用全局变量路径
        m_df = pd.merge(s_df, b_df, on=['season', 'week'], how='inner')
    except Exception as e:
        print(f"Data for Fig S-2 not found or error: {e}")
        return

    data = [m_df['P_pct'].dropna(), m_df['P'].dropna(),
            m_df['A_pct'].dropna(), m_df['A'].dropna(),
            m_df['D_pct'].dropna(), m_df['D'].dropna()]

    fig, ax = plt.subplots(figsize=(10, 6))
    pos = [1, 1.8, 3.5, 4.3, 6, 6.8]

    # --- 修正说明 ---
    # boxprops (箱体): 是 Patch，必须用 'edgecolor' (或 'ec')
    # whiskerprops (须): 是 Line2D，必须用 'color' (不能用 'edgecolor')
    # capprops (盖): 是 Line2D，必须用 'color' (不能用 'edgecolor')
    # medianprops (中位线): 是 Line2D，必须用 'color'

    bp = ax.boxplot(data, positions=pos, widths=0.6, patch_artist=True, showmeans=True,
                    meanprops={"marker": "o", "mfc": COLOR_DARK_GRAY, "mec": COLOR_DARK_GRAY, "ms": 4},
                    medianprops={'color': COLOR_DARK_GRAY, 'lw': 1.5},

                    # 修正点 1: 箱体使用 edgecolor
                    boxprops={'lw': 1.2, 'edgecolor': COLOR_DARK_GRAY},

                    # 修正点 2: 须使用 color
                    whiskerprops={'lw': 1.2, 'color': COLOR_DARK_GRAY},

                    # 修正点 3: 盖使用 color
                    capprops={'lw': 1.2, 'color': COLOR_DARK_GRAY},

                    flierprops={'marker': 'x', 'mec': COLOR_MUTED_GRAY})

    colors = [COLOR_MUTED_GRAY, COLOR_MAIN_BLUE] * 3
    hatches = ['///', ''] * 3  # Hatch for baseline, solid for PAGS
    for patch, color, hatch in zip(bp['boxes'], colors, hatches):
        patch.set_facecolor(color)
        patch.set_hatch(hatch)

    ax.set_xticks([1.4, 3.9, 6.4])
    ax.set_xticklabels(['P', 'A', 'D'], fontsize=12, weight='bold', color=COLOR_DARK_GRAY)

    legend_patches = [patches.Rectangle((0, 0), 1, 1, fc=COLOR_MUTED_GRAY, ec=COLOR_DARK_GRAY, hatch='///'),
                      patches.Rectangle((0, 0), 1, 1, fc=COLOR_MAIN_BLUE, ec=COLOR_DARK_GRAY, hatch='')]
    ax.legend(legend_patches, ['Percent (Baseline)', 'PAGS'], loc='upper left', frameon=False,
              labelcolor=COLOR_DARK_GRAY)

    ax.set_ylabel('Metric Value', color=COLOR_DARK_GRAY)
    ax.set_title('Figure S-2. P-A-D distribution (boxplots)', fontsize=14, fontweight='bold', pad=15,
                 color=COLOR_DARK_GRAY)
    ax.yaxis.grid(True, linestyle='--', color='#cccccc', alpha=0.6)
    ax.set_ylim(0, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('Figure_S-2_PAD_Boxplots_colored.png', dpi=300, bbox_inches='tight')
    print("Figure S-2 generated.")


# --- Run all plotting functions ---
if __name__ == "__main__":
    draw_compact_flowchart_colored()
    generate_pags_config_table_colored()
    plot_pad_comparison_colored()
    plot_bobby_bones_colored()
    plot_jerry_rice_colored()
    plot_gamma_dist_colored()
    plot_stage_weight_colored()
    plot_pad_boxplots_colored()
    print("\nAll colored figures generated successfully.")