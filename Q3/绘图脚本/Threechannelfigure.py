import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- O-Style 配置 ---
OSTYLE = {
    'box_fill': '#FFFFFF', 'box_edge': '#333333',
    'text_main': '#000000', 'text_sub': '#666666',
    'arrow': '#333333', 'line_width': 1.5
}


def draw_clean_box(ax, x, y, w, h, title, subtext=None, highlight=False):
    """绘制符合O奖审美的极简节点"""
    lw = 2.0 if highlight else OSTYLE['line_width']
    # 圆角矩形
    box = patches.FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                 boxstyle="round,pad=0.05,rounding_size=0.1",
                                 fc=OSTYLE['box_fill'], ec=OSTYLE['box_edge'], lw=lw, zorder=10)
    ax.add_patch(box)

    # 文字
    ax.text(x, y + (0.1 if subtext else 0), title, ha='center', va='center',
            weight='bold', fontsize=11, color=OSTYLE['text_main'], zorder=11)
    if subtext:
        ax.text(x, y - 0.15, subtext, ha='center', va='center',
                style='italic', fontsize=9, color=OSTYLE['text_sub'], zorder=11)
    # 返回连接锚点 (Right, Left)
    return (x + w / 2, y), (x - w / 2, y)


def draw_arrow(ax, start_xy, end_xy, style="->"):
    """绘制连接箭头"""
    ax.annotate("", xy=end_xy, xytext=start_xy,
                arrowprops=dict(arrowstyle=style, color=OSTYLE['arrow'], lw=OSTYLE['line_width'], shrinkA=5, shrinkB=5),
                zorder=5)


def plot_q3_architecture():
    """绘制 Q3 三通道模型架构图"""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # --- 1. 定义节点位置 ---
    # 左侧输入
    input_pos = (2.0, 3.5)

    # 中间三通道 (自上而下)
    tech_pos = (6.0, 5.5)  # Technical Channel
    pop_pos = (6.0, 3.5)  # Popularity Channel
    het_pos = (6.0, 1.5)  # Heterogeneity Channel

    # 右侧输出
    judge_out_pos = (10.0, 5.0)  # Judge Outcome
    fan_out_pos = (10.0, 2.0)  # Fan Outcome

    # --- 2. 绘制节点框 ---

    # Input Node
    input_r, _ = draw_clean_box(ax, *input_pos, 2.8, 1.5, "Input Covariates",
                                r"Age, Gender, Style, Week ($X_{i,s,t}$)")

    # Channel Nodes (核心)
    tech_r, tech_l = draw_clean_box(ax, *tech_pos, 3.2, 1.2, "Channel 1: Technical",
                                    r"Latent Skill & Effort ($T_{i,s,t}$)")
    pop_r, pop_l = draw_clean_box(ax, *pop_pos, 3.2, 1.2, "Channel 2: Popularity",
                                  r"Charisma & Narrative ($P_{i,s,t}$)")
    het_r, het_l = draw_clean_box(ax, *het_pos, 3.2, 1.2, "Channel 3: Heterogeneity",
                                  r"Contestant Baseline ($\gamma_i$)")

    # Output Nodes
    _, judge_l = draw_clean_box(ax, *judge_out_pos, 2.8, 1.2, "Observed Output 1",
                                r"Judge Scores ($J_{i,s,t}$)")
    _, fan_l = draw_clean_box(ax, *fan_out_pos, 2.8, 1.2, "Observed Output 2",
                              r"Fan Votes ($v_{i,s,t}$)")

    # --- 3. 绘制连接箭头 (Flow) ---

    # Input -> Channels (广播式输入)
    draw_arrow(ax, input_r, tech_l)
    draw_arrow(ax, input_r, pop_l)
    # Heterogeneity 主要是个体ID驱动，但也可以认为受输入特征约束，这里画上表示它是模型的一部分
    draw_arrow(ax, input_r, het_l, style="-|>")  # 使用不同的箭头风格表示这主要是随机效应的来源

    # Channels -> Outputs (汇聚式输出)
    # Judge Score 主要受 Technical 和 Heterogeneity 影响
    draw_arrow(ax, tech_r, judge_l)
    draw_arrow(ax, het_r, judge_l)
    # Fan Vote 主要受 Popularity 和 Heterogeneity 影响
    draw_arrow(ax, pop_r, fan_l)
    draw_arrow(ax, het_r, fan_l)

    # (可选) 添加弱连接：技术可能轻微影响粉丝，人气可能轻微影响评委
    # 使用虚线表示次要/潜在影响
    ax.annotate("", xy=fan_l, xytext=tech_r,
                arrowprops=dict(arrowstyle="->", color='#999999', lw=1.2, linestyle='--'))
    ax.annotate("", xy=judge_l, xytext=pop_r,
                arrowprops=dict(arrowstyle="->", color='#999999', lw=1.2, linestyle='--'))

    # --- 4. 添加区域标注 ---
    # 在中间区域上方加一个大括号或标题，强调这是潜在机制
    ax.text(6.0, 6.6, "Latent Mechanisms (Mixed-Effects Engine)", ha='center',
            fontsize=12, fontweight='bold', color=OSTYLE['text_main'])

    # 画一条竖虚线隔开输入和模型
    ax.vlines(4.0, 0.5, 6.5, colors='#BBBBBB', linestyles='--', lw=1.0)
    # 画一条竖虚线隔开模型和输出
    ax.vlines(8.0, 0.5, 6.5, colors='#BBBBBB', linestyles='--', lw=1.0)

    plt.tight_layout()
    plt.savefig('q3_three_channel_architecture_ostyle.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_q3_architecture()