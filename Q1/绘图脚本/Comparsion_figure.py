import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- 统一配色风格 (Figure 2 Style) ---
STYLE = {
    'blue_fill': '#DAE8FC', 'blue_edge': '#6C8EBF',  # Inputs / Judges
    'red_fill': '#F8CECC', 'red_edge': '#B85450',  # Methods / Contrast
    'green_fill': '#D5E8D4', 'green_edge': '#82B366',  # Outcome / Fans
    'text': '#333333',
    'line': '#555555',
    'highlight': '#FF0000'  # 用于强调逆转
}


def draw_box(ax, x, y, w, h, fill, edge, text_list, font_size=10, z=10):
    box = patches.FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                 boxstyle="round,pad=0.1,rounding_size=0.15",
                                 facecolor=fill, edgecolor=edge, linewidth=2, zorder=z)
    ax.add_patch(box)
    for i, txt in enumerate(text_list):
        weight = 'bold' if i == 0 else 'normal'
        ax.text(x, y + (len(text_list) - 1) * 0.06 - i * 0.12, txt, ha='center', va='center',
                fontsize=font_size, fontweight=weight, color=STYLE['text'], zorder=z + 1)
    return (x, y + h / 2), (x, y - h / 2)  # Return Top/Bottom anchors


def plot_q1_rank_inversion():
    """
    Q1可视化：展示Jerry Rice如何通过粉丝投票实现惊天逆转
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 标题
    ax.text(5, 9.5, "Q1 Insight: The Judge-Fan Rank Inversion (Case: Jerry Rice, S2)",
            ha='center', fontsize=14, fontweight='bold')

    # 模拟数据 (Rank 1 is Top)
    contestants = [
        {'name': 'Stacy Keibler', 'j_rank': 1, 'f_rank': 2},
        {'name': 'Drew Lachey', 'j_rank': 2, 'f_rank': 1},
        {'name': 'Lisa Rinna', 'j_rank': 3, 'f_rank': 4},
        {'name': 'George H.', 'j_rank': 4, 'f_rank': 5},
        {'name': 'Jerry Rice', 'j_rank': 5, 'f_rank': 3},  # 逆转选手
        {'name': 'Tia Carrere', 'j_rank': 6, 'f_rank': 6}  # 被淘汰
    ]

    # 绘制两列
    left_x, right_x = 2.5, 7.5
    y_start = 8.0
    y_gap = 1.2

    # Column Headers
    draw_box(ax, left_x, 9.0, 3.0, 0.6, STYLE['blue_fill'], STYLE['blue_edge'],
             ["Judge Technical Rank", "(Based on Scores)"])
    draw_box(ax, right_x, 9.0, 3.0, 0.6, STYLE['green_fill'], STYLE['green_edge'],
             ["Latent Fan Rank", "(Posterior Estimate)"])

    for item in contestants:
        # Calculate Y positions (Rank 1 at top)
        y_j = y_start - (item['j_rank'] - 1) * y_gap
        y_f = y_start - (item['f_rank'] - 1) * y_gap

        is_jerry = item['name'] == 'Jerry Rice'

        # 绘制连接线
        line_color = '#D9534F' if is_jerry else '#AAAAAA'
        line_width = 3.0 if is_jerry else 1.0
        alpha = 1.0 if is_jerry else 0.4
        z = 5 if is_jerry else 1

        ax.plot([left_x + 1.0, right_x - 1.0], [y_j, y_f], color=line_color, lw=line_width, alpha=alpha, zorder=z)

        # 绘制左侧节点 (Judge)
        box_color = STYLE['red_fill'] if is_jerry else '#F0F0F0'
        edge_color = STYLE['red_edge'] if is_jerry else '#BBBBBB'
        draw_box(ax, left_x, y_j, 2.0, 0.7, box_color, edge_color, [f"#{item['j_rank']} {item['name']}"], 9)

        # 绘制右侧节点 (Fan)
        draw_box(ax, right_x, y_f, 2.0, 0.7, box_color, edge_color, [f"#{item['f_rank']} {item['name']}"], 9)

    # 标注
    ax.annotate("Massive Popularity Lift", xy=(5, 5.5), xytext=(5, 6.5),
                arrowprops=dict(arrowstyle="->", color='red'), ha='center', color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig('q1_rank_inversion.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_q2_divergence_tree():
    """
    Q2可视化：反事实分歧树，展示规则如何改变命运
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # 标题
    ax.text(6, 7.5, "Q2 Result: Counterfactual Divergence (Example Week)",
            ha='center', fontsize=14, fontweight='bold')

    # 1. 顶部：输入数据
    top_pos = (6, 6.5)
    draw_box(ax, top_pos[0], top_pos[1], 4.0, 1.2, STYLE['blue_fill'], STYLE['blue_edge'],
             ["Input: Season 4 Week 8", "Billy Ray Cyrus (Low Score / High Fan)",
              "Joey Fatone (High Score / High Fan)"])

    # 2. 中间：规则分支
    mid_y = 4.0
    left_mid = (3, mid_y)
    right_mid = (9, mid_y)

    draw_box(ax, left_mid[0], left_mid[1], 3.5, 1.0, STYLE['red_fill'], STYLE['red_edge'],
             ["Scenario A: Percent Rule", "(Sensitive to Score Gaps)"])

    draw_box(ax, right_mid[0], right_mid[1], 3.5, 1.0, STYLE['red_fill'], STYLE['red_edge'],
             ["Scenario B: Rank Rule", "(Ignores Score Magnitude)"])

    # 3. 底部：结果
    bot_y = 1.5
    left_bot = (3, bot_y)
    right_bot = (9, bot_y)

    draw_box(ax, left_bot[0], left_bot[1], 3.0, 1.2, '#E6E6E6', '#999999',
             ["Eliminated:", "Billy Ray Cyrus", "(Gap too large to bridge)"])

    draw_box(ax, right_bot[0], right_bot[1], 3.0, 1.2, STYLE['green_fill'], STYLE['green_edge'],
             ["Saved / Safe:", "Billy Ray Cyrus", "(Rank Inversion Protects)"])

    # 4. 连线
    props = dict(arrowstyle="->", color=STYLE['line'], lw=2)

    # Input -> Rules
    ax.annotate("", xy=(left_mid[0], mid_y + 0.6), xytext=(top_pos[0], top_pos[1] - 0.7), arrowprops=props)
    ax.annotate("", xy=(right_mid[0], mid_y + 0.6), xytext=(top_pos[0], top_pos[1] - 0.7), arrowprops=props)

    # Rules -> Outcomes
    ax.annotate("", xy=(left_bot[0], bot_y + 0.7), xytext=(left_mid[0], mid_y - 0.6), arrowprops=props)
    ax.annotate("", xy=(right_bot[0], bot_y + 0.7), xytext=(right_mid[0], mid_y - 0.6), arrowprops=props)

    # 标注
    ax.text(6, 2.5, "CRITICAL DIVERGENCE", ha='center', va='center',
            fontsize=12, fontweight='bold', color='#D9534F', bbox=dict(facecolor='white', edgecolor='none'))

    plt.tight_layout()
    plt.savefig('q2_divergence_tree.png', dpi=300, bbox_inches='tight')
    plt.show()


# 运行绘制
if __name__ == "__main__":
    plot_q1_rank_inversion()
    plot_q2_divergence_tree()