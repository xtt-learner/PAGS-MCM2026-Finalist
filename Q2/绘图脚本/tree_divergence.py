import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- O-Style 极简黑白灰配色配置 ---
OSTYLE = {
    'box_fill': '#FFFFFF',  # 纯白背景 (最干净)
    'box_edge': '#333333',  # 深灰边框 (比纯黑柔和，更有质感)
    'text_main': '#000000',  # 主标题黑色
    'text_sub': '#666666',  # 副标题灰色
    'arrow_color': '#333333'  # 箭头深灰
}


def draw_clean_box(ax, x, y, w, h, text_list, highlight=False):
    """绘制符合O奖审美标准的极简框"""
    # 高亮逻辑：重点结果(Highlight)用粗边框 (lw=2.5)，普通框用细边框 (lw=1.2)
    lw = 2.5 if highlight else 1.2

    # 绘制圆角矩形 (Rounding size=0.1 保持硬朗的圆角)
    box = patches.FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                 boxstyle="round,pad=0.05,rounding_size=0.1",
                                 facecolor=OSTYLE['box_fill'],
                                 edgecolor=OSTYLE['box_edge'],
                                 linewidth=lw,
                                 zorder=10)
    ax.add_patch(box)

    # 渲染文字
    for i, txt in enumerate(text_list):
        if i == 0:
            # 第一行：主标题 (Bold, Black)
            ax.text(x, y + 0.12, txt, ha='center', va='center',
                    fontsize=12, fontweight='bold', color=OSTYLE['text_main'], zorder=11)
        else:
            # 第二行：技术细节/公式 (Italic, Grey, Smaller)
            ax.text(x, y - 0.18, txt, ha='center', va='center',
                    fontsize=10, fontstyle='italic', color=OSTYLE['text_sub'], zorder=11)


def plot_q2_divergence_bw():
    """绘制 Q2 反事实分歧树 (Counterfactual Divergence Tree)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')  # 关闭坐标轴

    # --- 布局坐标 ---
    top_xy = (5.0, 5.2)  # 顶部输入
    mid_l_xy = (2.5, 3.2)  # 左分支 (Rule A)
    mid_r_xy = (7.5, 3.2)  # 右分支 (Rule B)
    bot_l_xy = (2.5, 1.2)  # 左结果 (Eliminated)
    bot_r_xy = (7.5, 1.2)  # 右结果 (Saved - Highlight)

    # --- 1. 绘制方框 ---

    # 顶部输入节点
    draw_clean_box(ax, *top_xy, 3.2, 1.0,
                   ["Input: Season 4 Week 8", r"Score $\ll$ Avg, Fan $\gg$ Avg"])

    # 中间规则节点
    draw_clean_box(ax, *mid_l_xy, 3.0, 1.0,
                   ["Rule A: Percent", r"minimize $(Score\% + Vote\%)$"])
    draw_clean_box(ax, *mid_r_xy, 3.0, 1.0,
                   ["Rule B: Rank", r"maximize $(Rank^J + Rank^F)$"])

    # 底部结果节点
    # 左边：普通淘汰
    draw_clean_box(ax, *bot_l_xy, 2.8, 1.0,
                   ["Eliminated", "Billy Ray Cyrus"], highlight=False)

    # 右边：高亮存活 (反事实结果)
    draw_clean_box(ax, *bot_r_xy, 2.8, 1.0,
                   ["Saved / Safe", "Rank Inversion Effect"], highlight=True)

    # --- 2. 绘制连线 (极简箭头) ---
    arrow_props = dict(arrowstyle="->", color=OSTYLE['arrow_color'], lw=1.5, shrinkA=0, shrinkB=0)

    # 路径 1: Input -> Rule A -> Eliminated
    ax.annotate("", xy=(mid_l_xy[0], mid_l_xy[1] + 0.5), xytext=(top_xy[0], top_xy[1] - 0.5), arrowprops=arrow_props)
    ax.annotate("", xy=(bot_l_xy[0], bot_l_xy[1] + 0.5), xytext=(mid_l_xy[0], mid_l_xy[1] - 0.5),
                arrowprops=arrow_props)

    # 路径 2: Input -> Rule B -> Saved
    ax.annotate("", xy=(mid_r_xy[0], mid_r_xy[1] + 0.5), xytext=(top_xy[0], top_xy[1] - 0.5), arrowprops=arrow_props)
    ax.annotate("", xy=(bot_r_xy[0], bot_r_xy[1] + 0.5), xytext=(mid_r_xy[0], mid_r_xy[1] - 0.5),
                arrowprops=arrow_props)

    # --- 3. 辅助标注 ---
    # 中间加一个淡淡的 "vs."
    ax.text(5.0, 3.2, "vs.", ha='center', va='center',
            fontsize=12, fontstyle='italic', color='#999999')

    plt.tight_layout()
    plt.savefig('q2_divergence_tree_ostyle.png', dpi=300, bbox_inches='tight')
    plt.show()


# 执行绘图
if __name__ == "__main__":
    plot_q2_divergence_bw()