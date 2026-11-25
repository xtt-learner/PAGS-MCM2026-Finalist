import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_q1_framework_style_fig2():
    # 1. 设置画布
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')  # 关闭坐标轴

    # --- 配色方案 (完全复刻 Figure 2) ---
    colors = {
        'input_fill': '#DAE8FC',  # 浅蓝
        'input_edge': '#6C8EBF',  # 深蓝
        'proc_fill': '#F8CECC',  # 浅红
        'proc_edge': '#B85450',  # 深红
        'out_fill': '#D5E8D4',  # 浅绿
        'out_edge': '#82B366',  # 深绿
        'text': '#333333',  # 深灰文字
        'line': '#555555'  # 连接线颜色
    }

    # 字体设置
    font_main = {'family': 'sans-serif', 'size': 10, 'weight': 'normal', 'color': colors['text']}
    font_bold = {'family': 'sans-serif', 'size': 11, 'weight': 'bold', 'color': colors['text']}

    # --- 绘图辅助函数 ---
    def draw_node(center_x, center_y, width, height, fill, edge, text_lines, z=10):
        # 计算左下角坐标
        x = center_x - width / 2
        y = center_y - height / 2

        # 绘制圆角矩形
        box = patches.FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.1,rounding_size=0.15",  # 圆角
            facecolor=fill, edgecolor=edge, linewidth=2, zorder=z
        )
        ax.add_patch(box)

        # 绘制多行文字
        total_lines = len(text_lines)
        for i, line in enumerate(text_lines):
            # 简单的垂直居中逻辑
            offset = (total_lines - 1) * 0.18 / 2 - i * 0.18
            # 第一行加粗 (Title)
            font_use = font_bold if i == 0 else font_main
            ax.text(center_x, center_y + offset, line,
                    ha='center', va='center', **font_use, zorder=z + 1)

        # 【修正点】：将坐标打包成两个元组返回
        top_anchor = (center_x, y + height + 0.1 + 0.2)  # +0.2是padding
        bottom_anchor = (center_x, y - 0.1 - 0.2)
        return top_anchor, bottom_anchor

    def draw_arrow(start_xy, end_xy):
        # 绘制直线箭头
        ax.annotate("",
                    xy=end_xy, xytext=start_xy,
                    arrowprops=dict(arrowstyle="->", color=colors['line'], lw=2.0),
                    zorder=1)

    # --- 1. 定义节点数据 (Top-Down Layout) ---

    # Layer 1: Inputs (y=8.0)
    # 左边：连续型数据，右边：离散约束数据
    l1_nodes = [
        {'x': 3.0, 'w': 2.8, 'text': ["Metric Inputs", "Judge Scores ($J_{s,t}$)", "Active Set ($C_{s,t}$)"],
         'type': 'input'},
        {'x': 7.0, 'w': 2.8,
         'text': ["Constraint Inputs", "Elimination Result ($E_{s,t}$)", "Rule Config ($\mathcal{R}$)"],
         'type': 'input'}
    ]

    # Layer 2: Core Components (y=5.0)
    # 拆解为：先验 -> 兼容性 -> 推断
    l2_nodes = [
        {'x': 2.0, 'w': 2.4, 'text': ["Step 1: Prior Model", "Weak Trend Dirichlet", "$v \sim Dir(\\alpha \cdot b)$"],
         'type': 'proc'},
        {'x': 5.0, 'w': 2.6,
         'text': ["Step 2: Compatibility", "Rule Check $\mathbb{I}(v \in \mathcal{A})$", "Does $v$ match $E$?"],
         'type': 'proc'},
        {'x': 8.0, 'w': 2.4, 'text': ["Step 3: Inference", "Rejection Sampling", "Filter Proposals"], 'type': 'proc'}
    ]

    # Layer 3: Output (y=2.0)
    l3_nodes = [
        {'x': 5.0, 'w': 4.0, 'text': ["Posterior Outputs (Q1 Result)", "Latent Vote Samples $\{v^{(m)}\}$",
                                      "Uncertainty Metrics ($\sigma, H$)"], 'type': 'out'}
    ]

    # 存储连接点
    l1_anchors = []
    l2_anchors = []
    l3_anchors = []

    # --- 2. 绘制节点 ---

    # Draw L1
    for n in l1_nodes:
        top, bottom = draw_node(n['x'], 8.0, n['w'], 1.2, colors['input_fill'], colors['input_edge'], n['text'])
        l1_anchors.append((top, bottom))

    # Draw L2
    for n in l2_nodes:
        top, bottom = draw_node(n['x'], 5.0, n['w'], 1.3, colors['proc_fill'], colors['proc_edge'], n['text'])
        l2_anchors.append((top, bottom))

    # Draw L3
    for n in l3_nodes:
        top, bottom = draw_node(n['x'], 2.0, n['w'], 1.2, colors['out_fill'], colors['out_edge'], n['text'])
        l3_anchors.append((top, bottom))

    # --- 3. 绘制连接线 (Logic Flow) ---

    # Logic:
    # Inputs -> Components
    # Judge Scores (L1_0) drives Prior (L2_0) AND Compatibility (L2_1)
    draw_arrow(l1_anchors[0][1], l2_anchors[0][0])  # Input 1 -> Prior
    draw_arrow(l1_anchors[0][1], l2_anchors[1][0])  # Input 1 -> Constraint (Need scores to calc total)

    # Elimination (L1_1) drives Compatibility (L2_1) ONLY
    draw_arrow(l1_anchors[1][1], l2_anchors[1][0])  # Input 2 -> Constraint

    # Components -> Output
    # All three components define the Posterior
    draw_arrow(l2_anchors[0][1], l3_anchors[0][0])  # Prior -> Output
    draw_arrow(l2_anchors[1][1], l3_anchors[0][0])  # Constraint -> Output
    draw_arrow(l2_anchors[2][1], l3_anchors[0][0])  # Inference -> Output

    # --- 4. 添加标题 ---
    plt.text(5.0, 9.5, "Figure 1: Bayesian Inverse Estimation Framework (Q1 Methodology)",
             ha='center', fontsize=14, fontweight='bold', color='#222222')

    plt.text(5.0, 9.1, "Reconstructing Latent Fan Votes from Sparse Elimination Constraints",
             ha='center', fontsize=11, style='italic', color='#555555')

    plt.tight_layout()
    # 保存为透明背景
    plt.savefig('q1_bayesian_framework_fig2_style.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()


# 运行绘图
if __name__ == "__main__":
    draw_q1_framework_style_fig2()