import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def draw_fig3_ridgeline():
    fig, ax = plt.subplots(figsize=(7, 5))

    # 模拟数据 (均值, 方差)
    contestants = [
        ("Bobby Bones", 0.45, 0.08),  # 争议选手：方差大
        ("Milo M.", 0.35, 0.03),  # 稳定选手
        ("Evanna L.", 0.25, 0.03),
        ("Alexis R.", 0.15, 0.02)
    ]

    x = np.linspace(0, 0.8, 200)
    y_step = 0.8

    for i, (name, mu, sigma) in enumerate(reversed(contestants)):
        y = norm.pdf(x, mu, sigma)
        y = y / y.max() * 1.0  # 归一化高度

        # 绘制填充
        base = i * y_step
        ax.fill_between(x, base, base + y, facecolor='white', edgecolor='black', lw=1.2, zorder=i)

        # 标签
        ax.text(0, base + 0.2, name, ha='right', fontsize=10, weight='bold')

    # 标注
    ax.annotate("High Uncertainty\n(Wide Tail)", xy=(0.6, 2.5), xytext=(0.6, 1.5),
                arrowprops=dict(arrowstyle="->", color='black'), ha='center', fontsize=9)

    ax.set_yticks([])
    ax.set_xlabel("Latent Fan Vote Share (Posterior Density)", fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    draw_fig3_ridgeline()