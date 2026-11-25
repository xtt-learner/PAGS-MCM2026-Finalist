import matplotlib.pyplot as plt
import numpy as np


def draw_fig5_scissors():
    fig, ax = plt.subplots(figsize=(7, 5))

    age = np.linspace(20, 70, 100)
    # 模拟逻辑：评委分随年龄直线下降，粉丝分持平或微升
    z_judge = 1.0 - 0.04 * (age - 20)
    z_fan = 0.5 - 0.01 * (age - 20) + 0.0003 * (age - 30) ** 2

    # 绘制线条
    ax.plot(age, z_judge, 'k-', lw=2, label='Judge Score (Technical)')
    ax.plot(age, z_fan, 'k--', lw=2, label='Fan Support (Sympathy)')

    # 填充剪刀差区域
    ax.fill_between(age, z_judge, z_fan, where=(z_fan > z_judge),
                    color='#E0E0E0', alpha=0.5, hatch='///', label='Sympathy Gap')

    # 装饰
    ax.set_xlabel("Contestant Age", fontsize=11, weight='bold')
    ax.set_ylabel("Standardized Effect (Z-Score)", fontsize=11, weight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 标注
    ax.annotate("The 'Sympathy Gap'", xy=(55, 0.0), xytext=(35, -0.5),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color='black'),
                fontsize=10, weight='bold', bbox=dict(boxstyle="round", fc="white", ec="black"))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    draw_fig5_scissors()