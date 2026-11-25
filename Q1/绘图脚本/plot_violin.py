import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- 设置绘图风格 ---
sns.set_theme(style="whitegrid")
# 字体设置：优先尝试 SimHei，如果不行回退到 Arial (防止提交到没有中文字体的环境报错)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def draw_violin_plot(csv_path):
    # 获取绝对路径以进行调试打印 (方便你看它到底在找哪里)
    abs_path = os.path.abspath(csv_path)

    if not os.path.exists(csv_path):
        print(f"❌ 找不到文件: {csv_path}")
        print(f"   (系统尝试寻找的绝对路径是: {abs_path})")
        print("💡 请检查：\n1. 你的脚本是否在项目根目录下？\n2. 是否已经运行 Q1 代码生成了 csv？")
        return

    print(f"正在读取数据: {csv_path} ...")
    df = pd.read_csv(csv_path)

    # 创建画布
    plt.figure(figsize=(12, 8))

    # 绘制小提琴图
    sns.violinplot(data=df, x="vote_share", y="celebrity_name",
                   hue="celebrity_name", inner="quartile", palette="muted", legend=False)

    plt.title("粉丝投票分布概率估算 (Posterior Distribution of Fan Votes)", fontsize=16)
    plt.xlabel("估算的粉丝得票率 (Vote Share)", fontsize=12)
    plt.ylabel("选手 (Celebrity)", fontsize=12)

    plt.xlim(0, None)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    current_script_dir = os.path.dirname(os.path.abspath(__file__))


    project_root = os.path.dirname(os.path.dirname(current_script_dir))

    data_path = os.path.join(project_root, "data", "Q1data", "violin_samples.csv")

    # 运行画图
    draw_violin_plot(data_path)