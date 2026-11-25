import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# --- 设置绘图风格 (MCM 标准英文风格) ---
sns.set_theme(style="whitegrid")
# 字体设置：优先尝试 Arial/Helvetica，如果没有则回退到 sans-serif
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def plot_mechanism_comparison(csv_path):
    # 1. 智能路径检查
    if not os.path.exists(csv_path):
        # 尝试自动定位 (针对你的项目结构)
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))  # 回退到根目录
            auto_path = os.path.join(project_root, "data", "Q1data", "week_diagnostics.csv")
            if os.path.exists(auto_path):
                csv_path = auto_path
            else:
                print(f"❌ File not found at: {csv_path}")
                return
        except:
            print(f"❌ Path error. Please check if week_diagnostics.csv exists.")
            return

    print(f"Reading data: {csv_path}")
    df = pd.read_csv(csv_path)

    # 2. 准备绘图画布 (1行2列)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- 子图 1: Acceptance Rate 对比 (争议度) ---
    sns.boxplot(x='mechanism', y='accept_rate', data=df, ax=axes[0], palette="Set2")
    axes[0].set_title('Distribution of Model Acceptance Rate', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Voting Mechanism', fontsize=12)
    axes[0].set_ylabel('Acceptance Rate (Probability)', fontsize=12)

    # --- 子图 2: Margin Mean 对比 (竞争激烈度) ---
    sns.boxplot(x='mechanism', y='margin_mean', data=df, ax=axes[1], palette="Set2")
    axes[1].set_title('Distribution of Safety Margin (Competitiveness)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Voting Mechanism', fontsize=12)
    axes[1].set_ylabel('Margin Mean (Distance to Elimination)', fontsize=12)

    # 3. 添加统计显著性测试 (Mann-Whitney U Test)
    # 自动识别数据中的两种机制 (percent vs rank)
    mech_types = df['mechanism'].unique()

    if len(mech_types) >= 2:
        m1, m2 = mech_types[0], mech_types[1]  # 通常是 'percent' 和 'rank'

        # --- 测试 Acceptance Rate ---
        data1 = df[df['mechanism'] == m1]['accept_rate'].dropna()
        data2 = df[df['mechanism'] == m2]['accept_rate'].dropna()
        u_stat, p_val = stats.mannwhitneyu(data1, data2)

        # 在图上标注 P值 (如果极小用科学计数法)
        p_text = f'p < 0.001' if p_val < 0.001 else f'p = {p_val:.3f}'
        # 标注位置：图顶端中间
        axes[0].text(0.5, 0.95, f'Mann-Whitney Test\n{p_text}',
                     transform=axes[0].transAxes, ha='center', va='top',
                     color='crimson', fontsize=11, fontweight='bold',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        # --- 测试 Margin Mean ---
        data1 = df[df['mechanism'] == m1]['margin_mean'].dropna()
        data2 = df[df['mechanism'] == m2]['margin_mean'].dropna()
        u_stat, p_val = stats.mannwhitneyu(data1, data2)

        p_text = f'p < 0.001' if p_val < 0.001 else f'p = {p_val:.3f}'
        axes[1].text(0.5, 0.95, f'Mann-Whitney Test\n{p_text}',
                     transform=axes[1].transAxes, ha='center', va='top',
                     color='crimson', fontsize=11, fontweight='bold',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 这里的路径写法是为了适配你的项目根目录运行
    # 请根据实际情况修改
    # 假设脚本在根目录下运行：
    data_path = os.path.join("data", "Q1data", "week_diagnostics.csv")

    plot_mechanism_comparison(data_path)