import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
# 1. 读取数据
import os
import pandas as pd

# 1. 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 获取项目根目录 (根据你的代码，往上回退两级)
project_root = os.path.dirname(os.path.dirname(current_dir))

# 3. 定义数据文件夹的路径 (避免重复写路径)
# 假设数据都在 project_root/data/Q2data 下
data_dir = os.path.join(project_root, "data", "Q2data")

# 4. 构建每个 CSV 文件的完整绝对路径
path_season = os.path.join(data_dir, "Q2_season_summary.csv")
path_probs  = os.path.join(data_dir, "Q2_contestant_week_probs.csv")
path_cases  = os.path.join(data_dir, "Q2_controversy_case_summary.csv")

# --- 调试代码：建议加上这两行检查路径是否正确 ---
print(f"尝试读取路径: {path_probs}")
if not os.path.exists(path_probs):
    print("❌ 错误：找不到文件，请检查文件名或目录结构！")
# ---------------------------------------------

# 5. 正确读取 CSV
# df_season = pd.read_csv(path_season) # 如果你需要读取这个
df_probs = pd.read_csv(path_probs)
df_cases = pd.read_csv(path_cases)

print("✅ 数据读取成功！")
# 2. 数据预处理
# 为了图表清晰，我们只保留有一定淘汰风险的数据点 (任意一种规则下概率 > 1%)
plot_data = df_probs[(df_probs['p_elim_pct'] > 0.01) | (df_probs['p_elim_rank'] > 0.01)].copy()

# 3. 绘图设置
plt.figure(figsize=(10, 10))
sns.set_style("whitegrid")

# 定义颜色映射 (E_rank_gap: 正值越大 = 评委排名差但粉丝排名好 = 红色争议)
# 使用 diverging colormap (RdBu_r), 红色代表争议选手 (Fan Favorite), 蓝色代表评委宠儿
norm = plt.Normalize(df_probs['E_rank_gap'].min(), df_probs['E_rank_gap'].max())
sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
sm.set_array([])

# 4. 绘制主散点图
scatter = sns.scatterplot(
    data=plot_data,
    x='p_elim_rank',
    y='p_elim_pct',
    hue='E_rank_gap',
    palette="RdBu_r",
    size='E_rank_gap',
    sizes=(20, 200),
    alpha=0.7,
    edgecolor='w',
    linewidth=0.5,
    legend=False  # 我们手动添加 colorbar
)

# 5. 添加辅助线和区域
# 对角线 (一致性线)
plt.plot([0, 1], [0, 1], ls='--', c='gray', alpha=0.5, label='Perfect Agreement')

# 标注 "Rank Rule Safety Zone" (Rank规则下的安全区)
# 定义为: Percent规则下高风险 (>0.5), 但Rank规则下低风险 (<0.1)
from matplotlib.patches import Rectangle

rect = Rectangle((0, 0.5), 0.2, 0.5, linewidth=2, edgecolor='#D62728', facecolor='none', linestyle='--')
plt.gca().add_patch(rect)
plt.text(0.1, 0.95, 'Rank Rule\nSafety Zone\n(Structural Flaw)',
         color='#D62728', fontsize=12, fontweight='bold', ha='center', va='top')

# 6. 标注争议案例 (来自 Q2_controversy_case_summary.csv)
# 我们需要把争议案例的数据合并进来，或者直接用名字匹配
for _, row in df_cases.iterrows():
    # 在主数据中找到对应的点
    match = plot_data[
        (plot_data['season'] == row['season']) &
        (plot_data['week'] == row['week']) &
        (plot_data['celebrity_name'] == row['celebrity_name'])
        ]

    if not match.empty:
        x = match['p_elim_rank'].values[0]
        y = match['p_elim_pct'].values[0]
        label = f"{row['celebrity_name']}\n(S{int(row['season'])}W{int(row['week'])})"

        # 添加标注点和文字
        plt.scatter([x], [y], color='red', s=100, marker='x', linewidth=2, zorder=10)
        plt.text(x + 0.02, y, label, fontsize=9, fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='red', boxstyle='round,pad=0.2'))

# 7. 装饰图表
plt.title('Risk Conflict Map: Rank Rule vs. Percent Rule\n(Impact of Aggregation Logic on Elimination Risk)', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Elimination Probability under Rank Rule ($P_{elim}^{Rank}$)', fontsize=12)
plt.ylabel('Elimination Probability under Percent Rule ($P_{elim}^{Pct}$)', fontsize=12)
plt.xlim(-0.02, 1.02)
plt.ylim(-0.02, 1.02)

# --- 修复部分开始 ---
# 获取当前的坐标轴对象 (Axes)
ax = plt.gca()

# 添加 Colorbar，明确指定 ax 参数
# ax=ax 告诉 matplotlib 从当前的散点图旁边腾出空间来放 colorbar
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Judge-Fan Disagreement ($E[r^J - r^F]$)\nRed = Fan Favorite / Blue = Judge Favorite')
# --- 修复部分结束 ---

plt.tight_layout()
plt.show()

# 保存
plt.savefig('risk_conflict_map.png', dpi=300)
# 保存
# plt.savefig('risk_conflict_map.png', dpi=300)