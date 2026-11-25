import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置绘图风格
sns.set_theme(style="white", context="talk")
# 解决中文显示 (Windows)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_data(data_dir):
    """读取 Q1 算出来的两个结果文件"""
    contestant_path = os.path.join(data_dir, "contestant_posterior_summary.csv")
    week_path = os.path.join(data_dir, "week_diagnostics.csv")

    if not os.path.exists(contestant_path):
        raise FileNotFoundError(f"找不到 {contestant_path}，请先运行 Q1 代码！")

    df_c = pd.read_csv(contestant_path)
    df_w = pd.read_csv(week_path)
    return df_c, df_w


def plot_heatmap_with_accept_rate(df_c, df_w, season_id):
    """
    绘制方案一：热力图 + 接受率折线图
    """
    # 1. 筛选特定赛季的数据
    data_c = df_c[df_c['season'] == season_id].copy()
    data_w = df_w[df_w['season'] == season_id].copy()

    if data_c.empty:
        print(f"Season {season_id} 没有数据！")
        return

    # 2. 准备热力图数据 (Pivot Table)
    # Index=选手名, Columns=周次, Values=得票率
    heatmap_data = data_c.pivot(index="celebrity_name", columns="week", values="vote_mean")

    # 按第一周的得票率排序，好看一点
    first_week = heatmap_data.columns.min()
    heatmap_data = heatmap_data.sort_values(by=first_week, ascending=False)

    # 3. 创建画布：上下两部分 (GridSpec)
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 5], hspace=0.15)

    ax_top = fig.add_subplot(gs[0])
    ax_bottom = fig.add_subplot(gs[1], sharex=ax_top)

    # --- 上图：Accept Rate 折线 ---
    sns.lineplot(data=data_w, x="week", y="accept_rate", marker="o", color="crimson", linewidth=2.5, ax=ax_top)
    ax_top.set_ylabel("接受率\n(Accept Rate)", fontsize=10)
    ax_top.set_title(f"Season {season_id}: 粉丝投票分布与模型置信度", fontsize=16, fontweight='bold')
    ax_top.grid(True, linestyle='--', alpha=0.5)

    # 标记出接受率极低的“争议时刻”
    low_accept_threshold = 0.01  # 阈值可调
    controversial_weeks = data_w[data_w['accept_rate'] < low_accept_threshold]
    for _, row in controversial_weeks.iterrows():
        ax_top.annotate(f"争议!\nW{int(row['week'])}",
                        xy=(row['week'], row['accept_rate']),
                        xytext=(row['week'], row['accept_rate'] + 0.1),
                        arrowprops=dict(facecolor='black', shrink=0.05),
                        fontsize=9, color='red')

    # --- 下图：Vote Mean 热力图 ---
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu",
                linewidths=.5, cbar_kws={'label': '估算粉丝得票率 (Vote Mean)'}, ax=ax_bottom)

    ax_bottom.set_xlabel("周次 (Week)")
    ax_bottom.set_ylabel("选手 (Celebrity)")

    # 隐藏上图的 X 轴刻度，美观
    plt.setp(ax_top.get_xticklabels(), visible=False)


    plt.show()


def plot_trend_with_ci(df_c, season_id, top_n=5):
    """
    绘制方案二：带误差带的趋势图 (Ribbon Plot)
    """
    data = df_c[df_c['season'] == season_id].copy()

    # 选出该赛季存活最久或者平均票数最高的 Top N 选手画图，不然太乱
    top_players = data.groupby('celebrity_name')['vote_mean'].mean().nlargest(top_n).index
    plot_data = data[data['celebrity_name'].isin(top_players)]

    plt.figure(figsize=(12, 6))

    # 循环画每个选手
    colors = sns.color_palette("tab10", n_colors=top_n)
    for i, player in enumerate(top_players):
        player_data = plot_data[plot_data['celebrity_name'] == player]

        # 画线
        plt.plot(player_data['week'], player_data['vote_mean'], label=player, color=colors[i], marker='.')

        # 画误差带 (CI Low ~ CI High)
        plt.fill_between(player_data['week'],
                         player_data['ci_low'],
                         player_data['ci_high'],
                         color=colors[i], alpha=0.2)  # alpha 是透明度

    plt.title(f"Season {season_id}: Top {top_n} 热门选手得票率走势 (带 95% 置信区间)", fontsize=14)
    plt.xlabel("周次")
    plt.ylabel("粉丝得票率")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 你的数据目录
    data_dir = r"/data/Q1data"

    try:
        df_c, df_w = load_data(data_dir)

        # 画 Season 1 (你可以改成任意你想看的赛季)
        plot_heatmap_with_accept_rate(df_c, df_w, season_id=1)

        # 画趋势图
        plot_trend_with_ci(df_c, season_id=1, top_n=4)

        print("✅ 图表绘制完成！")
    except Exception as e:
        print(f"❌ 错误: {e}")