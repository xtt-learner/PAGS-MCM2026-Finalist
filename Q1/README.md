# Q1：潜在粉丝投票反演

主要输入是 data/processed_data/cleaned_data.csv；已有结果见 data/Q1data。Q1 后验被 Q2、Q3、Q4 及稳健性分析复用。

此分区是原样代码副本，运行路径尚未适配；请先读根目录 README.md。

| 脚本 | 用途 | 原始相对位置 |
| --- | --- | --- |
| [代码/Q1_Fan_Estimation.py](<代码/Q1_Fan_Estimation.py>) | 粉丝投票反演；受约束后验采样、汇总与诊断。 | `Q1_Fan_Estimation.py` |
| [绘图脚本/AcceptanceRateOverviewPlot.py](<绘图脚本/AcceptanceRateOverviewPlot.py>) | Q1 配套绘图或结果展示脚本；保留既有实现与原文件名。 | `utils/Q1/AcceptanceRateOverviewPlot.py` |
| [绘图脚本/Comparsion_figure.py](<绘图脚本/Comparsion_figure.py>) | 包含 Q1 秩反演和 Q2 分歧树的示意图；跨 Q1/Q2，沿用 Q1 分类。 | `utils/Q1/Comparsion_figure.py` |
| [绘图脚本/Figure_2.py](<绘图脚本/Figure_2.py>) | Q1 配套绘图或结果展示脚本；保留既有实现与原文件名。 | `utils/Q1/Figure_2.py` |
| [绘图脚本/plot_heatmap_overview.py](<绘图脚本/plot_heatmap_overview.py>) | Q1 配套绘图或结果展示脚本；保留既有实现与原文件名。 | `utils/Q1/plot_heatmap_overview.py` |
| [绘图脚本/plot_violin.py](<绘图脚本/plot_violin.py>) | Q1 配套绘图或结果展示脚本；保留既有实现与原文件名。 | `utils/Q1/plot_violin.py` |
| [绘图脚本/Q1_visualization.py](<绘图脚本/Q1_visualization.py>) | Q1 配套绘图或结果展示脚本；保留既有实现与原文件名。 | `utils/Q1/Q1_visualization.py` |
| [绘图脚本/Uncertainty Ridgeline.py](<绘图脚本/Uncertainty Ridgeline.py>) | 使用手工均值与标准差构造的示意脊线图，不是直接读取模型输出。 | `utils/Q1/Uncertainty Ridgeline.py` |
