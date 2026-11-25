# README 图表与结果来源

本页记录 README 展示材料的来源。两种语言版本共用相同图像；全部 PNG 为 LaTeX 工程中原图的逐字节副本，没有重绘、修改坐标或改写数据。

## 展示图像

| README 图像 | 原文件 | 论文位置 | 展示主题 |
| --- | --- | --- | --- |
| inverse-reconstruction.png | latex项目/figures/f1.png | 图 1 | Q1 反演流程 |
| posterior-diagnostics.png | latex项目/figures/Q1figure2.png | 图 2 | 后验预测一致性与不确定性 |
| rule-disagreement.png | latex项目/figures/Q2.4.png | 图 3 的实际图像 | 按赛季的 Rank–Percent 分歧概率 |
| risk-conflict-map.png | latex项目/figures/risk_conflict_map.png | 图 5 | 选手周的规则风险冲突 |
| channel-effects.png | latex项目/figures/Q3.1.png | 图 6(a) | 评委与粉丝通道效应 |
| partner-value-add.png | latex项目/figures/Q3.2.png | 图 6(b) | 舞伴技术与人气残差增益 |
| elimination-hazard.png | latex项目/figures/Q3.3.png | 图 7(a) | 淘汰风险回归 |
| feature-importance.png | latex项目/figures/Q3.4.png | 图 7(b) | 随机森林重要性 |

上述原文件相对于项目根目录。副本位于 [assets/readme](../assets/readme/)。逐项 SHA-256 记录在同目录的 [figure-manifest.json](../assets/readme/figure-manifest.json)。

## 数值证据

| README 结果 | LaTeX 来源 |
| --- | --- |
| PPC = 90.5% | 0.abstract.tex；2-1.part-2-1.tex 第 3.3 节 |
| 264 个淘汰周；P_diff = 0.260；总体 rho = 0.765 vs. 0.678 | 2-2.part-2-2.tex 第 4.4.1 节；6.memo.tex |
| 每周后验样本 M = 2000 | Q1 方法说明、Q3 多重插补说明、Q4 评估定义 |
| P/A/D：0.419/0.583/0.234 → 0.497/0.503/0.253 | 2-3.part-2-4.tex 表 5 |
| Bobby Bones S27W8：Percent ≈0.000；PAGS = 0.856 | 2-3.part-2-4.tex 的 eq:q4_stress_probs |
| Jerry Rice S2W3：PAGS = 0.247 | 同上 |
| 共享六个实质协变量的排序相关 rho_S = 0.657 | 稳健性.tex 第 7.2 节 |

以上来源均在 [latex项目](../latex项目/) 内，最终论文为 [论文.pdf](../论文.pdf)。

## 图注处理

`Q2.4.png` 实际图像的横轴是赛季，纵轴是 Rank–Percent 淘汰分歧概率，并标记 26% 总体均值。论文图 3 的文字图注却将它描述为 Percent/Rank 的相关系数分布。README 按实际图像内容说明，并将 0.765/0.678 的相关系数结果独立归于论文正文和备忘录，不将其描述为图中可见结果。

本次没有修改论文 PDF、LaTeX 源文件、原始插图或模型脚本；图注差异仅在展示文案中明确区分。稳健性章节的 S1/S2 图未用于首页展示。

## 论文与代码状态

本页所有结果均为论文报告值，未重新运行模型。此前比对显示，当前 LaTeX 工程在允许继续处理全角括号错误的模式下，生成的 26 页 PDF 与正式论文提取文本一致，96 DPI 页面渲染逐像素一致。该核验说明论文与源码相符，不等于模型代码与全部论文结果已完成复现。
