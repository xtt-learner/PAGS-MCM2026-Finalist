# 代码归档与运行说明

本说明记录 2026-11-25 的代码分类归档，保留目录映射、数据路径与版本说明。项目随后加入正式论文 PDF、LaTeX 工程与论文展示首页。

- 原项目：`E:\PYthonCode\2026_MCM_Problem_C`
- 本副本：`E:\PYthonCode\2026_MCM_Problem_C_按题整理`
- 本次代码归档复制了 41 个项目 Python 文件，未复制 venv、缓存、IDE 配置、数据、图像或模型结果；当前项目另有后来加入的论文 PDF 和 LaTeX 工程。
- 41 个脚本按字节复制，保留原文件名；原项目文件未移动、未改写。

## 阅读入口

| 分区 | 内容 | Python 文件数 |
| --- | --- | --- |
| [Q1](<../Q1/README.md>) | 潜在粉丝投票反演 | 8 |
| [Q2](<../Q2/README.md>) | 投票聚合与淘汰规则比较 | 9 |
| [Q3](<../Q3/README.md>) | 舞伴与选手特征影响分析 | 7 |
| [Q4](<../Q4/README.md>) | 替代赛制与策略 | 8 |
| [稳健性](<../稳健性/README.md>) | 先验、采样与赛制稳定性 | 5 |
| [公共](<../公共/README.md>) | 数据预处理与跨题检查 | 2 |
| [其他](<../其他/README.md>) | 示例与关联待确认文件 | 2 |
| [Q5](../Q5/README.md) | 备忘录说明，无独立 Python 模型 | 0 |

每个分区的 README 列出用途和原始位置；[文件对应清单.csv](../文件对应清单.csv) 记录全部 41 个源位置、新位置、字节数、SHA-256 和运行提示。

## 运行与数据路径

**这是按题分类的源码归档，还不是迁移后可直接运行的独立工程。** 本次没有改写模型代码或适配路径，也没有重新运行模型。
脚本保留原项目的绝对路径、相对路径和基于 __file__ 的目录推算。移动到本副本后，部分目录推算不再对应原数据位置；Q3、Q4、稳健性及检查脚本还依赖 Q1 模块。
需要运行时，应先复核各脚本的数据路径、工作目录、输出位置和导入路径。仅设置工作目录或 PYTHONPATH 不能解决所有路径问题。现有原项目保留，可作为后续适配的参照。

| 原项目位置 | 内容 |
| --- | --- |
| data/original_data | 原始竞赛数据 |
| data/processed_data | 预处理结果、cleaned_data.csv |
| data/Q1data | Q1 后验汇总与诊断 |
| data/Q2data | Q2 规则比较结果 |
| data/Q3data | Q3 票样本、回归与风险模型结果 |
| data/Q4data | Q4 赛制仿真结果 |
| sec7_outputs | 稳健性分析结果 |
| images | 已有图像 |

以上位置均相对于原项目，而非本副本。约 542 MB 的 vote_samples.csv 保留在原项目，不重复复制；GitHub 的 Parquet 版本也未下载。

## 分类说明

- Q1_Q3.py 放入 Q3/数据准备，注明其依赖 Q1；Q4_process_data.py 放入 Q4/数据准备。
- Q3_NEW.py、Q3_Impact_Analysis.py、Strategy.py、Q4_Strategy.py 及各版绘图全部保留，不指定最终版本。
- Comparsion_figure.py 同时涉及 Q1/Q2，沿用其原有 Q1 分类并注明跨题用途。
- Uncertainty Ridgeline.py 和 The Age Scissors.py 使用手工示意数据/函数，不能直接当作真实拟合结果。
- main.py 是 IDE 示例；utils/data_check.py 内容为 UAV/SfM 流程图，放入其他/待确认。根目录 check_data.py 才是本项目样本检查脚本。

## 与论文仓库的关系

[GitHub 论文资料仓库](https://github.com/739542/2026-C-Finalist-)，既有比较基准提交为 `40c74fefaeafd9ac628c1033800a8c10d78c5ffc`。本次未更新远端或比较副本。
此前核验的 42 个结果文件内容一致属于既有比较结论，本次未重新核验。不能据此认定本地脚本就是生成最终论文的确切版本；Q1 诊断 CSV 等已知差异仍未完成数值等价核验。

本次只验证代码分类完整性、静态语法和复制哈希，不代表模型已复现、论文数值已验证或迁移后的运行兼容性已验证。

## 论文与展示资料

- [英文论文主页](../README.md)
- [中文论文主页](../README.zh-CN.md)
- [正式论文 PDF](../论文.pdf)
- [LaTeX 主文件](../latex项目/main.tex)
- [README 图表溯源](figure-provenance.md)
