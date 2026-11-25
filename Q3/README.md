# Q3：舞伴与选手特征影响分析

使用清洗数据和 Q1 后验票样本；已有结果见 data/Q3data。Q1_Q3.py 是样本导出桥接，Q3 两个候选主脚本均保留。

此分区是原样代码副本，运行路径尚未适配；请先读根目录 README.md。

| 脚本 | 用途 | 原始相对位置 |
| --- | --- | --- |
| [数据准备/Q1_Q3.py](<数据准备/Q1_Q3.py>) | 从 Q1 导出票样本供 Q3 使用；跨题桥接，依赖 Q1 模块。 | `Q1_Q3.py` |
| [代码/Q3_Impact_Analysis.py](<代码/Q3_Impact_Analysis.py>) | 舞伴与选手特征影响；混合效应、多重插补、hazard 和随机森林。 | `Q3_Impact_Analysis.py` |
| [代码/Q3_NEW.py](<代码/Q3_NEW.py>) | Q3 的另一候选版本；保留原名，未认定为最终版本。 | `Q3_NEW.py` |
| [绘图脚本/clean_04_file.py](<绘图脚本/clean_04_file.py>) | 清理 Q3 固定效应输出，生成绘图使用的派生 CSV。 | `utils/Q3/clean_04_file.py` |
| [绘图脚本/Pipeline.py](<绘图脚本/Pipeline.py>) | Q3 配套绘图或结果展示脚本；保留既有实现与原文件名。 | `utils/Q3/Pipeline.py` |
| [绘图脚本/The Age Scissors.py](<绘图脚本/The Age Scissors.py>) | 使用手工函数构造年龄剪刀差示意，不是直接读取拟合结果。 | `utils/Q3/The Age Scissors.py` |
| [绘图脚本/Threechannelfigure.py](<绘图脚本/Threechannelfigure.py>) | Q3 配套绘图或结果展示脚本；保留既有实现与原文件名。 | `utils/Q3/Threechannelfigure.py` |
