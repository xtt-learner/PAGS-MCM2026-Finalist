# 公共：数据预处理与跨题检查

预处理读取 data/original_data，写入 data/processed_data；check_data 检查 Q1→Q3 样本并依赖 Q1 模块。

此分区是原样代码副本，运行路径尚未适配；请先读根目录 README.md。

| 脚本 | 用途 | 原始相对位置 |
| --- | --- | --- |
| [数据检查/check_data.py](<数据检查/check_data.py>) | 检查 Q1→Q3 投票样本与清洗数据的一致性；依赖 Q1 模块。 | `check_data.py` |
| [数据预处理/pre-processingdata.py](<数据预处理/pre-processingdata.py>) | 原始竞赛数据预处理；含原项目绝对路径。 | `utils/pre-processingdata.py` |
