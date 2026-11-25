# 稳健性：先验、采样与赛制稳定性

Q1 检验依赖 Q1_Fan_Estimation；Q4 检验读取 q4_week_summary.csv。已有结果在原项目 sec7_outputs。

此分区是原样代码副本，运行路径尚未适配；请先读根目录 README.md。

| 脚本 | 用途 | 原始相对位置 |
| --- | --- | --- |
| [代码/Robustanalysis1.py](<代码/Robustanalysis1.py>) | Q1 先验敏感性与 Monte Carlo 收敛检查；依赖 Q1 模块。 | `Robustanalysis1.py` |
| [代码/Robustanalysis2.py](<代码/Robustanalysis2.py>) | 读取已有 Q4 周汇总，计算 gamma 稳定性；无需重新仿真。 | `Robustanalysis2.py` |
| [绘图脚本/Montecarlo.py](<绘图脚本/Montecarlo.py>) | 绘制 Q1 Monte Carlo 收敛图；保留原有路径设置。 | `utils/Montecarlo.py` |
| [绘图脚本/Phase_Stability Region.py](<绘图脚本/Phase_Stability Region.py>) | 绘制 gamma 稳定性区域图。 | `utils/Phase_Stability Region.py` |
| [绘图脚本/PriorsensitivityBayesiandefense.py](<绘图脚本/PriorsensitivityBayesiandefense.py>) | 绘制 Q1 先验敏感性图。 | `utils/PriorsensitivityBayesiandefense.py` |
