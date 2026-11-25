import pandas as pd
import os


def clean_04_file():
    # 1. 文件路径配置
    # 请根据您实际的文件位置调整路径

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))


    file_path = os.path.join(project_root,"data", "Q3data", "04_mixed_judges_fixed.csv")
    New_file_path = os.path.join(project_root,"data", "Q3data", "04_mixed_judges_fixed(with out week and era).csv")


    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 错误：找不到文件 {file_path}")
        return

    print(f"正在读取文件: {file_path} ...")
    df = pd.read_csv(file_path)

    # 记录原始行数
    original_rows = len(df)

    # 2. 定义删除逻辑
    # 我们保留那些【不满足】删除条件的行
    # 条件1: 第一列以 "C(week)" 开头
    # 条件2: 第一列包含 ":C(week)" (用于匹配 interaction 项 C(era)...:C(week)...)

    # 假设第一列的列名是 'term' (根据您之前的CSV结构)
    # 如果第一列名字不同，请修改下面的 'term'
    col_name = df.columns[0]

    # 创建过滤器：保留即不以 C(week) 开头 且 不包含 :C(week)
    mask = (
            ~df[col_name].astype(str).str.startswith("C(week)") &
            ~df[col_name].astype(str).str.contains(":C(week)", regex=False)
    )

    df_cleaned = df[mask]

    # 3. 保存文件
    # 直接覆盖原文件 (如果您想保存为新文件，修改下面的文件名即可)
    df_cleaned.to_csv(New_file_path, index=False)

    # 4. 输出结果
    removed_rows = original_rows - len(df_cleaned)
    print(f"✅ 清洗完成！")
    print(f"   - 原始行数: {original_rows}")
    print(f"   - 删除行数: {removed_rows}")
    print(f"   - 剩余行数: {len(df_cleaned)}")
    print(f"   - 文件已保存至: {New_file_path}")


if __name__ == "__main__":
    clean_04_file()