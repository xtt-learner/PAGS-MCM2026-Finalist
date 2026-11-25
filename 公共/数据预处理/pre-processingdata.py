import pandas as pd
import numpy as np
import os
import re


class DWTSDataPreprocessor:
    def __init__(self, data_dir):
        """
        :param data_dir: 包含 csv 文件的文件夹路径
        """
        self.data_dir = data_dir
        # 自动寻找目录下以 .csv 结尾的文件
        self.filepath = self._find_csv_file()
        if self.filepath:
            print(f"正在读取文件: {self.filepath}")
            self.df = pd.read_csv(self.filepath)
        else:
            raise FileNotFoundError(f"在 {data_dir} 下没找到 .csv 文件！")

    def _find_csv_file(self):
        """辅助函数：自动查找 CSV"""
        for file in os.listdir(self.data_dir):
            if file.endswith(".csv"):
                return os.path.join(self.data_dir, file)
        return None

    def clean_column_names(self):
        """标准化列名：week1_judge1_score -> w1_j1"""
        new_columns = {}
        for col in self.df.columns:
            # 转小写，去空格
            clean = col.lower().strip().replace(' ', '_')
            clean = clean.replace('/', '_')  # 处理 homecountry/region

            # 正则匹配 C题格式: weekX_judgeY_score
            match = re.search(r'week(\d+)_judge(\d+)_score', clean)
            if match:
                w, j = match.groups()
                new_columns[col] = f"w{w}_j{j}"
            else:
                new_columns[col] = clean

        self.df.rename(columns=new_columns, inplace=True)
        print("✅ 列名清洗完成")

    def parse_results(self):
        """解析 Results 列，确定淘汰周"""

        def get_elimination_week(row):
            res = str(row.get('results', '')).lower()
            if 'eliminated' in res:
                # 提取数字: "eliminated week 5" -> 5
                match = re.search(r'week\s*(\d+)', res)
                return int(match.group(1)) if match else 99
            # 没被淘汰（冠军/亚军/季军），暂时设为一个大数
            if any(x in res for x in ['winner', 'runner', 'place', 'final']):
                return 100
            return np.nan  # 数据缺失

        self.df['week_eliminated'] = self.df.apply(get_elimination_week, axis=1)

        # 修正：将 100 替换为该赛季的实际最大周数
        for season in self.df['season'].unique():
            # 找到该赛季所有 wX_jY 列中的最大周数 X
            cols = [c for c in self.df.columns if re.match(r'w\d+_j\d+', c)]
            if not cols: continue
            max_week = max([int(c.split('_')[0][1:]) for c in cols])

            mask = (self.df['season'] == season) & (self.df['week_eliminated'] == 100)
            self.df.loc[mask, 'week_eliminated'] = max_week

        print("✅ 淘汰状态解析完成")

    def process_scores(self):
        """处理评委分数：计算平均分、总分，处理 N/A 和 淘汰0分"""
        # 找到所有评委分列
        judge_cols = [c for c in self.df.columns if re.match(r'w\d+_j\d+', c)]

        # 强制转为数字，非数字变为 NaN
        for col in judge_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # 提取所有周次
        weeks = sorted(list(set([int(c.split('_')[0][1:]) for c in judge_cols])))

        for w in weeks:
            prefix = f"w{w}_"
            # 动态获取本周评委列 (w1_j1, w1_j2...)
            current_cols = [c for c in judge_cols if c.startswith(prefix)]

            # 1. 计算平均分 (自动忽略 NaN)
            self.df[f'w{w}_avg'] = self.df[current_cols].mean(axis=1)

            # 2. 计算总分 (若全为 NaN 则保持 NaN)
            self.df[f'w{w}_total'] = self.df[current_cols].sum(axis=1, min_count=1)

            # 3. 淘汰处理：如果当前周 > 淘汰周，分数置 0
            # 注意：只有明确被淘汰后的周次才置0
            elim_mask = self.df['week_eliminated'] < w
            self.df.loc[elim_mask, f'w{w}_avg'] = 0
            self.df.loc[elim_mask, f'w{w}_total'] = 0

        print("✅ 分数统计计算完成")

    def run(self):
        self.clean_column_names()
        self.parse_results()
        self.process_scores()
        return self.df


# --- 执行部分 ---

if __name__ == "__main__":
    # 1. 设置路径 (注意 Windows 路径前面的 r)
    input_dir = r"E:\PYthonCode\2026_MCM_Problem_C\data\original_data"
    output_dir = r"E:\PYthonCode\2026_MCM_Problem_C\data\processed_data"

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # 2. 初始化并运行
        processor = DWTSDataPreprocessor(input_dir)
        cleaned_df = processor.run()

        # 3. 保存清洗后的数据
        output_path = os.path.join(output_dir, "cleaned_data.csv")
        cleaned_df.to_csv(output_path, index=False)

        print("\n" + "=" * 30)
        print(f"🎉 成功！清洗后的数据已保存至:\n{output_path}")
        print("=" * 30)

        # 4. 打印前5行看看效果
        print(cleaned_df[['celebrity_name', 'season', 'week_eliminated', 'w1_avg', 'w1_total']].head())

    except Exception as e:
        print(f"❌ 出错了: {e}")