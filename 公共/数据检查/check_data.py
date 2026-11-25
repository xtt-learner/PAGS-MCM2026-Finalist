import pandas as pd
import os
import Q1_Fan_Estimation as Q1  # 确保 Q1_Fan_Estimation.py 在同级目录

# ---------------- 配置路径 ----------------
# 1. 待检查的投票样本文件 (Q1_Q3 输出的 CSV)
vs_path = "data/Q3data/vote_samples.csv"

# 2. 原始清洗后的宽表文件 (Reference Data)
current_dir = os.path.dirname(os.path.abspath(__file__))
cleaned_data_path = os.path.join(current_dir, "data", "processed_data", "cleaned_data.csv")


# ----------------------------------------

def check_all():
    print("=" * 60)
    print("   STARTING COMPREHENSIVE DATA CHECK")
    print("=" * 60)

    # --- 0. 加载数据 ---
    print(f"\n[Loading] Reading vote samples from: {vs_path} ...")
    if not os.path.exists(vs_path):
        print(f"❌ Error: File not found: {vs_path}")
        return

    vs = pd.read_csv(vs_path)
    print(f"   -> Loaded {len(vs)} rows.")

    # =========================================================
    # 维度 1: 数值范围检查 (Value Range Check)
    # =========================================================
    print(f"\n[Check 1/3] Value Range Check (v)")
    print("-" * 40)

    min_v = vs["v"].min()
    max_v = vs["v"].max()
    neg_count = (vs["v"] < 0).sum()
    gt1_count = (vs["v"] > 1).sum()

    print(f"   min v     = {min_v}")
    print(f"   max v     = {max_v}")
    print(f"   neg count = {neg_count} (Rows with v < 0)")
    print(f"   gt1 count = {gt1_count} (Rows with v > 1)")

    if neg_count == 0 and gt1_count == 0:
        print("   ✅ PASS: All vote proportions are within [0, 1].")
    else:
        print("   ❌ FAIL: Found invalid values!")

    # =========================================================
    # 维度 2: 每组人数检查 (Group Size Check)
    # =========================================================
    print(f"\n[Check 2/3] Contestant Count per Group")
    print("-" * 40)

    # 按 (赛季, 周, 第m个样本) 分组，统计有多少个唯一的 pair_id (选手)
    g = vs.groupby(["season", "week", "m"])["pair_id"].nunique()

    print("   Group size statistics:")
    print(g.describe())

    single_contestant_groups = (g == 1).sum()
    print(f"   Groups with only 1 contestant = {single_contestant_groups}")

    if single_contestant_groups > 0:
        print("   ⚠️ WARNING: Found groups with only 1 contestant (Automatic win?). Check logic.")
    else:
        print("   ✅ PASS: No single-contestant groups found.")

    # =========================================================
    # 维度 3: 一致性配对检查 (Consistency Check with Panel)
    # =========================================================
    print(f"\n[Check 3/3] Pair Consistency Check (vs. cleaned_data)")
    print("-" * 40)

    if not os.path.exists(cleaned_data_path):
        print(f"❌ Error: Reference file not found: {cleaned_data_path}")
        return

    # 3.1 准备 Reference Panel (从 cleaned_data 构建)
    print("   Building reference panel from cleaned_data...")
    df_clean = pd.read_csv(cleaned_data_path)
    weeks = Q1.detect_weeks(df_clean, col_suffix="_total")
    base = Q1.build_base_table(df_clean, weeks, score_template=Q1.SCORE_TEMPLATE)
    long_df = Q1.wide_to_long(base, weeks, score_template=Q1.SCORE_TEMPLATE)

    # 只取 active 的选手作为标准
    panel_active = long_df[long_df["active"] == True].copy()

    # 构造 pair_id (格式必须与 Q1_Q3 输出一致: "season||name||partner")
    panel_active["pair_id"] = (
            panel_active["season"].astype(int).astype(str) + "||" +
            panel_active["celebrity_name"].astype(str) + "||" +
            panel_active["ballroom_partner"].astype(str)
    )

    # 3.2 提取 Keys 进行对比
    keys_vs = vs[["season", "week", "pair_id"]].drop_duplicates()
    keys_panel = panel_active[["season", "week", "pair_id"]].drop_duplicates()

    print(f"   Unique keys in Vote Samples: {len(keys_vs)}")
    print(f"   Unique keys in Reference Panel: {len(keys_panel)}")

    # 3.3 执行 Merge
    miss = keys_vs.merge(keys_panel, on=["season", "week", "pair_id"], how="left", indicator=True)
    counts = miss["_merge"].value_counts()

    print("\n   Merge Results:")
    print(counts)

    if counts.get("left_only", 0) == 0:
        print("   ✅ PASS: All keys in vote_samples are valid (exist in reference).")
    else:
        print(f"   ❌ FAIL: Found {counts['left_only']} keys in vote_samples NOT in reference!")
        print("   Example mismatches:")
        print(miss[miss["_merge"] == "left_only"].head())

    # 反向检查: 是否有遗漏?
    reverse_miss = keys_panel.merge(keys_vs, on=["season", "week", "pair_id"], how="left", indicator=True)
    rev_counts = reverse_miss["_merge"].value_counts()

    if rev_counts.get("left_only", 0) == 0:
        print("   ✅ PASS: Full coverage (No missing contestants).")
    else:
        print(f"   ⚠️ NOTE: Reference has {rev_counts['left_only']} keys NOT in vote_samples.")
        print("   (This is normal ONLY IF you skipped certain weeks/seasons intentionally).")

    print("\n" + "=" * 60)
    print("   CHECK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    check_all()