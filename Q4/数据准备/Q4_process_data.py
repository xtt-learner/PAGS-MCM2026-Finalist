# Q1-Q4补充.py
# -*- coding: utf-8 -*-

"""
唯一目标：
基于 Q1.py 的拒绝采样逻辑，生成 Q4 可直接使用的 posterior_votes.pkl

- 输入：Q1 宽表 cleaned_data.csv
- 输出：posterior_votes.pkl
- pair_id 口径：严格采用 Q3 的 season||celebrity||ballroom_partner
"""
from __future__ import annotations
import os

import argparse
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Set

# -----------------------------
# 1) 导入 Q1 模块
# -----------------------------
import Q1_Fan_Estimation as Q1  # 假设文件名就是 Q1.py


# -----------------------------
# 2) Q3 口径的 pair_id 构造
# -----------------------------
def make_pair_id(season: int, celeb: str, partner: str) -> str:
    return (
        f"{int(season)}||"
        f"{str(celeb).strip()}||"
        f"{str(partner).strip()}"
    )


# -----------------------------
# 3) 主流程
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    current_dir = os.path.dirname(os.path.abspath(__file__))

    project_root = os.path.dirname(os.path.dirname(current_dir))

    # 拼接默认路径
    default_csv_path = os.path.join(project_root, "data", "processed_data", "cleaned_data.csv")

    # 确保输出目录存在
    output_dir = os.path.join(project_root, "data", "Q4data")
    default_pkl_path = os.path.join(output_dir, "posterior_votes.pkl")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    parser.add_argument("--input_csv", default=default_csv_path, help="Q1 宽表 cleaned_data.csv")
    parser.add_argument("--out_pkl", default= default_pkl_path)
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20240229)
    args = parser.parse_args()

    # 读取 Q1 宽表
    df_wide = pd.read_csv(args.input_csv)

    # === Q1 标准流程 ===
    weeks = Q1.detect_weeks(df_wide, col_suffix="_total")
    base = Q1.build_base_table(df_wide, weeks, Q1.SCORE_TEMPLATE)
    long_df = Q1.wide_to_long(base, weeks, Q1.SCORE_TEMPLATE)

    # 构造 (season, week) -> eliminated contestant_id
    elim_map: Dict[Tuple[int, int], Set[int]] = {}
    for _, r in base.iterrows():
        s = int(r["season"])
        cid = int(r["contestant_id"])
        if r["withdrew"]:
            continue
        ew = r["eliminated_week"]
        if pd.notna(ew):
            elim_map.setdefault((s, int(ew)), set()).add(cid)

    rng = np.random.default_rng(args.seed)

    posterior: Dict[Tuple[int, int], Dict[str, object]] = {}

    grouped = long_df[long_df["active"]].groupby(["season", "week"], sort=True)

    for (s_raw, t_raw), grp in grouped:
        s, t = int(s_raw), int(t_raw)

        mech, judges_save = Q1.mechanism_for_season(s)

        grp = grp.sort_values("contestant_id").reset_index(drop=True)
        cids = grp["contestant_id"].tolist()

        # === Q3 口径 pair_id ===
        ids = [
            make_pair_id(s, nm, pt)
            for nm, pt in zip(grp["celebrity_name"], grp["ballroom_partner"])
        ]

        J = grp["J"].astype(float).to_numpy()
        pJ = J / J.sum()

        elim_cids = elim_map.get((s, t), set())
        elim_idx = {cids.index(cid) for cid in elim_cids if cid in cids}

        # === 拒绝采样 ===
        if mech == "percent":
            b = Q1.build_prior_b(pJ, beta=Q1.InferenceConfig.beta)
            v_hat, ok = Q1.solve_percent_map_no_slack(pJ, elim_idx, b)
            if not ok:
                v_hat, _, ok = Q1.solve_percent_map_with_slack(pJ, elim_idx, b)
            samples, _ = Q1.posterior_samples_percent(
                rng,
                pJ,
                elim_idx,
                v_center=v_hat,
                n_samples=args.n_samples,
                kappa=Q1.InferenceConfig.kappa_percent,
                max_attempts=Q1.InferenceConfig.max_attempts_percent,
            )
        else:
            samples, _, _ = Q1.posterior_samples_rank(
                rng,
                judge_scores=J,
                pJ=pJ,
                elim_set=elim_idx,
                judges_save=judges_save,
                alpha=Q1.InferenceConfig.alpha_rank,
                beta=Q1.InferenceConfig.beta,
                n_samples=args.n_samples,
                max_attempts=Q1.InferenceConfig.max_attempts_rank,
            )

        n_avail = samples.shape[0]

        if n_avail < args.n_samples:
            # --- 1. 风险监控 ---
            missing_ratio = 1.0 - (n_avail / args.n_samples)
            print(f"⚠️ [Warning] (Season={s}, Week={t}) 样本不足。")
            print(f"   - 实际可用: {n_avail}")
            print(f"   - 目标数量: {args.n_samples}")
            print(f"   - 缺失比例: {missing_ratio:.1%}")

            # 严重风险阈值 (例如少于目标量的 25% 或 500个)
            if n_avail < 500:
                print(f"   ❌ [CRITICAL] 有效样本极少 (<500)，该周次的分布可能不准确！请检查 Q1 参数。")
            else:
                print(f"   -> 启用 Bootstrap (replace=True) 补齐样本...")

            # --- 2. 重复采样 (Bootstrap) ---
            # 从 0 到 n_avail-1 中，有放回地抽取 args.n_samples 个索引
            # 注意：rng 是您代码前面定义的 np.random.default_rng
            indices = rng.choice(n_avail, size=args.n_samples, replace=True)
            samples = samples[indices]

        elif n_avail > args.n_samples:
            # 如果 Q1 返回的样本比要求的多，随机截取 (无放回)
            indices = rng.choice(n_avail, size=args.n_samples, replace=False)
            samples = samples[indices]




        samples = np.maximum(samples, 1e-12)
        samples = samples / samples.sum(axis=1, keepdims=True)

        posterior[(s, t)] = {
            "ids": ids,
            "samples": samples.astype(np.float32),
        }

    # === 输出 ===
    with open(args.out_pkl, "wb") as f:
        pickle.dump(posterior, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[OK] posterior_votes.pkl saved to: {args.out_pkl}")
    print(f"[INFO] total (season, week) blocks: {len(posterior)}")


if __name__ == "__main__":
    main()
