#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export only vote posterior samples needed by Q3.
Final CSV Version: Guaranteed to output standard text CSV.
"""

import argparse
import os
import numpy as np
import pandas as pd

# -----------------------------------------------------------
# Ensure Q1_Fan_Estimation.py is in the same folder
# -----------------------------------------------------------
import Q1_Fan_Estimation as Q1
from Q1_Fan_Estimation import (
    SCORE_TEMPLATE,
    DEFAULT_JUDGES_SAVE_FROM_SEASON,
    InferenceConfig,
    build_prior_b,
    solve_percent_map_no_slack,
    solve_percent_map_with_slack,
    posterior_samples_percent,
    posterior_samples_rank,
)


def export_vote_samples_to_csv(
        df_path: str,
        out_csv: str,
        cfg: InferenceConfig,
        judges_save_from_season: int = DEFAULT_JUDGES_SAVE_FROM_SEASON,
        strict_M: bool = True,
) -> None:
    print(f"Loading data from {df_path}...")

    # 1) Read Data
    df = pd.read_csv(df_path)

    # 2) Detect weeks
    weeks = Q1.detect_weeks(df, col_suffix="_total")
    print(f"Detected {len(weeks)} weeks in dataset.")

    # 3) Build base tables
    base = Q1.build_base_table(df, weeks, score_template=SCORE_TEMPLATE)
    long_df = Q1.wide_to_long(base, weeks, score_template=SCORE_TEMPLATE)

    rng = np.random.default_rng(cfg.seed)

    # 4) Build elimination map
    elim_map = {}
    for _, r in base.iterrows():
        s = int(r["season"])
        cid = int(r["contestant_id"])
        if bool(r["withdrew"]):
            continue
        ew = r["eliminated_week"]
        if pd.notna(ew):
            t = int(ew)
            elim_map.setdefault((s, t), set()).add(cid)

    # 5) Group by Season/Week
    grouped = long_df[long_df["active"]].groupby(["season", "week"])

    # Prepare output directory
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Delete existing file to ensure a fresh start
    if os.path.exists(out_csv):
        try:
            os.remove(out_csv)
            print(f"Removed existing file: {out_csv}")
        except OSError:
            pass

    total_batches = 0
    is_first_batch = True

    print(f"Starting sampling and export loop to {out_csv}...")

    for (s_raw, t_raw), grp in grouped:
        s = int(s_raw)
        t = int(t_raw)

        mech, judges_save = Q1.mechanism_for_season(s, judges_save_from=judges_save_from_season)

        grp = grp.sort_values("contestant_id")
        cids = grp["contestant_id"].astype(int).tolist()
        names = grp["celebrity_name"].astype(str).tolist()
        pros = grp["ballroom_partner"].astype(str).tolist()

        J = grp["J"].astype(float).to_numpy()
        J_sum = np.sum(J)
        if J_sum == 0:
            pJ = np.ones_like(J) / len(J)
        else:
            pJ = J / J_sum

        elim_cids = elim_map.get((s, t), set())
        elim_idx = {cids.index(cid) for cid in elim_cids if cid in cids}

        # ---- Bayesian Sampling ----
        if mech == "percent":
            b = build_prior_b(pJ, beta=cfg.beta)
            v_hat, ok = solve_percent_map_no_slack(pJ, elim_idx, b)
            if (not ok) and cfg.use_slack_fallback:
                v_hat, _, _ = solve_percent_map_with_slack(pJ, elim_idx, b)

            samples, _ = posterior_samples_percent(
                rng, pJ, elim_idx,
                v_center=v_hat,
                n_samples=cfg.n_samples,
                kappa=cfg.kappa_percent,
                max_attempts=cfg.max_attempts_percent
            )
        else:
            samples, _, _ = posterior_samples_rank(
                rng, J, pJ, elim_idx,
                judges_save=judges_save,
                alpha=cfg.alpha_rank,
                beta=cfg.beta,
                n_samples=cfg.n_samples,
                max_attempts=cfg.max_attempts_rank
            )

        if strict_M and samples.shape[0] != cfg.n_samples:
            print(f"Warning: Season {s} Week {t} returned {samples.shape[0]} samples (Expected {cfg.n_samples}).")
            if strict_M:
                raise RuntimeError("Strict M check failed.")

        # ---- Build Batch Data ----
        M = samples.shape[0]
        n = samples.shape[1]

        m_vec = np.repeat(np.arange(1, M + 1), n)
        cid_vec = np.tile(np.array(cids, dtype=int), M)
        name_vec = np.tile(np.array(names, dtype=object), M)
        pro_vec = np.tile(np.array(pros, dtype=object), M)

        pair_ids = [f"{s}||{names[j]}||{pros[j]}" for j in range(n)]
        pair_vec = np.tile(np.array(pair_ids, dtype=object), M)

        v_vec = samples.reshape(-1)

        batch = pd.DataFrame({
            "season": s,
            "week": t,
            "m": m_vec,
            "contestant_id": cid_vec,
            "celebrity_name": name_vec,
            "ballroom_partner": pro_vec,
            "pair_id": pair_vec,
            "v": v_vec
        })

        # ---- Write to CSV ----
        # Using standard Pandas to_csv with append mode
        batch.to_csv(out_csv, mode='a', header=is_first_batch, index=False)

        is_first_batch = False
        total_batches += 1

        if t == 1:
            print(f"Processed Season {s}...")

    print(f"[OK] Successfully exported {total_batches} batches to: {out_csv}")


def main():
    ap = argparse.ArgumentParser()

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # -------------------------------------------------------------
    # Input Path Config
    # -------------------------------------------------------------
    default_csv_path = os.path.join(current_dir, "data", "processed_data", "cleaned_data.csv")

    ap.add_argument("--input", default=default_csv_path, help="Path to cleaned_data.csv")
    ap.add_argument("--out", default="vote_samples.csv", help="Output csv path")

    ap.add_argument("--n_samples", type=int, default=2000, help="M samples per (season, week)")
    ap.add_argument("--seed", type=int, default=123, help="Random seed")
    ap.add_argument("--beta", type=float, default=0.0, help="Prior beta")
    ap.add_argument("--alpha_rank", type=float, default=1.0, help="Dirichlet scale")
    ap.add_argument("--kappa_percent", type=float, default=300.0, help="Proposal concentration")
    ap.add_argument("--judges_save_from", type=int, default=DEFAULT_JUDGES_SAVE_FROM_SEASON)
    ap.add_argument("--strict_M", action="store_true", help="Raise if samples < M")

    args = ap.parse_args()

    # Check input existence
    if not os.path.exists(args.input):
        print(f"Error: Input file not found at: {args.input}")
        return

    cfg = InferenceConfig(
        beta=args.beta,
        alpha_rank=args.alpha_rank,
        n_samples=args.n_samples,
        kappa_percent=args.kappa_percent,
        seed=args.seed,
    )

    try:
        export_vote_samples_to_csv(
            df_path=args.input,
            out_csv=args.out,
            cfg=cfg,
            judges_save_from_season=args.judges_save_from,
            strict_M=args.strict_M
        )
    except Exception as e:
        print(f"❌ Error occurred: {e}")


if __name__ == "__main__":
    main()