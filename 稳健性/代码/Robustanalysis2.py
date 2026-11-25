#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section 7 audit (Q4): Empirical gamma-stability from EXISTING outputs (no re-sim needed)

Reads:
- q4_week_summary.csv (already produced by your Q4 pipeline)

Writes:
- sec7_q4_gamma_empirical_bins.csv  (gamma-binned stability summary)
- sec7_q4_gamma_empirical_raw.csv   (raw subset with only needed columns)

Usage:
python sec7_q4_postprocess.py --q4_week_summary Q4结果/q4_week_summary.csv --out_dir sec7_outputs
"""
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q4_week_summary", required=True)
    ap.add_argument("--out_dir", default="sec7_outputs")
    ap.add_argument("--n_bins", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.q4_week_summary)

    # Keep only columns used in Sec7 plots/tables
    keep = [c for c in ["season","week","era","gamma","w","guardrail_on","P","A","D","ENT","rho","pad_score","lambda_damp"] if c in df.columns]
    raw = df[keep].copy()
    raw.to_csv(os.path.join(args.out_dir, "sec7_q4_gamma_empirical_raw.csv"), index=False, encoding="utf-8-sig")

    # Drop NaNs in gamma
    g = raw.dropna(subset=["gamma"]).copy()
    if len(g) == 0:
        raise ValueError("No valid gamma values found.")

    # Quantile bins so each bin has data
    n_bins = max(3, int(args.n_bins))
    g["gamma_bin"] = pd.qcut(g["gamma"], q=n_bins, duplicates="drop")
    agg_cols = ["guardrail_on","P","A","D"]
    for c in agg_cols:
        if c not in g.columns:
            raise ValueError(f"Column {c} not found in input.")

    out = g.groupby("gamma_bin").agg(
        n=("gamma","size"),
        gamma_min=("gamma","min"),
        gamma_max=("gamma","max"),
        gamma_mean=("gamma","mean"),
        w_mean=("w","mean") if "w" in g.columns else ("gamma","size"),
        guardrail_rate=("guardrail_on","mean"),
        P_mean=("P","mean"),
        A_mean=("A","mean"),
        D_mean=("D","mean"),
        pad_score_mean=("pad_score","mean") if "pad_score" in g.columns else ("P","mean"),
    ).reset_index(drop=True)

    out.to_csv(os.path.join(args.out_dir, "sec7_q4_gamma_empirical_bins.csv"), index=False, encoding="utf-8-sig")
    print("[OK] Wrote:")
    print(" - sec7_q4_gamma_empirical_raw.csv")
    print(" - sec7_q4_gamma_empirical_bins.csv")

if __name__ == "__main__":
    main()
