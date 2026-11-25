#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section 7 audit (Q1): Prior Sensitivity + Monte Carlo Convergence
---------------------------------------------------------------
Lightweight script that reuses Q1.py functions and re-runs ONLY ONE (season,week).

Outputs (CSV):
1) sec7_q1_prior_grid_summary.csv   # one row per beta
2) sec7_q1_mc_convergence.csv       # convergence vs sample size
3) (optional) sec7_q1_violin_samples.csv  # long-form posterior samples for that week (baseline beta)

Usage example:
python sec7_q1_audit.py --input cleaned_data.csv --season 27 --week 1 --target "Bobby" \
  --beta_grid "0,0.5,1,2,5" --n_samples 2000 --kappa_percent 300 --seed 123 --out_dir sec7_outputs
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Import your Q1 implementation (file must be named Q1.py or be importable as module)
import Q1_Fan_Estimation as Q1  # noqa: E402


@dataclass
class AuditCfg:
    beta: float
    alpha_rank: float
    n_samples: int
    kappa_percent: float
    seed: int
    max_attempts_percent: int = 200000
    max_attempts_rank: int = 300000
    use_slack_fallback: bool = True


def _make_infer_cfg(a: AuditCfg) -> Q1.InferenceConfig:
    return Q1.InferenceConfig(
        beta=a.beta,
        alpha_rank=a.alpha_rank,
        n_samples=a.n_samples,
        kappa_percent=a.kappa_percent,
        seed=a.seed,
        max_attempts_percent=a.max_attempts_percent,
        max_attempts_rank=a.max_attempts_rank,
        use_slack_fallback=a.use_slack_fallback,
    )


def infer_one_week(df_path: str, season: int, week: int, cfg: AuditCfg) -> dict:
    """Run Q1 inference for a single (season, week); return samples + key diagnostics."""
    df = pd.read_csv(df_path)
    weeks = Q1.detect_weeks(df, col_suffix="_total")
    base = Q1.build_base_table(df, weeks, score_template=Q1.SCORE_TEMPLATE)
    long_df = Q1.wide_to_long(base, weeks, score_template=Q1.SCORE_TEMPLATE)

    rng = np.random.default_rng(cfg.seed)

    grp = long_df[(long_df["season"] == season) & (long_df["week"] == week) & (long_df["active"])].copy()
    if len(grp) == 0:
        raise ValueError(f"No active rows found for season={season}, week={week}. Check your input CSV.")
    grp = grp.sort_values("contestant_id")

    cids = grp["contestant_id"].astype(int).tolist()
    names = grp["celebrity_name"].astype(str).tolist()

    J = grp["J"].astype(float).to_numpy()
    pJ = J / np.sum(J)

    # elimination set (based on "Eliminated Week t" only; Withdrew excluded)
    elim_map: Dict[Tuple[int, int], Set[int]] = {}
    for _, r in base.iterrows():
        s = int(r["season"])
        cid = int(r["contestant_id"])
        if bool(r["withdrew"]):
            continue
        ew = r["eliminated_week"]
        if pd.notna(ew):
            w = int(ew)
            elim_map.setdefault((s, w), set()).add(cid)

    elim_cids = elim_map.get((season, week), set())
    elim_idx = {cids.index(cid) for cid in elim_cids if cid in cids}

    mech, judges_save = Q1.mechanism_for_season(season)

    samples = np.zeros((0, len(J)))
    accept_rate = 0.0
    ppc_exact = np.nan
    ppc_relaxed = np.nan
    eps_hat = 0.0
    margin_mean = np.nan

    if mech == "percent":
        b = Q1.build_prior_b(pJ, beta=cfg.beta)
        v_hat, ok = Q1.solve_percent_map_no_slack(pJ, elim_idx, b)
        if (not ok) and cfg.use_slack_fallback:
            v_hat, eps_hat, _ = Q1.solve_percent_map_with_slack(pJ, elim_idx, b)
        samples, accept_rate = Q1.posterior_samples_percent(
            rng, pJ, elim_idx,
            v_center=v_hat,
            n_samples=cfg.n_samples,
            kappa=cfg.kappa_percent,
            max_attempts=cfg.max_attempts_percent,
        )

        if len(elim_idx) > 0 and len(samples) > 0:
            exact_hits = 0
            K = len(elim_idx)
            for v in samples:
                c = pJ + v
                worst = set(np.argsort(c)[:K])  # lowest K
                exact_hits += int(worst == set(elim_idx))
            ppc_exact = exact_hits / len(samples)

        if len(samples) > 0:
            margins = [Q1.percent_week_margin(pJ, v, elim_idx) for v in samples]
            margin_mean = float(np.nanmean(margins))

    else:  # rank
        samples, accept_rate, extra = Q1.posterior_samples_rank(
            rng, J, pJ, elim_idx,
            judges_save=judges_save,
            alpha=cfg.alpha_rank,
            beta=cfg.beta,
            n_samples=cfg.n_samples,
            max_attempts=cfg.max_attempts_rank,
        )
        if len(elim_idx) > 0 and len(samples) > 0:
            relaxed_hits = 0
            exact_hits = 0
            for v in samples:
                ok_relaxed, _ = Q1.accept_rank_sample(J, v, elim_idx, judges_save=judges_save, rng=rng)
                ok_exact, _ = Q1.accept_rank_sample(J, v, elim_idx, judges_save=False, rng=rng)
                relaxed_hits += int(ok_relaxed)
                exact_hits += int(ok_exact)
            ppc_relaxed = relaxed_hits / len(samples)
            ppc_exact = exact_hits / len(samples)

        if len(samples) > 0:
            margins = [Q1.rank_week_margin(J, v) for v in samples]
            margin_mean = float(np.nanmean(margins))

    summ = Q1.summarize_posterior(samples)

    return {
        "season": season,
        "week": week,
        "mechanism": mech,
        "judges_save": bool(judges_save),
        "cids": cids,
        "names": names,
        "J": J,
        "pJ": pJ,
        "elim_idx": sorted(list(elim_idx)),
        "samples": samples,
        "summ": summ,
        "accept_rate": float(accept_rate),
        "ppc_exact": float(ppc_exact) if not np.isnan(ppc_exact) else np.nan,
        "ppc_relaxed": float(ppc_relaxed) if not np.isnan(ppc_relaxed) else np.nan,
        "eps_hat": float(eps_hat),
        "margin_mean": float(margin_mean),
    }


def find_target_index(names: List[str], target: str) -> int:
    t = target.strip().lower()
    for i, nm in enumerate(names):
        if t in str(nm).lower():
            return i
    raise ValueError(f'Target "{target}" not found in celebrity_name list: {names}')


def export_violin(samples: np.ndarray, names: List[str], out_csv: str) -> None:
    rows = []
    for m in range(samples.shape[0]):
        for j, nm in enumerate(names):
            rows.append({"sample_id": m, "celebrity_name": nm, "vote_share": float(samples[m, j])})
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")


def mc_convergence(samples: np.ndarray, target_idx: int, M_list: List[int]) -> pd.DataFrame:
    full_mean = samples.mean(axis=0)
    rows = []
    for M in M_list:
        M = int(M)
        if M <= 0 or M > samples.shape[0]:
            continue
        pref = samples[:M]
        mean_pref = pref.mean(axis=0)
        l1 = float(np.sum(np.abs(mean_pref - full_mean)))
        tmean = float(mean_pref[target_idx])
        tci_low = float(np.quantile(pref[:, target_idx], 0.025))
        tci_high = float(np.quantile(pref[:, target_idx], 0.975))
        rows.append({
            "M": M,
            "L1_mean_to_full": l1,
            "target_mean": tmean,
            "target_ci_low": tci_low,
            "target_ci_high": tci_high,
            "target_ci_width": tci_high - tci_low,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Same preprocessed CSV used in Q1.")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--target", type=str, required=True, help='Substring match, e.g. "Bobby".')
    ap.add_argument("--beta_grid", type=str, default="0,0.5,1,2,5")
    ap.add_argument("--alpha_rank", type=float, default=1.0)
    ap.add_argument("--n_samples", type=int, default=2000)
    ap.add_argument("--kappa_percent", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out_dir", type=str, default="sec7_outputs")
    ap.add_argument("--export_violin", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    beta_list = [float(x.strip()) for x in args.beta_grid.split(",") if x.strip()]

    baseline_beta = beta_list[0] if len(beta_list) > 0 else 0.0

    # Run baseline first (used for MC convergence + L1 comparisons)
    base_cfg = AuditCfg(
        beta=baseline_beta,
        alpha_rank=args.alpha_rank,
        n_samples=args.n_samples,
        kappa_percent=args.kappa_percent,
        seed=args.seed,
    )
    base = infer_one_week(args.input, args.season, args.week, base_cfg)
    target_idx = find_target_index(base["names"], args.target)

    # MC convergence
    M_list = [50, 100, 200, 500, 800, 1000, 1500, 2000]
    conv = mc_convergence(base["samples"], target_idx, M_list)
    conv.to_csv(os.path.join(args.out_dir, "sec7_q1_mc_convergence.csv"), index=False, encoding="utf-8-sig")

    # Optional: violin samples (baseline)
    if args.export_violin:
        export_violin(base["samples"], base["names"], os.path.join(args.out_dir, "sec7_q1_violin_samples.csv"))

    # Prior grid summary
    base_mean = base["samples"].mean(axis=0) if len(base["samples"]) > 0 else None

    rows = []
    for beta in beta_list:
        cfg = AuditCfg(
            beta=float(beta),
            alpha_rank=args.alpha_rank,
            n_samples=args.n_samples,
            kappa_percent=args.kappa_percent,
            seed=args.seed,  # important: reset RNG per beta for fair comparison
        )
        out = infer_one_week(args.input, args.season, args.week, cfg)
        summ = out["summ"]
        if summ["mean"].size == 0:
            continue

        tmean = float(summ["mean"][target_idx])
        tci_low = float(summ["ci_low"][target_idx])
        tci_high = float(summ["ci_high"][target_idx])

        mean_vec = summ["mean"]
        l1_to_base = float(np.sum(np.abs(mean_vec - base_mean))) if base_mean is not None else np.nan

        # also track target's rank by posterior mean (1=highest vote share)
        order = np.argsort(-mean_vec)
        t_rank = int(np.where(order == target_idx)[0][0] + 1)

        rows.append({
            "season": args.season,
            "week": args.week,
            "mechanism": out["mechanism"],
            "beta": float(beta),
            "n_samples": int(out["samples"].shape[0]),
            "accept_rate": out["accept_rate"],
            "ppc_exact": out["ppc_exact"],
            "ppc_relaxed": out["ppc_relaxed"],
            "eps_hat": out["eps_hat"],
            "margin_mean": out["margin_mean"],
            "target_name": out["names"][target_idx],
            "target_vote_mean": tmean,
            "target_ci_low": tci_low,
            "target_ci_high": tci_high,
            "target_ci_width": tci_high - tci_low,
            "target_vote_rank": t_rank,
            "L1_mean_to_baseline": l1_to_base,
        })

    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "sec7_q1_prior_grid_summary.csv"),
                              index=False, encoding="utf-8-sig")

    print("[OK] Wrote:")
    print(" - sec7_q1_mc_convergence.csv")
    print(" - sec7_q1_prior_grid_summary.csv")
    if args.export_violin:
        print(" - sec7_q1_violin_samples.csv")


if __name__ == "__main__":
    main()
