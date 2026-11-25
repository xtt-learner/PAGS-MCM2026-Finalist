import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# ============================================================
# 0) Shared Utilities
# ============================================================

def average_rank_desc(values: pd.Series) -> pd.Series:
    """Descending average rank: higher is better; rank 1 is best."""
    return values.rank(ascending=False, method="average")


def spearman_from_ranks(r1: pd.Series, r2: pd.Series) -> float:
    """Spearman rho."""
    x = r1.to_numpy(dtype=float)
    y = r2.to_numpy(dtype=float)
    if np.isclose(x.std(), 0.0) or np.isclose(y.std(), 0.0):
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def fit_dirichlet_alpha0(means: np.ndarray, vars_: np.ndarray) -> float:
    eps = 1e-12
    mask = (vars_ > eps) & (means > eps) & (means < 1 - eps)
    if mask.sum() < 2:
        return 20.0
    cand = means[mask] * (1 - means[mask]) / vars_[mask] - 1.0
    cand = cand[np.isfinite(cand) & (cand > 0)]
    if cand.size == 0:
        return 20.0
    return float(np.median(cand))


def dirichlet_samples_from_mean_std(mean: np.ndarray, std: np.ndarray, M: int, rng: np.random.Generator) -> np.ndarray:
    var_ = std ** 2
    mean = np.clip(mean, 0.0, None)
    if mean.sum() <= 0:
        raise ValueError("vote_mean contains only non-positive values.")
    mean = mean / mean.sum()
    alpha0 = fit_dirichlet_alpha0(mean, var_)
    alpha = np.clip(mean * alpha0, 1e-3, None)
    return rng.dirichlet(alpha, size=M)


def melt_judges_from_cleaned(cleaned_path: str) -> pd.DataFrame:
    df = pd.read_csv(cleaned_path)
    avg_cols = [c for c in df.columns if re.match(r"w\d+_avg$", c)]
    if not avg_cols:
        raise ValueError("cleaned_data.csv: No columns matching 'wX_avg'.")
    long = df[["season", "celebrity_name"] + avg_cols].melt(
        id_vars=["season", "celebrity_name"],
        value_vars=avg_cols,
        var_name="week_col",
        value_name="wt_avg",
    )
    long["week"] = long["week_col"].str.extract(r"w(\d+)_avg").astype(int)
    return long.drop(columns=["week_col"]).dropna(subset=["wt_avg"]).copy()


# ============================================================
# PART A: Macro Metrics Calculation (Pdiff, FAI, Rho)
# ============================================================

def pct_elim_set(j: pd.Series, v: pd.Series, k: int) -> Tuple[List[int], pd.Series]:
    pJ = j / j.sum()
    c = pJ + v
    elim = list(c.sort_values(ascending=True).head(k).index)
    return elim, c


def rank_elim_set(j: pd.Series, v: pd.Series, k: int) -> Tuple[List[int], pd.Series, pd.Series, pd.Series]:
    rJ = average_rank_desc(j)
    rF = average_rank_desc(v)
    B = rJ + rF
    elim = list(B.sort_values(ascending=False).head(k).index)
    return elim, B, rJ, rF


def bottom2_from_pct(c: pd.Series) -> List[int]:
    return list(c.sort_values(ascending=True).head(2).index)


def bottom2_from_rank(B: pd.Series) -> List[int]:
    return list(B.sort_values(ascending=False).head(2).index)


def ims_elim_from_bottom2(j: pd.Series, bottom2: List[int]) -> int:
    return int(j.loc[bottom2].sort_values(ascending=True).index[0])


def compute_week_metrics_aggregate(
        season: int, week: int, k: int,
        j: pd.Series, mean: pd.Series, std: pd.Series,
        M: int, rng: np.random.Generator, ims_start_season: int = 28
) -> Dict:
    """Calculate weekly macro metrics (Pdiff, FAI, Rho)"""
    common = sorted(set(j.index) & set(mean.index) & set(std.index))
    if len(common) < 3:
        raise ValueError(f"Contestant intersection too small.")

    j = j.loc[common].astype(float)
    m = mean.loc[common].to_numpy(dtype=float)
    s = std.loc[common].to_numpy(dtype=float)

    V = dirichlet_samples_from_mean_std(m, s, M, rng)
    judge_bottom = int(j.sort_values(ascending=True).index[0])

    diff_flags = np.zeros(M, dtype=int)
    D_vals = np.zeros(M, dtype=float)
    fai_pct = np.zeros(M, dtype=float)
    fai_rank = np.zeros(M, dtype=float)
    rho_pct = np.zeros(M, dtype=float)
    rho_rank = np.zeros(M, dtype=float)

    ims_enabled = (season >= ims_start_season) and (k == 1)
    fai_pct_ims = np.full(M, np.nan)
    fai_rank_ims = np.full(M, np.nan)

    for idx in range(M):
        v = pd.Series(V[idx, :], index=common)
        fan_bottom = int(v.sort_values(ascending=True).index[0])

        e_pct, c = pct_elim_set(j, v, k)
        e_rank, B, _, rF = rank_elim_set(j, v, k)

        diff_flags[idx] = int(set(e_pct) != set(e_rank))
        symdiff = len(set(e_pct) ^ set(e_rank))
        D_vals[idx] = symdiff / max(k, 1)

        fai_pct[idx] = int(fan_bottom in e_pct) - int(judge_bottom in e_pct)
        fai_rank[idx] = int(fan_bottom in e_rank) - int(judge_bottom in e_rank)

        rPct = average_rank_desc(c)
        rho_pct[idx] = spearman_from_ranks(rPct, rF)
        rRank = average_rank_desc(-(B))
        rho_rank[idx] = spearman_from_ranks(rRank, rF)

        if ims_enabled:
            b2_pct = bottom2_from_pct(c)
            elim_pct_ims = ims_elim_from_bottom2(j, b2_pct)
            fai_pct_ims[idx] = int(fan_bottom == elim_pct_ims) - int(judge_bottom == elim_pct_ims)

            b2_rank = bottom2_from_rank(B)
            elim_rank_ims = ims_elim_from_bottom2(j, b2_rank)
            fai_rank_ims[idx] = int(fan_bottom == elim_rank_ims) - int(judge_bottom == elim_rank_ims)

    out = {
        "season": season, "week": week, "n_active": len(common), "n_elim": k, "M": M,
        "Pdiff": float(diff_flags.mean()), "Dexp": float(D_vals.mean()),
        "FAI_pct": float(np.nanmean(fai_pct)), "FAI_rank": float(np.nanmean(fai_rank)),
        "rho_pct": float(np.nanmean(rho_pct)), "rho_rank": float(np.nanmean(rho_rank)),
    }
    if ims_enabled:
        out.update({
            "FAI_pct_ims": float(np.nanmean(fai_pct_ims)),
            "FAI_rank_ims": float(np.nanmean(fai_rank_ims)),
            "dFAI_pct": float(np.nanmean(fai_pct_ims) - np.nanmean(fai_pct)),
            "dFAI_rank": float(np.nanmean(fai_rank_ims) - np.nanmean(fai_rank)),
        })
    return out


# ============================================================
# PART B: Micro Contestant Probability Calculation
# ============================================================

def ranks_desc_no_tie(v: np.ndarray) -> np.ndarray:
    M, n = v.shape
    order = np.argsort(-v, axis=1)
    ranks = np.empty_like(order, dtype=float)
    row_idx = np.arange(M)[:, None]
    ranks[row_idx, order] = np.arange(1, n + 1)[None, :]
    return ranks


def compute_week_contestant_probs(
        season: int, week: int, k: int,
        j: pd.Series, mean: pd.Series, std: pd.Series,
        M: int, rng: np.random.Generator,
        celebrity_map: Dict[int, str],
        apply_ims_counterfactual: bool = True
) -> pd.DataFrame:
    """Calculate contestant-level elimination probabilities"""
    common = sorted(set(j.index) & set(mean.index) & set(std.index))
    if len(common) < 3:
        return pd.DataFrame()

    j_vec = j.loc[common].to_numpy().astype(float)
    mean_vec = mean.loc[common].to_numpy().astype(float)
    std_vec = std.loc[common].to_numpy().astype(float)

    cid = np.array(common, dtype=int)
    n = len(common)
    V = dirichlet_samples_from_mean_std(mean_vec, std_vec, M, rng)

    # --- Percent ---
    pJ = j_vec / j_vec.sum()
    C = V + pJ[None, :]
    elim_pct_idx = np.argpartition(C, kth=k - 1, axis=1)[:, :k]
    b2_pct_idx = np.argpartition(C, kth=1, axis=1)[:, :2]

    # --- Rank ---
    orderJ = np.argsort(-j_vec)
    rankJ = np.empty(n, dtype=float)
    rankJ[orderJ] = np.arange(1, n + 1, dtype=float)
    rankF = ranks_desc_no_tie(V)
    B = rankF + rankJ[None, :]
    elim_rank_idx = np.argpartition(-B, kth=k - 1, axis=1)[:, :k]
    b2_rank_idx = np.argpartition(-B, kth=1, axis=1)[:, :2]

    # --- Counts ---
    counts = {
        "elim_pct": np.zeros(n), "elim_rank": np.zeros(n),
        "b2_pct": np.zeros(n), "b2_rank": np.zeros(n),
        "elim_pct_ims": np.zeros(n), "elim_rank_ims": np.zeros(n),
        "gap_sum": np.zeros(n),
    }

    np.add.at(counts["elim_pct"], elim_pct_idx.ravel(), 1.0)
    np.add.at(counts["elim_rank"], elim_rank_idx.ravel(), 1.0)
    np.add.at(counts["b2_pct"], b2_pct_idx.ravel(), 1.0)
    np.add.at(counts["b2_rank"], b2_rank_idx.ravel(), 1.0)
    counts["gap_sum"] = (rankJ[None, :] - rankF).sum(axis=0)

    # IMS Counterfactual (apply to ALL seasons if enabled, to see "what if")
    if apply_ims_counterfactual and k == 1:
        # Bottom 2 judge score tie-breaker
        # 1. Percent IMS
        j_b2_pct = j_vec[b2_pct_idx]
        pick_pct = np.argmin(j_b2_pct, axis=1)
        ims_pct_idx = b2_pct_idx[np.arange(M), pick_pct]
        np.add.at(counts["elim_pct_ims"], ims_pct_idx, 1.0)

        # 2. Rank IMS
        j_b2_rank = j_vec[b2_rank_idx]
        pick_rank = np.argmin(j_b2_rank, axis=1)
        ims_rank_idx = b2_rank_idx[np.arange(M), pick_rank]
        np.add.at(counts["elim_rank_ims"], ims_rank_idx, 1.0)
    else:
        counts["elim_pct_ims"][:] = np.nan
        counts["elim_rank_ims"][:] = np.nan

    out = pd.DataFrame({
        "season": season, "week": week, "n_elim": k,
        "contestant_id": cid,
        "celebrity_name": [celebrity_map.get(int(x), "") for x in cid],
        "p_elim_pct": counts["elim_pct"] / M,
        "p_elim_rank": counts["elim_rank"] / M,
        "p_b2_pct": counts["b2_pct"] / M,
        "p_b2_rank": counts["b2_rank"] / M,
        "E_rank_gap": counts["gap_sum"] / M,
        "p_elim_pct_ims": counts["elim_pct_ims"] / M,
        "p_elim_rank_ims": counts["elim_rank_ims"] / M,
    })
    return out


# ============================================================
# 5) MAIN INTEGRATION: Execution Logic
# ============================================================

def get_era(season):
    if season <= 2:
        return "Rank Era (S1-2)"
    elif season <= 27:
        return "Percent Era (S3-27)"
    else:
        return "Save Era (S28-34)"


def run_integrated_analysis(
        cleaned_data_path: str,
        summary_data_path: str,
        out_dir: str,
        M: int = 2000,
        seed: int = 42,
        ims_start_season: int = 28,  # For Aggregate Metrics (True History)
        ims_counterfactual: bool = True,  # For Contestant Metrics (What-if Analysis)
        controversy_list: List[Tuple[int, str]] = None
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Loading data...\n  Cleaned: {cleaned_data_path}\n  Summary: {summary_data_path}")

    judges_long = melt_judges_from_cleaned(cleaned_data_path)
    q1 = pd.read_csv(summary_data_path)

    # Merge contestant_id to judges
    key = q1[["season", "contestant_id", "celebrity_name"]].drop_duplicates()
    judges_long = judges_long.merge(key, on=["season", "celebrity_name"], how="inner")

    # Map for names
    celebs_map = key.set_index("contestant_id")["celebrity_name"].to_dict()

    rng = np.random.default_rng(seed)

    # Result Containers
    agg_rows = []  # For Outputs 1, 2, 3
    contestant_rows = []  # For Outputs 4, 5

    print(f"🔄 Starting simulation (M={M})...")
    grouped = q1.groupby(["season", "week"], sort=True)
    total = len(grouped)
    cnt = 0

    for (s, t), grp in grouped:
        cnt += 1
        k = int(grp["n_elim"].max())
        if k <= 0: continue

        j_df = judges_long[(judges_long["season"] == s) & (judges_long["week"] == t)]
        if j_df.empty: continue

        # Prepare Data Series
        j_series = j_df.set_index("contestant_id")["wt_avg"]
        m_series = grp.set_index("contestant_id")["vote_mean"]
        s_series = grp.set_index("contestant_id")["vote_std"]

        # --- Task A: Calculate Macro Metrics (Outputs 1-3) ---
        try:
            agg_met = compute_week_metrics_aggregate(
                s, t, k, j_series, m_series, s_series, M, rng, ims_start_season
            )
            agg_rows.append(agg_met)
        except Exception as e:
            print(f"❌ [Agg] Error S{s}W{t}: {e}")

        # --- Task B: Calculate Micro Contestant Probabilities (Outputs 4-5) ---
        try:
            cont_df = compute_week_contestant_probs(
                s, t, k, j_series, m_series, s_series, M, rng,
                celebs_map, ims_counterfactual
            )
            if not cont_df.empty:
                contestant_rows.append(cont_df)
        except Exception as e:
            print(f"❌ [Cont] Error S{s}W{t}: {e}")

        if cnt % 20 == 0:
            print(f"   Progress: {cnt}/{total} weeks...")

    # ==========================================
    # Save Outputs 1, 2, 3 (Macro)
    # ==========================================
    if agg_rows:
        df_agg = pd.DataFrame(agg_rows)
        df_agg['Era'] = df_agg['season'].apply(get_era)

        # Output 1: Detailed Metrics
        p1 = out_dir / "Q2_simulation_metrics.csv"
        df_agg.to_csv(p1, index=False)

        # Output 2: Season Summary
        season_agg = df_agg.drop(columns=["week"]).groupby(["season", "Era"]).mean(numeric_only=True).reset_index()
        p2 = out_dir / "Q2_season_summary.csv"
        season_agg.to_csv(p2, index=False)

        # Output 3: Statistics Summary (Cleaned)
        cols_excl = ["season", "week", "n_active", "n_elim", "M"]
        global_df = df_agg.drop(columns=cols_excl, errors='ignore')
        g_mean = global_df.mean(numeric_only=True).to_frame().T
        g_mean.insert(0, 'Category', 'Global_Mean')

        era_df = df_agg.drop(columns=cols_excl, errors='ignore')
        e_mean = era_df.groupby('Era').mean(numeric_only=True).reset_index().rename(columns={'Era': 'Category'})

        stats = pd.concat([g_mean, e_mean], ignore_index=True)
        # Reorder cols
        cols = ['Category'] + [c for c in stats.columns if c != 'Category']
        p3 = out_dir / "Q2_statistics_summary.csv"
        stats[cols].to_csv(p3, index=False)

        print(f"✅ Macro metrics generated: \n  - {p1.name}\n  - {p2.name}\n  - {p3.name}")

    # ==========================================
    # Save Outputs 4, 5 (Micro)
    # ==========================================
    if contestant_rows:
        df_cont = pd.concat(contestant_rows, ignore_index=True)

        # Output 4: Contestant Weekly Probabilities
        p4 = out_dir / "Q2_contestant_week_probs.csv"
        df_cont.to_csv(p4, index=False)

        # Output 5: Controversy Case Study
        if controversy_list is None:
            controversy_list = [
                (2, "Jerry Rice"), (4, "Billy Ray Cyrus"),
                (11, "Bristol Palin"), (27, "Bobby Bones")
            ]

        case_rows = []
        for s, name in controversy_list:
            sub = df_cont[(df_cont["season"] == s) & (df_cont["celebrity_name"] == name)].copy()
            if sub.empty: continue

            # Find the week where they benefited most from Rank (Rank elim < Pct elim)
            sub["benefit_from_rank"] = sub["p_elim_pct"] - sub["p_elim_rank"]
            idx = sub["benefit_from_rank"].idxmax()
            best_week = sub.loc[idx]

            case_rows.append({
                "season": s, "celebrity_name": name,
                "week": int(best_week["week"]),
                "p_elim_pct": float(best_week["p_elim_pct"]),
                "p_elim_rank": float(best_week["p_elim_rank"]),
                "benefit_prob": float(best_week["benefit_from_rank"]),
                "p_elim_rank_ims": float(best_week["p_elim_rank_ims"]),
                "avg_rank_gap": float(sub["E_rank_gap"].mean())
            })

        p5 = out_dir / "Q2_controversy_case_summary.csv"
        pd.DataFrame(case_rows).to_csv(p5, index=False)

        print(f"✅ Micro metrics generated: \n  - {p4.name}\n  - {p5.name}")


if __name__ == "__main__":
    # Configuration Paths
    path_cleaned = os.path.join("data", "processed_data", "cleaned_data.csv")
    path_summary = os.path.join("data", "Q1data", "contestant_posterior_summary.csv")
    path_output = os.path.join("data", "Q2data")

    run_integrated_analysis(
        cleaned_data_path=path_cleaned,
        summary_data_path=path_summary,
        out_dir=path_output,
        M=2000,
        seed=42
    )