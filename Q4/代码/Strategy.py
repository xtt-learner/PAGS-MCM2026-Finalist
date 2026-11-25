# q4_pags_adm.py
from __future__ import annotations

import os
import re
import json
import math
import pickle
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import pandas as pd


# =========================
# 0) Utilities
# =========================

def safe_std(x: np.ndarray, ddof: int = 1) -> float:
    """Return sample std; if degenerate, return 0."""
    s = float(np.std(x, ddof=ddof))
    return s

def normalize_simplex(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Clip at eps then renormalize to sum=1 along last axis."""
    v2 = np.maximum(v, eps)
    s = np.sum(v2, axis=-1, keepdims=True)
    return v2 / s

def bottom_m_indices_rowwise(X: np.ndarray, m: int) -> np.ndarray:
    """
    For each row of X (shape MxN), return indices of m smallest entries (shape Mxm).
    Uses argpartition for efficiency. Output order inside each row is not guaranteed sorted.
    """
    if m <= 0:
        return np.zeros((X.shape[0], 0), dtype=int)
    if m == 1:
        return np.argmin(X, axis=1).reshape(-1, 1)
    part = np.argpartition(X, kth=m-1, axis=1)[:, :m]
    return part

def indices_to_bitmask(idxs: np.ndarray) -> np.ndarray:
    """
    Convert indices array (Mxm) to uint64 bitmask (length M). Assumes N<=63.
    """
    if idxs.size == 0:
        return np.zeros((idxs.shape[0],), dtype=np.uint64)
    masks = np.zeros((idxs.shape[0],), dtype=np.uint64)
    for j in range(idxs.shape[1]):
        masks |= (np.uint64(1) << idxs[:, j].astype(np.uint64))
    return masks

def spearman_corr_from_scores(a: np.ndarray, b: np.ndarray) -> float:
    """
    Spearman rank correlation between two 1D arrays.
    Tie handling: average rank via pandas.
    """
    if len(a) < 2:
        return float("nan")
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


# =========================
# 1) Config
# =========================

@dataclass
class Q4Config:
    # cleaned_data.csv columns
    season_col: str = "season"
    week_col: str = "week"
    score_col: str = "wt_avg"              # J_{i,s,t}
    celeb_col: str = "celebrity_name"
    pro_col: str = "ballroom_partner"
    pair_id_col: str = "pair_id"           # if not exist, will be constructed

    results_col: Optional[str] = "results" # can be None if not available

    # Posterior file
    posterior_pkl_path: str = "posterior_votes.pkl"  # dict[(s,t)] -> {"ids":..., "samples":...}
    M: int = 2000

    # PAGS-ADM parameters
    tau: float = 2.0                # Z-score threshold
    kappa: float = 1.0              # exponential compression intensity
    gamma_min: float = 0.35         # lower bound on gamma_{s,t}
    eps_vote: float = 1e-12         # clip for votes

    # stage weights w_{s,t} (judge weight)
    w_min: float = 0.50
    w_max: float = 0.70
    finalists_default: int = 3      # n_min used if cannot infer

    # guardrail
    use_guardrail: bool = True
    guardrail_last_weeks: int = 3   # enable in last K weeks of each season
    guardrail_topK: int = 2         # protect Top-K by p(J)

    # output
    out_dir: str = "q4_outputs"

    # baselines for comparison (optional, but recommended)
    compute_baselines: bool = True


# =========================
# 2) Load and prepare data
# =========================

_ELIM_RE = re.compile(r"Eliminated\s*Week\s*(\d+)", flags=re.IGNORECASE)
_WITHDREW_RE = re.compile(r"Withdrew", flags=re.IGNORECASE)

def build_pair_id(df: pd.DataFrame, cfg: Q4Config) -> pd.DataFrame:
    if cfg.pair_id_col in df.columns:
        return df
    if cfg.celeb_col not in df.columns or cfg.pro_col not in df.columns:
        raise ValueError(f"Missing {cfg.pair_id_col} and cannot build from {cfg.celeb_col}/{cfg.pro_col}.")
    df = df.copy()
    df[cfg.pair_id_col] = df[cfg.celeb_col].astype(str).str.strip() + " | " + df[cfg.pro_col].astype(str).str.strip()
    return df

def parse_elim_week_and_withdrew(results: Any) -> Tuple[Optional[int], bool, bool]:
    """
    From results string, infer:
    - elim_week: integer if 'Eliminated Week X', else None
    - withdrew: True if 'Withdrew'
    - finalist: True if Winner/Runner-up/Finalist-like (no elimination event)
    """
    if results is None or (isinstance(results, float) and np.isnan(results)):
        return None, False, False
    s = str(results)
    withdrew = bool(_WITHDREW_RE.search(s))
    m = _ELIM_RE.search(s)
    elim_week = int(m.group(1)) if m else None
    finalist = bool(re.search(r"Winner|Runner|Final|Place", s, flags=re.IGNORECASE)) and (elim_week is None)
    return elim_week, withdrew, finalist

def infer_m_elim_by_week(df: pd.DataFrame, cfg: Q4Config) -> Dict[Tuple[int, int], int]:
    """
    Infer m_{s,t} = number of vote-based eliminations in week t of season s.
    Requires results_col; if missing, user must provide m_{s,t} externally.
    """
    if cfg.results_col is None or cfg.results_col not in df.columns:
        raise ValueError("results_col missing; cannot infer weekly elimination counts m_{s,t}.")
    meta = df[[cfg.season_col, cfg.pair_id_col, cfg.results_col]].drop_duplicates()
    counts: Dict[Tuple[int, int], int] = {}
    for _, row in meta.iterrows():
        s = int(row[cfg.season_col])
        elim_week, withdrew, finalist = parse_elim_week_and_withdrew(row[cfg.results_col])
        if elim_week is None:
            continue
        if withdrew or finalist:
            continue
        key = (s, int(elim_week))
        counts[key] = counts.get(key, 0) + 1
    return counts

def build_week_roster(df: pd.DataFrame, cfg: Q4Config) -> Dict[Tuple[int, int], List[str]]:
    """
    Build C_{s,t}: contestants active in week t of season s.
    Uses rule: wt_avg > 0 and not NaN.
    """
    d = df.copy()
    d = d[np.isfinite(d[cfg.score_col])]
    d = d[d[cfg.score_col] > 0]
    roster: Dict[Tuple[int, int], List[str]] = {}
    for (s, t), g in d.groupby([cfg.season_col, cfg.week_col], sort=True):
        ids = g[cfg.pair_id_col].astype(str).tolist()
        roster[(int(s), int(t))] = ids
    return roster

def build_judge_scores(df: pd.DataFrame, cfg: Q4Config) -> Dict[Tuple[int, int], pd.Series]:
    """
    Return mapping (s,t) -> Series indexed by pair_id giving J_{i,s,t} (wt_avg).
    Only for active contestants in that week (wt_avg>0).
    """
    d = df.copy()
    d = d[np.isfinite(d[cfg.score_col])]
    d = d[d[cfg.score_col] > 0]
    out: Dict[Tuple[int, int], pd.Series] = {}
    for (s, t), g in d.groupby([cfg.season_col, cfg.week_col], sort=True):
        ser = pd.Series(g[cfg.score_col].values, index=g[cfg.pair_id_col].astype(str).values)
        out[(int(s), int(t))] = ser
    return out

def infer_last_weeks_by_season(roster: Dict[Tuple[int, int], List[str]]) -> Dict[int, List[int]]:
    weeks_by_season: Dict[int, List[int]] = {}
    for (s, t) in roster.keys():
        weeks_by_season.setdefault(s, []).append(t)
    for s in weeks_by_season:
        weeks_by_season[s] = sorted(set(weeks_by_season[s]))
    return weeks_by_season

def infer_n_min_by_season(roster: Dict[Tuple[int, int], List[str]], finalists_default: int) -> Dict[int, int]:
    """
    Infer n_min (final size) as minimum roster size observed in season.
    Fallback to finalists_default if inference fails.
    """
    weeks_by_season = infer_last_weeks_by_season(roster)
    nmin: Dict[int, int] = {}
    for s, weeks in weeks_by_season.items():
        sizes = [len(roster[(s, t)]) for t in weeks]
        if len(sizes) == 0:
            nmin[s] = finalists_default
        else:
            nmin[s] = int(max(2, min(sizes)))  # at least 2
    return nmin


# =========================
# 3) Posterior loader
# =========================

class PosteriorVotes:
    """
    Loads Q1 posterior samples:
      posterior[(s,t)] = {"ids": [...], "samples": np.ndarray (M,n)}
    """
    def __init__(self, posterior_path: str, M: int):
        with open(posterior_path, "rb") as f:
            self.posterior = pickle.load(f)
        self.M = M

    def get_matrix(self, s: int, t: int, ids_order: List[str], eps: float) -> np.ndarray:
        key = (int(s), int(t))
        if key not in self.posterior:
            raise KeyError(f"Posterior missing for week {(s,t)}.")
        block = self.posterior[key]
        ids = list(map(str, block["ids"]))
        V = np.asarray(block["samples"], dtype=float)
        if V.shape[0] != self.M:
            raise ValueError(f"Posterior M mismatch at {(s,t)}: expected {self.M}, got {V.shape[0]}")
        # Reorder columns to match ids_order
        col_index = {pid: j for j, pid in enumerate(ids)}
        cols = []
        missing = []
        for pid in ids_order:
            if pid not in col_index:
                missing.append(pid)
            else:
                cols.append(col_index[pid])
        if missing:
            raise ValueError(f"Posterior ids missing in {(s,t)}: {missing[:5]} ... (total {len(missing)})")
        V = V[:, cols]
        V = normalize_simplex(V, eps=eps)
        return V


# =========================
# 4) Core PAGS-ADM rule
# =========================

def compute_week_weight(n_s: int, n_t: int, n_min: int, w_min: float, w_max: float) -> float:
    """
    w_{s,t} = w_min + (w_max-w_min) * (n_s - n_t)/(n_s - n_min)
    Monotone in n_t decreasing -> later weeks bigger weight on judges.
    """
    denom = (n_s - n_min)
    if denom <= 0:
        return float(w_max)
    frac = (n_s - n_t) / denom
    frac = min(1.0, max(0.0, frac))
    return float(w_min + (w_max - w_min) * frac)

def compute_gamma_from_tech_deviation(J: np.ndarray, tau: float, kappa: float, gamma_min: float) -> float:
    """
    Adaptive damping gamma_{s,t} in (gamma_min, 1]:
    - compute Z-score within week
    - d_i = max(0, -Z_i - tau)
    - d_max = max_i d_i
    - gamma = max(gamma_min, exp(-kappa * d_max))
    """
    mu = float(np.mean(J))
    sd = safe_std(J, ddof=1)
    if sd <= 1e-12:
        return 1.0
    Z = (J - mu) / sd
    d = np.maximum(0.0, -Z - tau)
    d_max = float(np.max(d))
    gamma = float(math.exp(-kappa * d_max))
    return float(max(gamma_min, min(1.0, gamma)))

def pags_elimination_for_week(
    J_series: pd.Series,
    V: np.ndarray,  # shape (M,n) aligned to ids
    m_elim: int,
    cfg: Q4Config,
    w_st: float,
    guardrail_on: bool,
) -> Dict[str, Any]:
    """
    Compute PAGS elimination for one week:
    Inputs:
      - J_series: index=pair_id, value=J_{i,s,t}
      - V: posterior samples matrix (M,n)
      - m_elim: number of eliminations that week
      - w_st: judge weight for that week
      - guardrail_on: whether to enable guardrail for this week
    Outputs:
      dict with elimination masks, probabilities, and per-sample auxiliary masks.
    """
    ids = list(map(str, J_series.index.tolist()))
    J = J_series.values.astype(float)
    n = len(ids)
    if n != V.shape[1]:
        raise ValueError("Roster size mismatch between J_series and V matrix.")

    # judge shares p(J)
    pJ = J / np.sum(J)

    # adaptive damping gamma_{s,t} (week-level, same for all samples)
    gamma = compute_gamma_from_tech_deviation(J, tau=cfg.tau, kappa=cfg.kappa, gamma_min=cfg.gamma_min)

    # compressed fan shares: v^gamma renormalized (gamma<1 compresses disparities)
    V_tilde = normalize_simplex(np.power(V, gamma), eps=cfg.eps_vote)

    # composite score S_{i,s,t}^{(m)} = w pJ + (1-w) v_tilde
    S = w_st * pJ.reshape(1, -1) + (1.0 - w_st) * V_tilde  # shape (M,n)

    # guardrail: protect Top-K by pJ from elimination (only in late weeks)
    if guardrail_on and cfg.use_guardrail and cfg.guardrail_topK > 0:
        K = min(cfg.guardrail_topK, n-1)
        protected = np.argsort(-pJ)[:K]  # highest pJ protected
        S_mod = S.copy()
        S_mod[:, protected] = np.inf
        elim_idx = bottom_m_indices_rowwise(S_mod, m_elim)
    else:
        elim_idx = bottom_m_indices_rowwise(S, m_elim)

    elim_mask = indices_to_bitmask(elim_idx)  # uint64 mask per sample

    # also compute judge-only and fan-only elimination masks (for P-A-D)
    judge_elim_idx = np.argsort(pJ)[:m_elim].reshape(1, -1).repeat(V.shape[0], axis=0)
    judge_mask = indices_to_bitmask(judge_elim_idx)

    fan_elim_idx = bottom_m_indices_rowwise(V, m_elim)
    fan_mask = indices_to_bitmask(fan_elim_idx)

    # elimination probabilities under PAGS
    counts = np.zeros(n, dtype=float)
    for j in range(m_elim):
        counts += np.bincount(elim_idx[:, j], minlength=n)
    p_elim = counts / V.shape[0]

    # P-A-D indicators
    P = float(np.mean(elim_mask == judge_mask))
    A = float(1.0 - P)
    D = float(np.mean((elim_mask != judge_mask) & (elim_mask != fan_mask)))

    # ENT (entropy of elimination probability distribution)
    p_safe = np.maximum(p_elim, 1e-15)
    ENT = float(-np.sum(p_safe * np.log(p_safe)))

    # Spearman between composite score (mean over samples) and mean fan shares
    S_mean = np.mean(S, axis=0)
    V_mean = np.mean(V, axis=0)
    rho_S_vs_V = spearman_corr_from_scores(-S_mean, -V_mean)  # higher score better -> use negative for ranks
    rho_S_vs_J = spearman_corr_from_scores(-S_mean, -pJ)

    return {
        "ids": ids,
        "n": n,
        "pJ": pJ,
        "gamma": gamma,
        "w": w_st,
        "elim_mask": elim_mask,
        "judge_mask": judge_mask,
        "fan_mask": fan_mask,
        "p_elim": p_elim,
        "P": P,
        "A": A,
        "D": D,
        "ENT": ENT,
        "rho_S_vs_V": rho_S_vs_V,
        "rho_S_vs_J": rho_S_vs_J,
    }


# =========================
# 5) Baseline rules (optional)
# =========================

def percent_elimination_for_week(J_series: pd.Series, V: np.ndarray, m_elim: int) -> np.ndarray:
    ids = list(map(str, J_series.index.tolist()))
    J = J_series.values.astype(float)
    pJ = J / np.sum(J)
    S = pJ.reshape(1, -1) + V
    elim_idx = bottom_m_indices_rowwise(S, m_elim)
    return indices_to_bitmask(elim_idx)

def rank_elimination_for_week(J_series: pd.Series, V: np.ndarray, m_elim: int) -> np.ndarray:
    """
    Rank rule: B = r(J)+r(F), eliminate largest B (worst).
    Implemented per sample via average ranks.
    """
    J = J_series.values.astype(float)
    n = len(J)
    # ranks of J are constant across samples
    rJ = pd.Series(-J).rank(method="average").to_numpy()  # 1 best
    masks = np.zeros((V.shape[0],), dtype=np.uint64)
    for m in range(V.shape[0]):
        rF = pd.Series(-V[m, :]).rank(method="average").to_numpy()
        B = rJ + rF
        # worst are largest B
        idx = np.argpartition(-B, kth=m_elim-1)[:m_elim]
        mask = np.uint64(0)
        for j in idx:
            mask |= (np.uint64(1) << np.uint64(j))
        masks[m] = mask
    return masks


# =========================
# 6) Main runner
# =========================

def run_q4_pags(cfg: Q4Config, cleaned_csv_path: str) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Load cleaned data
    df = pd.read_csv(cleaned_csv_path)
    df = build_pair_id(df, cfg)

    required = [cfg.season_col, cfg.week_col, cfg.score_col, cfg.pair_id_col]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"cleaned_data missing required column: {c}")

    # Build week roster C_{s,t} and judge scores J_{i,s,t}
    roster = build_week_roster(df, cfg)
    J_map = build_judge_scores(df, cfg)

    # Infer m_{s,t}
    m_elim_map = infer_m_elim_by_week(df, cfg)

    # infer week list by season and n_min
    weeks_by_season = infer_last_weeks_by_season(roster)
    n_min_by_season = infer_n_min_by_season(roster, cfg.finalists_default)

    # Load posterior
    post = PosteriorVotes(cfg.posterior_pkl_path, cfg.M)

    week_rows = []
    prob_rows = []

    # Optional: baselines
    baseline_rows = []  # store P-A-D for baselines (Percent/Rank) if enabled

    for (s, t), ids in sorted(roster.items()):
        key = (s, t)
        if key not in m_elim_map:
            # not an elimination week (or cannot infer); skip for Q4 evaluation
            continue
        m_elim = int(m_elim_map[key])
        if m_elim <= 0:
            continue
        if key not in J_map:
            continue

        J_series_full = J_map[key]
        # Align J_series to roster order
        J_series = J_series_full.reindex(ids)
        if J_series.isna().any():
            # Should not happen if roster built from score>0; but guard anyway
            J_series = J_series.dropna()
            ids = list(J_series.index.tolist())

        n_t = len(ids)
        if n_t <= m_elim:
            continue

        # Posterior vote samples aligned
        V = post.get_matrix(s, t, ids_order=ids, eps=cfg.eps_vote)

        # stage weight
        n_s = len(roster[(s, weeks_by_season[s][0])])
        n_min = n_min_by_season.get(s, cfg.finalists_default)
        w_st = compute_week_weight(n_s=n_s, n_t=n_t, n_min=n_min, w_min=cfg.w_min, w_max=cfg.w_max)

        # guardrail on? last K weeks of the season
        last_weeks = weeks_by_season[s][-cfg.guardrail_last_weeks:] if cfg.guardrail_last_weeks > 0 else []
        guardrail_on = (t in set(last_weeks))

        # Run PAGS
        out = pags_elimination_for_week(
            J_series=J_series,
            V=V,
            m_elim=m_elim,
            cfg=cfg,
            w_st=w_st,
            guardrail_on=guardrail_on,
        )

        # Store week summary
        week_rows.append({
            "season": s,
            "week": t,
            "n_contestants": out["n"],
            "m_elim": m_elim,
            "w": out["w"],
            "gamma": out["gamma"],
            "guardrail_on": int(guardrail_on and cfg.use_guardrail),
            "P": out["P"],
            "A": out["A"],
            "D": out["D"],
            "ENT": out["ENT"],
            "rho_S_vs_V": out["rho_S_vs_V"],
            "rho_S_vs_J": out["rho_S_vs_J"],
        })

        # Store contestant elimination probabilities
        for pid, pelim in zip(out["ids"], out["p_elim"]):
            prob_rows.append({
                "season": s,
                "week": t,
                "pair_id": pid,
                "p_elim_pags": float(pelim),
            })

        # Baselines (optional)
        if cfg.compute_baselines:
            judge_mask = out["judge_mask"]
            fan_mask = out["fan_mask"]

            pct_mask = percent_elimination_for_week(J_series, V, m_elim)
            P_pct = float(np.mean(pct_mask == judge_mask))
            A_pct = float(1.0 - P_pct)
            D_pct = float(np.mean((pct_mask != judge_mask) & (pct_mask != fan_mask)))

            rk_mask = rank_elimination_for_week(J_series, V, m_elim)
            P_rk = float(np.mean(rk_mask == judge_mask))
            A_rk = float(1.0 - P_rk)
            D_rk = float(np.mean((rk_mask != judge_mask) & (rk_mask != fan_mask)))

            baseline_rows.append({
                "season": s,
                "week": t,
                "P_pct": P_pct, "A_pct": A_pct, "D_pct": D_pct,
                "P_rank": P_rk, "A_rank": A_rk, "D_rank": D_rk,
            })

    # Export
    week_df = pd.DataFrame(week_rows).sort_values(["season", "week"])
    prob_df = pd.DataFrame(prob_rows).sort_values(["season", "week", "pair_id"])

    week_df.to_csv(os.path.join(cfg.out_dir, "q4_week_summary.csv"), index=False)
    prob_df.to_csv(os.path.join(cfg.out_dir, "q4_contestant_elim_prob.csv"), index=False)

    if cfg.compute_baselines and baseline_rows:
        base_df = pd.DataFrame(baseline_rows).sort_values(["season", "week"])
        base_df.to_csv(os.path.join(cfg.out_dir, "q4_week_baselines_PAD.csv"), index=False)

    # Also dump config for reproducibility
    with open(os.path.join(cfg.out_dir, "q4_run_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, ensure_ascii=False, indent=2)

    print(f"[OK] Exported:\n  - {cfg.out_dir}/q4_week_summary.csv\n  - {cfg.out_dir}/q4_contestant_elim_prob.csv")
    if cfg.compute_baselines:
        print(f"  - {cfg.out_dir}/q4_week_baselines_PAD.csv")
    print(f"  - {cfg.out_dir}/q4_run_config.json")


if __name__ == "__main__":
    # ====== USER EDITABLE ======
    CLEANED_CSV = "cleaned_data.csv"
    POSTERIOR_PKL = "posterior_votes.pkl"
    OUT_DIR = "q4_outputs"

    cfg = Q4Config(
        posterior_pkl_path=POSTERIOR_PKL,
        out_dir=OUT_DIR,

        # core parameters (you can grid-search these)
        tau=2.0,
        kappa=1.0,
        gamma_min=0.35,

        w_min=0.50,
        w_max=0.70,

        use_guardrail=True,
        guardrail_last_weeks=3,
        guardrail_topK=2,

        compute_baselines=True,
    )
    run_q4_pags(cfg, CLEANED_CSV)
