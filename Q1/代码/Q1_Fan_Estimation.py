

"""
Q1：DWTS 粉丝投票反向估计（2026 MCM Problem C）

【给负责编程的同学看的快速说明】
----------------------------------------------------------------------
目标：在已知每周评委得分与历史淘汰结果的情况下，反推每位选手每周的“粉丝投票占比”分布。
- 输出不是唯一投票值，而是“后验样本”→ 均值/中位数 + 置信区间 + 标准差(Std Dev)。
- 规则分两类（按题目附录）：
  1) Percent 合成（假设 Season 3–27）：c_i = judge_percent_i + fan_percent_i，最低者淘汰。
  2) Rank 合成（假设 Season 1–2、28–34）：B_i = judge_rank_i + fan_rank_i，最差者淘汰。
     对 Season 28+ 额外加入 Judges' Save：只要求“被淘汰者在 bottom-2 之内”，不要求必须最差。
----------------------------------------------------------------------
输入：预处理后的 CSV（宽表），需要包含：
- season, results, placement, celebrity_name, ballroom_partner 等基础列
- 每周评委总分列：w1_total ... wW_total（W 由文件自动识别）
  （如果你们用的是 w*_avg 也可以改 SCORE_TEMPLATE）
----------------------------------------------------------------------
输出（默认输出两个 CSV）：
1) contestant_posterior_summary.csv
   每行：season, week, contestant_id, celebrity_name, vote_mean, vote_std, ci_low, ci_high, ...
2) week_diagnostics.csv
   每行：season, week, n_active, n_elim, mechanism, judges_save, accept_rate, ppc_exact, ppc_relaxed, margin_mean ...

可选：保存某些(赛季,周)的后验样本用于画小提琴图（violin plot）。
"""

from __future__ import annotations
import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# =========================
# 0) 配置区（可按需修改）
# =========================

# 默认使用总分列作为评委分数（匹配题目附录里的 Total Judges Score）
SCORE_TEMPLATE = "w{t}_total"

# 赛季机制假设（题目说明：3-27 用 percent；1-2 以及大约 28+ 用 rank；28+ 引入 judges save）
PERCENT_SEASONS = set(range(3, 28))         # 3..27
RANK_SEASONS = set([1, 2]) | set(range(28, 35))  # 1,2,28..34

DEFAULT_JUDGES_SAVE_FROM_SEASON = 28  # Season >= 28 视作 judges save 生效


# =========================
# 1) 工具函数：解析与整理
# =========================

def detect_weeks(df: pd.DataFrame, col_suffix: str = "_total") -> List[int]:
    """从列名中自动识别周次，例如 w1_total -> 1。"""
    weeks = set()
    pat = re.compile(r"^w(\d+)" + re.escape(col_suffix) + r"$")
    for c in df.columns:
        m = pat.match(c)
        if m:
            weeks.add(int(m.group(1)))
    return sorted(weeks)


def parse_results(results: str) -> Dict[str, Optional[int]]:
    """
    解析 results 字段，识别“淘汰周次”或“退出(Withdrew)”。
    注意：数据里 Withdrew 往往不带周次，需要用分数序列推断最后出场周。
    """
    if results is None or (isinstance(results, float) and np.isnan(results)):
        return {"eliminated_week": None, "withdrew": False}

    s = str(results).strip()
    m = re.search(r"Eliminated Week\s+(\d+)", s, flags=re.IGNORECASE)
    if m:
        return {"eliminated_week": int(m.group(1)), "withdrew": False}

    if re.fullmatch(r"Withdrew", s, flags=re.IGNORECASE):
        return {"eliminated_week": None, "withdrew": True}

    return {"eliminated_week": None, "withdrew": False}


def infer_last_active_week(row: pd.Series, weeks: List[int], score_template: str) -> Optional[int]:
    """对 Withdrew 选手，用“最后一个 >0 的周得分”推断其最后参赛周。"""
    last = None
    for t in weeks:
        val = row.get(score_template.format(t=t), np.nan)
        if pd.notna(val) and float(val) > 0:
            last = t
    return last


def build_base_table(df: pd.DataFrame, weeks: List[int], score_template: str) -> pd.DataFrame:
    """为每个选手行补充 contestant_id、eliminated_week、withdrew、withdrew_week。"""
    base = df.copy()
    base = base.reset_index(drop=True)
    base["contestant_id"] = np.arange(len(base))

    parsed = base["results"].apply(parse_results)
    base["eliminated_week"] = parsed.apply(lambda d: d["eliminated_week"])
    base["withdrew"] = parsed.apply(lambda d: d["withdrew"])
    base["withdrew_week"] = base.apply(lambda r: infer_last_active_week(r, weeks, score_template), axis=1)

    return base


def wide_to_long(base: pd.DataFrame, weeks: List[int], score_template: str) -> pd.DataFrame:
    """
    宽表 -> 长表：
    每行表示 (season, week, contestant) 的当周评委分数 J。
    """
    recs = []
    for t in weeks:
        col = score_template.format(t=t)
        if col not in base.columns:
            continue
        tmp = base[
            [
                "contestant_id", "season", "celebrity_name", "ballroom_partner",
                "placement", "results", "eliminated_week", "withdrew", "withdrew_week", col
            ]
        ].copy()
        tmp = tmp.rename(columns={col: "J"})
        tmp["week"] = t
        recs.append(tmp)

    long = pd.concat(recs, ignore_index=True)
    long["J"] = pd.to_numeric(long["J"], errors="coerce")
    # active: 当周确实参赛（题目数据说明：淘汰后记 0）
    long["active"] = long["J"].fillna(0.0) > 0.0
    return long


def season_length_map(long_df: pd.DataFrame) -> Dict[int, int]:
    """每个 season 的有效周数（当周至少有 1 人 active）。"""
    out = {}
    for s, grp in long_df[long_df["active"]].groupby("season"):
        out[int(s)] = int(grp["week"].max())
    return out


def mechanism_for_season(season: int, judges_save_from: int = DEFAULT_JUDGES_SAVE_FROM_SEASON) -> Tuple[str, bool]:
    """
    返回 (mechanism, judges_save)
    - mechanism: "percent" or "rank"
    - judges_save: True/False（仅对 rank seasons 有意义）
    """
    if season in PERCENT_SEASONS:
        return "percent", False
    if season in RANK_SEASONS:
        return "rank", (season >= judges_save_from)
    # 如果出现不在假设集合内的赛季，默认按 percent（也可改成 raise）
    return "percent", False


# =========================
# 2) 先验与排序工具
# =========================

def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = np.max(x)
    ex = np.exp(x - m)
    return ex / np.sum(ex)


def build_prior_b(pJ: np.ndarray, beta: float) -> np.ndarray:
    """弱先验：b ∝ exp(beta * judge_percent)，beta=0 时为均匀先验。"""
    if beta == 0:
        return np.ones_like(pJ) / len(pJ)
    return softmax(beta * pJ)


def rank_desc(values: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    将 values 转为排名：1 为最好（值最大），n 为最差。
    为避免浮点相等导致 ties，这里加极小抖动。
    """
    v = np.asarray(values, dtype=float)
    if rng is None:
        jitter = np.zeros_like(v)
    else:
        jitter = rng.normal(scale=1e-12, size=v.shape)
    order = np.argsort(-(v + jitter))  # descending
    ranks = np.empty_like(order, dtype=int)
    ranks[order] = np.arange(1, len(v) + 1)
    return ranks


# =========================
# 3) Percent 机制：KL-MAP + 条件采样
# =========================

def percent_constraints_satisfied(pJ: np.ndarray, v: np.ndarray, elim_set: Set[int], tol: float = 1e-12) -> bool:
    """
    Percent 机制的约束：
    若 elim_set 非空，则要求所有被淘汰者的 combined score <= 所有未淘汰者。
    combined score: c_i = pJ_i + v_i
    """
    n = len(v)
    E = set(elim_set)
    if len(E) == 0:
        return True
    S = [i for i in range(n) if i not in E]
    if len(S) == 0:
        return True
    c = pJ + v
    return (np.max(c[list(E)]) <= np.min(c[S]) + tol)


def solve_percent_map_no_slack(pJ: np.ndarray, elim_set: Set[int], b: np.ndarray,
                               eps_floor: float = 1e-12, maxiter: int = 2000) -> Tuple[np.ndarray, bool]:
    """
    Percent 机制的 MAP 点估计：
    在满足淘汰约束与 simplex 的条件下，最小化 KL(v||b)=Σ v log(v/b)。
    用 SLSQP 解小规模凸优化问题（n ≤ 20 一般很稳）。
    """
    n = len(pJ)
    idxE = sorted(list(elim_set))
    idxS = [i for i in range(n) if i not in elim_set]

    def obj(v):
        v = np.asarray(v, dtype=float)
        v_clip = np.clip(v, eps_floor, 1.0)
        b_clip = np.clip(b, eps_floor, 1.0)
        return float(np.sum(v_clip * np.log(v_clip / b_clip)))

    cons = [{"type": "eq", "fun": lambda v: np.sum(v) - 1.0}]
    if len(idxE) > 0 and len(idxS) > 0:
        for e in idxE:
            for k in idxS:
                cons.append({"type": "ineq", "fun": lambda v, e=e, k=k: (pJ[k] + v[k]) - (pJ[e] + v[e])})

    bounds = [(0.0, 1.0)] * n
    v0 = b.copy()
    res = minimize(obj, v0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"ftol": 1e-10, "maxiter": maxiter, "disp": False})
    v_hat = np.asarray(res.x, dtype=float)
    if np.sum(v_hat) > 0:
        v_hat = v_hat / np.sum(v_hat)
    return v_hat, bool(res.success)


def solve_percent_map_with_slack(pJ: np.ndarray, elim_set: Set[int], b: np.ndarray,
                                 lambda_eps: float = 1e4, eps_floor: float = 1e-12,
                                 maxiter: int = 2000) -> Tuple[np.ndarray, float, bool]:
    """
    若无松弛的 MAP 求解失败，则引入 slack eps ≥0：
    约束变为 (pJ_k+v_k) - (pJ_e+v_e) + eps ≥ 0。
    目标：KL(v||b) + lambda_eps * eps^2。
    """
    n = len(pJ)
    idxE = sorted(list(elim_set))
    idxS = [i for i in range(n) if i not in elim_set]

    def obj(x):
        v = x[:n]
        eps = x[-1]
        v_clip = np.clip(v, eps_floor, 1.0)
        b_clip = np.clip(b, eps_floor, 1.0)
        kl = float(np.sum(v_clip * np.log(v_clip / b_clip)))
        return kl + lambda_eps * (eps ** 2)

    cons = [{"type": "eq", "fun": lambda x: np.sum(x[:n]) - 1.0}]
    if len(idxE) > 0 and len(idxS) > 0:
        for e in idxE:
            for k in idxS:
                cons.append({
                    "type": "ineq",
                    "fun": lambda x, e=e, k=k: (pJ[k] + x[k]) - (pJ[e] + x[e]) + x[-1]
                })

    bounds = [(0.0, 1.0)] * n + [(0.0, None)]
    x0 = np.concatenate([b.copy(), [0.0]])
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"ftol": 1e-10, "maxiter": maxiter, "disp": False})
    v_hat = np.asarray(res.x[:n], dtype=float)
    if np.sum(v_hat) > 0:
        v_hat = v_hat / np.sum(v_hat)
    eps_hat = float(res.x[-1])
    return v_hat, eps_hat, bool(res.success)


def sample_dirichlet_centered(rng: np.random.Generator, center: np.ndarray, kappa: float) -> np.ndarray:
    """
    [修复版] 从 Dirichlet(kappa*center) 采样
    增加了对 NaN 和 0 的防护，防止程序崩溃。
    """
    # 1. 强力清洗数据：把 NaN 换成极小值，把负数变 0
    c = np.nan_to_num(np.array(center, dtype=float), nan=1e-9, posinf=1e-9, neginf=1e-9)
    c = np.maximum(c, 1e-9)  # 所有的值至少是 1e-9，防止全 0

    # 2. 安全归一化
    c_sum = np.sum(c)
    if c_sum == 0:
        c = np.ones_like(c) / len(c)  # 如果全为0，就平均分配
    else:
        c = c / c_sum

    # 3. 计算 alpha 并采样
    alpha = kappa * c

    try:
        # 有时候 alpha 太小也会导致 numpy 报错，加个 try
        return rng.dirichlet(alpha)
    except ValueError:
        # 如果出错了，返回一个均匀分布作为保底
        return np.ones_like(c) / len(c)


def posterior_samples_percent(rng: np.random.Generator, pJ: np.ndarray, elim_set: Set[int],
                              v_center: np.ndarray, n_samples: int, kappa: float,
                              max_attempts: int = 200000, tol: float = 1e-12) -> Tuple[np.ndarray, float]:
    """
    Percent 机制后验采样（易实现版）：
    以 MAP 点 v_center 为中心做 Dirichlet 提议，拒绝不满足淘汰约束的样本。
    返回：samples (M x n), accept_rate
    """
    accepted = []
    attempts = 0
    while len(accepted) < n_samples and attempts < max_attempts:
    #当程序跑了 20 万次还没凑够样本，它会退出循环。但此时 accepted 可能是空的，导致后续计算 margin_mean 时出现除以零或空切片
        v = sample_dirichlet_centered(rng, v_center, kappa=kappa)
        if percent_constraints_satisfied(pJ, v, elim_set, tol=tol):
            accepted.append(v)
        attempts += 1

    if len(accepted) == 0:
        return np.zeros((0, len(pJ))), 0.0

    samples = np.vstack(accepted)
    accept_rate = len(accepted) / max(1, attempts)
    return samples, accept_rate


# =========================
# 4) Rank 机制：蒙特卡洛拒绝采样（含 Judges' Save bottom-2）
# =========================

def accept_rank_sample(judge_scores: np.ndarray, v: np.ndarray, elim_set: Set[int],
                       judges_save: bool, rng: np.random.Generator,
                       bottom_m: Optional[int] = None) -> Tuple[bool, Dict[str, object]]:
    """
    判断一个 v 样本是否与淘汰结果相容：
    - 无 Judges' Save：要求 elim_set = combined_rank 最差的 K 名（K=|elim_set|）
    - 有 Judges' Save：只要求 elim_set ⊆ bottom-2（或 bottom_m）
    """
    n = len(v)
    K = len(elim_set)
    if bottom_m is None:
        bottom_m = max(2, K) if judges_save else K

    judge_rank = rank_desc(judge_scores, rng=rng)
    fan_rank = rank_desc(v, rng=rng)
    B = judge_rank + fan_rank

    worst_order = np.argsort(B)[::-1]  # largest B = worst
    predicted_worst = set(worst_order[:K]) if K > 0 else set()
    bottom_set = set(worst_order[:bottom_m]) if bottom_m > 0 else set()

    if K == 0:
        return True, {"B": B, "predicted_worst": predicted_worst, "bottom_set": bottom_set}

    if judges_save:
        acc = set(elim_set).issubset(bottom_set)
        return acc, {"B": B, "predicted_worst": predicted_worst, "bottom_set": bottom_set}
    else:
        acc = (set(elim_set) == predicted_worst)
        return acc, {"B": B, "predicted_worst": predicted_worst, "bottom_set": bottom_set}


def posterior_samples_rank(rng: np.random.Generator, judge_scores: np.ndarray, pJ: np.ndarray, elim_set: Set[int],
                           judges_save: bool, alpha: float, beta: float,
                           n_samples: int, max_attempts: int = 300000) -> Tuple[np.ndarray, float, Dict[str, float]]:
    """
    Rank 机制后验采样（拒绝采样/ABC）：
      1) 先验：v ~ Dirichlet(alpha * b), b ∝ exp(beta * pJ)
      2) 接受准则：依照 rank 规则（Season 28+ 用 bottom-2）
    返回：samples, accept_rate, extra_metrics（例如 exact_accept_rate）
    """
    n = len(judge_scores)
    b = build_prior_b(pJ, beta=beta)
    # Dirichlet 参数向量必须 >0
    eps = 1e-12
    b = np.clip(b, eps, 1.0)
    b = b / np.sum(b)

    accepted = []
    attempts = 0

    exact_ok = 0  # 统计“严格倒数第一/倒数K”匹配率（即使 judges_save 放宽，也可以算来评估一致性强弱）

    while len(accepted) < n_samples and attempts < max_attempts:
        v = rng.dirichlet(alpha * b)
        ok, info = accept_rank_sample(judge_scores, v, elim_set, judges_save=judges_save, rng=rng)
        if ok:
            accepted.append(v)
        # 额外统计：严格匹配（不放宽）时的比例
        ok_exact, _ = accept_rank_sample(judge_scores, v, elim_set, judges_save=False, rng=rng)
        if ok_exact:
            exact_ok += 1
        attempts += 1

    if len(accepted) == 0:
        return np.zeros((0, n)), 0.0, {"exact_accept_rate": 0.0}

    samples = np.vstack(accepted)
    accept_rate = len(accepted) / max(1, attempts)
    exact_accept_rate = exact_ok / max(1, attempts)
    return samples, accept_rate, {"exact_accept_rate": exact_accept_rate}


# =========================
# 5) 统计量：均值/Std Dev/CI/一致性/边际
# =========================

def summarize_posterior(samples: np.ndarray) -> Dict[str, np.ndarray]:
    """
    对 samples(M x n) 计算：
    mean, std, median, ci_low(2.5%), ci_high(97.5%)
    """
    if samples.size == 0:
        return {
            "mean": np.array([]),
            "std": np.array([]),
            "median": np.array([]),
            "ci_low": np.array([]),
            "ci_high": np.array([]),
        }
    mean = samples.mean(axis=0)
    std = samples.std(axis=0, ddof=1) if samples.shape[0] > 1 else np.zeros(samples.shape[1])
    median = np.median(samples, axis=0)
    ci_low = np.quantile(samples, 0.025, axis=0)
    ci_high = np.quantile(samples, 0.975, axis=0)
    return {"mean": mean, "std": std, "median": median, "ci_low": ci_low, "ci_high": ci_high}


def percent_week_margin(pJ: np.ndarray, v: np.ndarray, elim_set: Set[int]) -> float:
    """Percent 机制：margin = min_survivor(c) - max_elim(c)，越大表示淘汰越“稳”。"""
    E = set(elim_set)
    if len(E) == 0:
        return np.nan
    n = len(v)
    S = [i for i in range(n) if i not in E]
    if len(S) == 0:
        return np.nan
    c = pJ + v
    return float(np.min(c[S]) - np.max(c[list(E)]))


def rank_week_margin(judge_scores: np.ndarray, v: np.ndarray) -> float:
    """
    Rank 机制：用 combined rank 的“最差与倒数第二差”的差作为 margin。
    margin 越大表示 bottom 区分越明显。
    """
    rng = np.random.default_rng(0)
    jr = rank_desc(judge_scores, rng=rng)
    fr = rank_desc(v, rng=rng)
    B = jr + fr
    worst_order = np.argsort(B)[::-1]
    if len(worst_order) < 2:
        return np.nan
    return float(B[worst_order[0]] - B[worst_order[1]])


# =========================
# 6) 主流程：逐 (season, week) 推断
# =========================

@dataclass
class InferenceConfig:
    beta: float = 0.0              # 先验强度（score -> vote 的弱趋势）
    alpha_rank: float = 1.0        # rank seasons Dirichlet 参数缩放
    n_samples: int = 2000          # 每个 (season, week) 后验样本数
    kappa_percent: float = 300.0   # percent seasons: 以 v_hat 为中心的 Dirichlet 集中度
    seed: int = 123
    max_attempts_percent: int = 200000
    max_attempts_rank: int = 300000
    use_slack_fallback: bool = True


def infer_all(df_path: str, out_dir: str, cfg: InferenceConfig,
              judges_save_from_season: int = DEFAULT_JUDGES_SAVE_FROM_SEASON) -> Tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(df_path)
    weeks = detect_weeks(df, col_suffix="_total")
    base = build_base_table(df, weeks, score_template=SCORE_TEMPLATE)
    long_df = wide_to_long(base, weeks, score_template=SCORE_TEMPLATE)
    season_len = season_length_map(long_df)

    rng = np.random.default_rng(cfg.seed)

    contestant_rows = []
    week_rows = []

    # 建一个快速索引：按 (season, week) 取 active 行
    grouped = long_df[long_df["active"]].groupby(["season", "week"])

    # 建一个淘汰映射：仅对 “Eliminated Week t”，Withdrew 不计入投票淘汰
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

    # 主循环
    for (s_raw, t_raw), grp in grouped:
        s = int(s_raw); t = int(t_raw)
        mech, judges_save = mechanism_for_season(s, judges_save_from=judges_save_from_season)

        # active contestants of this week
        grp = grp.sort_values("contestant_id")
        cids = grp["contestant_id"].astype(int).tolist()
        names = grp["celebrity_name"].tolist()

        J = grp["J"].astype(float).to_numpy()
        pJ = J / np.sum(J)

        # elimination set indices in this local array
        elim_cids = elim_map.get((s, t), set())
        elim_idx = {cids.index(cid) for cid in elim_cids if cid in cids}

        # ========= 推断 =========
        samples = np.zeros((0, len(J)))
        accept_rate = 0.0
        extra = {}

        if mech == "percent":
            b = build_prior_b(pJ, beta=cfg.beta)

            v_hat, ok = solve_percent_map_no_slack(pJ, elim_idx, b)
            eps_hat = 0.0
            if (not ok) and cfg.use_slack_fallback:
                v_hat, eps_hat, ok2 = solve_percent_map_with_slack(pJ, elim_idx, b)
                # 注意：如果启用了 slack，则后续验收用 eps_hat 作为容忍度（这里简化：仍使用 tol=1e-12）
                ok = ok2

            # 采样（无淘汰周 elim_idx 为空时，约束恒成立，得到以 v_hat 为中心的分布）
            samples, accept_rate = posterior_samples_percent(
                rng, pJ, elim_idx, v_center=v_hat, n_samples=cfg.n_samples,
                kappa=cfg.kappa_percent, max_attempts=cfg.max_attempts_percent
            )

            # 一致性：严格意义下的 PPC（样本是否复现淘汰集合）
            if len(elim_idx) == 0:
                ppc_exact = np.nan
            else:
                # 由于我们是“条件采样”，理论上 ppc_exact≈1，仍计算作为 sanity check
                exact_hits = 0
                for v in samples:
                    c = pJ + v
                    K = len(elim_idx)
                    worst = set(np.argsort(c)[:K])  # lowest K
                    exact_hits += int(worst == set(elim_idx))
                ppc_exact = exact_hits / max(1, len(samples))

            # margin（平均）
            margins = [percent_week_margin(pJ, v, elim_idx) for v in samples] if len(samples) > 0 else []
            # 过滤掉可能的 None 或 NaN
            valid_margins = [m for m in margins if m is not None and not np.isnan(m)]

            if len(valid_margins) > 0:
                margin_mean = float(np.mean(valid_margins))
            else:
                margin_mean = 0.0

            extra = {"eps_hat": float(eps_hat)}

        else:  # mech == "rank"
            samples, accept_rate, extra2 = posterior_samples_rank(
                rng, J, pJ, elim_idx,
                judges_save=judges_save,
                alpha=cfg.alpha_rank,
                beta=cfg.beta,
                n_samples=cfg.n_samples,
                max_attempts=cfg.max_attempts_rank
            )
            extra.update(extra2)

            # 一致性：两种口径
            # - ppc_relaxed：judges_save 为 True 时，表示“是否在 bottom-2”（条件采样下应为 1）
            # - ppc_exact：严格要求“就是最差K”，用于量化 judges_save 带来的不确定性增大
            if len(elim_idx) == 0:
                ppc_relaxed = np.nan
                ppc_exact = np.nan
            else:
                relaxed_hits = 0
                exact_hits = 0
                for v in samples:
                    ok_relaxed, info = accept_rank_sample(J, v, elim_idx, judges_save=judges_save, rng=rng)
                    ok_exact, _ = accept_rank_sample(J, v, elim_idx, judges_save=False, rng=rng)
                    relaxed_hits += int(ok_relaxed)
                    exact_hits += int(ok_exact)
                ppc_relaxed = relaxed_hits / max(1, len(samples))
                ppc_exact = exact_hits / max(1, len(samples))

            margins = [percent_week_margin(pJ, v, elim_idx) for v in samples] if len(samples) > 0 else []
            # 过滤掉可能的 None 或 NaN
            valid_margins = [m for m in margins if m is not None and not np.isnan(m)]

            if len(valid_margins) > 0:
                margin_mean = float(np.mean(valid_margins))
            else:
                margin_mean = 0.0

            extra.update({"ppc_relaxed": ppc_relaxed})

        # ========= 汇总 =========
        summ = summarize_posterior(samples)

        for pos, cid in enumerate(cids):
            contestant_rows.append({
                "season": s,
                "week": t,
                "contestant_id": cid,
                "celebrity_name": names[pos],
                "mechanism": mech,
                "judges_save": bool(judges_save),
                "n_active": len(cids),
                "n_elim": len(elim_idx),
                "vote_mean": float(summ["mean"][pos]) if summ["mean"].size else np.nan,
                "vote_std": float(summ["std"][pos]) if summ["std"].size else np.nan,
                "vote_median": float(summ["median"][pos]) if summ["median"].size else np.nan,
                "ci_low": float(summ["ci_low"][pos]) if summ["ci_low"].size else np.nan,
                "ci_high": float(summ["ci_high"][pos]) if summ["ci_high"].size else np.nan,
                "ci_width": float(summ["ci_high"][pos] - summ["ci_low"][pos]) if summ["ci_high"].size else np.nan,
            })

        week_row = {
            "season": s,
            "week": t,
            "mechanism": mech,
            "judges_save": bool(judges_save),
            "n_active": len(cids),
            "n_elim": len(elim_idx),
            "accept_rate": float(accept_rate),
            "margin_mean": float(margin_mean),
        }
        # 填写一致性指标
        if mech == "percent":
            week_row["ppc_exact"] = float(ppc_exact) if "ppc_exact" in locals() else np.nan
            week_row["eps_hat"] = extra.get("eps_hat", 0.0)
        else:
            week_row["ppc_exact"] = float(ppc_exact) if "ppc_exact" in locals() else np.nan
            week_row["ppc_relaxed"] = extra.get("ppc_relaxed", np.nan)
            week_row["exact_accept_rate"] = extra.get("exact_accept_rate", np.nan)

        week_rows.append(week_row)

    contestant_df = pd.DataFrame(contestant_rows).sort_values(["season", "week", "contestant_id"])
    week_df = pd.DataFrame(week_rows).sort_values(["season", "week"])

    contestant_out = os.path.join(out_dir, "contestant_posterior_summary.csv")
    week_out = os.path.join(out_dir, "week_diagnostics.csv")
    contestant_df.to_csv(contestant_out, index=False, encoding="utf-8-sig")
    week_df.to_csv(week_out, index=False, encoding="utf-8-sig")

    return contestant_df, week_df


# =========================
# 7) 可选：画 Violin Plot（示意）
# =========================

def save_violin_data(df_path: str, out_path: str, season: int, week: int,
                     cfg: InferenceConfig) -> None:
    """

    仅对指定 (season, week) 输出 posterior samples（长表），便于 seaborn/matplotlib 画 violin plot。

    输出列：sample_id, contestant_id, celebrity_name, vote_share
    """
    df = pd.read_csv(df_path)
    weeks = detect_weeks(df, col_suffix="_total")
    base = build_base_table(df, weeks, score_template=SCORE_TEMPLATE)
    long_df = wide_to_long(base, weeks, score_template=SCORE_TEMPLATE)

    rng = np.random.default_rng(cfg.seed)

    # active
    grp = long_df[(long_df["season"] == season) & (long_df["week"] == week) & (long_df["active"])].copy()
    grp = grp.sort_values("contestant_id")
    cids = grp["contestant_id"].astype(int).tolist()
    names = grp["celebrity_name"].tolist()

    J = grp["J"].astype(float).to_numpy()
    pJ = J / np.sum(J)

    # elim set
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

    mech, judges_save = mechanism_for_season(season)

    if mech == "percent":
        b = build_prior_b(pJ, beta=cfg.beta)
        v_hat, ok = solve_percent_map_no_slack(pJ, elim_idx, b)
        if (not ok) and cfg.use_slack_fallback:
            v_hat, _, _ = solve_percent_map_with_slack(pJ, elim_idx, b)
        samples, _ = posterior_samples_percent(rng, pJ, elim_idx, v_center=v_hat,
                                               n_samples=cfg.n_samples, kappa=cfg.kappa_percent)
    else:
        samples, _, _ = posterior_samples_rank(rng, J, pJ, elim_idx, judges_save=judges_save,
                                               alpha=cfg.alpha_rank, beta=cfg.beta, n_samples=cfg.n_samples)

    rows = []
    for m in range(samples.shape[0]):
        for j in range(samples.shape[1]):
            rows.append({
                "sample_id": m,
                "contestant_id": cids[j],
                "celebrity_name": names[j],
                "vote_share": float(samples[m, j])
            })
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",
                    default=r"E:\PYthonCode\2026_MCM_Problem_C\data\processed_data\cleaned_data.csv",
                    help="输入文件路径")

    # ---------------------------------------------------------
    # 修改点 2: 同样给输出目录加上 default
    # ---------------------------------------------------------
    ap.add_argument("--out_dir",
                    default=r"E:\PYthonCode\2026_MCM_Problem_C\data\Q1data",
                    help="输出文件夹路径")
    ap.add_argument("--beta", type=float, default=0.0, help="先验参数 beta（score->vote 的弱趋势）")
    ap.add_argument("--alpha_rank", type=float, default=1.0, help="rank seasons Dirichlet 缩放 alpha")
    ap.add_argument("--n_samples", type=int, default=2000, help="每个 (season, week) 的后验样本数")
    ap.add_argument("--kappa_percent", type=float, default=300.0, help="percent seasons 提议分布集中度 kappa")
    ap.add_argument("--seed", type=int, default=123, help="随机种子")
    ap.add_argument("--judges_save_from", type=int, default=DEFAULT_JUDGES_SAVE_FROM_SEASON,
                    help="从哪个 season 开始启用 judges save（默认 28）")
    ap.add_argument("--export_violin", action="store_true", help="是否额外导出一个(season,week)的violin数据")
    ap.add_argument("--violin_season", type=int, default=28)
    ap.add_argument("--violin_week", type=int, default=1)
    ap.add_argument("--violin_out", default="violin_samples.csv")
    args = ap.parse_args()

    cfg = InferenceConfig(
        beta=args.beta,
        alpha_rank=args.alpha_rank,
        n_samples=args.n_samples,
        kappa_percent=args.kappa_percent,
        seed=args.seed,
    )

    infer_all(args.input, args.out_dir, cfg, judges_save_from_season=args.judges_save_from)

    if args.export_violin:
        save_violin_data(args.input, args.violin_out, args.violin_season, args.violin_week, cfg)
        print(f"[OK] violin 样本已导出到: {args.violin_out}")


if __name__ == "__main__":
    main()
