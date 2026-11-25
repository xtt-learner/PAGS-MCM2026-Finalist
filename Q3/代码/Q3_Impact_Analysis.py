# q3_pipeline.py
# -*- coding: utf-8 -*-
"""
Q3 Full Pipeline (Coding Stage)
--------------------------------
Implements:
1) Pro dancer feature engineering (Tenure, LOSO historical technical mean)
2) Matching bias tests (endogeneity diagnostics)
3) Three-channel models:
   - Judges channel: Linear Mixed Effects (MixedLM)
   - Fan channel: MI over Q1 posterior vote samples + MixedLM
   - Outcome channel: Discrete-time hazard (Logit) + era interactions (GEE / GLM as approximation)
4) Non-linear age (quadratic + optional spline features)
5) Quadrant mapping of dancer effects (b_d^(J), b_d^(V))
6) Random forest robustness + PDP + SHAP global interpretation

Hard definitions aligned with Q1/Q2:
- J_{i,s,t} = wt_avg (dynamic average judge score) on in-competition weeks only
- C_{s,t} shown as rows with wt_avg > 0 (post-elim placeholders excluded)
- v_{i,s,t} comes from Q1 posterior samples and is propagated via MI (Rubin rules)

Author: for team use (explain I/O carefully)
"""

import argparse
import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
# ================= ADDED: 导入 Q1 模块用于数据变形 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
try:
    import Q1_Fan_Estimation as Q1  # 假设你的文件名叫 Q1_Fan_Estimation.py
except ImportError:
    try:
        import Q1  # 备用尝试
    except ImportError:
        print("Warning: Could not import Q1 module. Data reshaping might fail.")
# =================================================================

# Stats / modeling
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable

# Nonlinearity / splines
from patsy.highlevel import dmatrix

# ML
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay

import matplotlib.pyplot as plt

# SHAP (optional)
try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False


# -----------------------------
# 0) Config + column inference
# -----------------------------

@dataclass
class ColMap:
    season: str
    week: str
    wt_avg: str
    ballroom_partner: str
    celebrity_name: str
    celebrity_age: str
    celebrity_industry: str
    week_eliminated: Optional[str] = None  # may be missing; code can infer from other columns if needed


def _lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def infer_columns(df: pd.DataFrame) -> ColMap:
    """
    Attempts to infer column names from common candidates.
    If inference fails, user should pass manual mapping via JSON config (see --colmap_json).
    """
    cols = set(df.columns)

    def pick(candidates: List[str], required: bool = True) -> Optional[str]:
        for c in candidates:
            if c in cols:
                return c
        if required:
            raise ValueError(f"Cannot infer required column from candidates={candidates}. "
                             f"Available columns={sorted(list(cols))[:50]} ...")
        return None

    season = pick(["season", "Season", "SEASON"])
    week = pick(["week", "Week", "WEEK", "week_number", "week_num"])
    wt_avg = pick(["wt_avg", "Wt_avg", "WT_AVG", "S_avg", "score_avg", "weekly_avg_score"])
    ballroom_partner = pick(["ballroom_partner", "Ballroom_partner", "pro_dancer", "partner", "pro"])
    celebrity_name = pick(["celebrity_name", "Celebrity_name", "star_name", "celebrity", "name"])
    celebrity_age = pick(["celebrity_age_during_season", "Age", "age", "celebrity_age"])
    celebrity_industry = pick(["celebrity_industry", "Industry", "industry"])

    week_eliminated = pick(["week_eliminated", "Week_eliminated", "elimination_week", "E_week"], required=False)

    return ColMap(
        season=season,
        week=week,
        wt_avg=wt_avg,
        ballroom_partner=ballroom_partner,
        celebrity_name=celebrity_name,
        celebrity_age=celebrity_age,
        celebrity_industry=celebrity_industry,
        week_eliminated=week_eliminated
    )


def era_from_season(s: int) -> int:
    # g(s) in the paper
    if s in [1, 2]:
        return 1  # Rank
    elif 3 <= s <= 27:
        return 2  # Percent
    else:
        return 3  # Judges' Save


# -------------------------------------------
# 1) Build C_{s,t} panel & judge-channel YJ
# -------------------------------------------

def build_panel_weekly(df_raw: pd.DataFrame, cm: ColMap) -> pd.DataFrame:
    """
    Construct weekly in-competition panel (C_{s,t}):
    - Keep rows with wt_avg > 0 (post-elimination placeholders excluded).
    - Create unique pair_id (contestant i): season + celebrity + pro dancer.

    Output columns include:
    - season, week, era
    - pair_id
    - dancer_id
    - age, industry
    - J = wt_avg
    - YJ = within-(season,week) standardized judge score
    """
    df = df_raw.copy()

    # Normalize dtypes
    df[cm.season] = df[cm.season].astype(int)
    df[cm.week] = df[cm.week].astype(int)

    # In-competition mask: strict rule wt_avg > 0
    df["in_competition"] = df[cm.wt_avg].notna() & (df[cm.wt_avg] > 0)
    dfC = df[df["in_competition"]].copy()

    # Build pair_id (i): must be stable across weeks within season
    dfC["pair_id"] = (
        dfC[cm.season].astype(str) + "||" +
        dfC[cm.celebrity_name].astype(str).str.strip() + "||" +
        dfC[cm.ballroom_partner].astype(str).str.strip()
    )

    dfC["dancer_id"] = dfC[cm.ballroom_partner].astype(str).str.strip()
    dfC["celebrity_id"] = dfC[cm.celebrity_name].astype(str).str.strip()
    dfC["age"] = pd.to_numeric(dfC[cm.celebrity_age], errors="coerce")
    dfC["industry"] = dfC[cm.celebrity_industry].astype(str).str.strip()
    K_INDUSTRY = 20  # frequency threshold; tune 10/20/30 if needed


    dfC["industry_clean"] = dfC["industry"].astype(str).str.strip().str.lower()
    freq = dfC["industry_clean"].value_counts(dropna=False)
    rare = freq[freq < K_INDUSTRY].index
    dfC["industry_grp"] = dfC["industry_clean"].where(~dfC["industry_clean"].isin(rare), other="other")

    dfC["season"] = dfC[cm.season]
    dfC["week"] = dfC[cm.week]
    dfC["era"] = dfC["season"].apply(era_from_season)

    # J = wt_avg
    dfC["J"] = pd.to_numeric(dfC[cm.wt_avg], errors="coerce")

    # Within (season,week) standardization for judge channel: YJ
    grp = dfC.groupby(["season", "week"])
    dfC["J_mean_st"] = grp["J"].transform("mean")
    dfC["J_sd_st"] = grp["J"].transform("std").replace(0.0, np.nan)
    dfC["YJ"] = (dfC["J"] - dfC["J_mean_st"]) / dfC["J_sd_st"]

    # If some (s,t) have sd==0 (rare), set YJ=0 (all equal)
    dfC["YJ"] = dfC["YJ"].fillna(0.0)

    return dfC


# ---------------------------------------------------------
# 2) Pro dancer feature engineering: tenure & muJ_loso
# ---------------------------------------------------------

def compute_pro_dancer_profile(dfC: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute dancer features:
    - tenure τ_d: number of seasons dancer appears (at least one in-competition week)
    - muJ_loso(d, -s): leave-one-season-out historical mean of J (wt_avg) for that dancer

    Returns:
    - df_profile: one row per dancer (tenure + overall stats)
    - dfC_aug: dfC with columns tenure, muJ_loso (season-specific LOSO)
    """
    # Tenure: seasons count
    tenure = (
        dfC.groupby(["dancer_id", "season"])
           .size()
           .reset_index(name="n_weeks")
           .groupby("dancer_id")["season"]
           .nunique()
           .rename("tenure")
           .reset_index()
    )

    # Sums/counts per dancer-season
    ds_sum_cnt = (
        dfC.groupby(["dancer_id", "season"])["J"]
           .agg(sumJ_s="sum", cntJ_s="count")
           .reset_index()
    )

    # Overall sums/counts per dancer
    d_sum_cnt = (
        dfC.groupby("dancer_id")["J"]
           .agg(sumJ="sum", cntJ="count", meanJ="mean")
           .reset_index()
    )

    # LOSO = (sumJ - sumJ_s) / (cntJ - cntJ_s)
    tmp = ds_sum_cnt.merge(d_sum_cnt, on="dancer_id", how="left")
    tmp["muJ_loso"] = (tmp["sumJ"] - tmp["sumJ_s"]) / (tmp["cntJ"] - tmp["cntJ_s"])
    tmp.loc[(tmp["cntJ"] - tmp["cntJ_s"]) <= 0, "muJ_loso"] = np.nan

    global_meanJ = dfC["J"].mean()
    tmp["muJ_loso"] = tmp["muJ_loso"].fillna(global_meanJ)

    # Attach back to weekly panel
    dfC_aug = dfC.merge(tenure, on="dancer_id", how="left")
    dfC_aug = dfC_aug.merge(tmp[["dancer_id", "season", "muJ_loso"]], on=["dancer_id", "season"], how="left")

    # dancer profile output
    df_profile = tenure.merge(d_sum_cnt, on="dancer_id", how="left").sort_values("tenure", ascending=False)

    return df_profile, dfC_aug


# ------------------------------------------------
# 3) Matching bias tests (endogeneity diagnostics)
# ------------------------------------------------

def matching_bias_tests(dfC_aug: pd.DataFrame, out_txt_path: str, hot_top_k: int = 3) -> None:
    """
    Test whether better historical dancers tend to be assigned to:
    - younger celebrities
    - hot industries (top-K by frequency)

    We use prior dancer quality Q_prior(d,s) computed from seasons < s.

    Output: writes regression summaries to a .txt file
    """
    # Build pairing table: unique (season, celebrity, dancer)
    pair = (
        dfC_aug.groupby(["season", "celebrity_id", "dancer_id"], as_index=False)
              .agg(age=("age", "first"),
                   industry=("industry", "first"))
    )

    # Compute dancer season stats and prior Q
    ds = (
        dfC_aug.groupby(["dancer_id", "season"])["J"]
               .agg(sumJ="sum", cntJ="count")
               .reset_index()
               .sort_values(["dancer_id", "season"])
    )
    ds["cum_sum_prev"] = ds.groupby("dancer_id")["sumJ"].cumsum().shift(1)
    ds["cum_cnt_prev"] = ds.groupby("dancer_id")["cntJ"].cumsum().shift(1)
    ds["Q_prior"] = ds["cum_sum_prev"] / ds["cum_cnt_prev"]

    pair = pair.merge(ds[["dancer_id", "season", "Q_prior"]], on=["dancer_id", "season"], how="left")

    # Hot industry definition: top-K by frequency (data-driven)
    top_ind = pair["industry"].value_counts().head(hot_top_k).index.tolist()
    pair["hot"] = pair["industry"].isin(top_ind).astype(int)

    # Regress age on Q_prior with season FE (robust SE)
    age_df = pair.dropna(subset=["Q_prior", "age"]).copy()
    age_model = smf.ols("age ~ Q_prior + C(season)", data=age_df).fit(cov_type="HC3")

    # Logit hot on Q_prior with season FE
    hot_df = pair.dropna(subset=["Q_prior"]).copy()
    hot_model = smf.logit("hot ~ Q_prior + C(season)", data=hot_df).fit(disp=False)

    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write("=== Matching Bias Tests (Endogeneity Diagnostics) ===\n\n")
        f.write(f"Hot industry defined as top-{hot_top_k} by frequency: {top_ind}\n\n")
        f.write("\n--- (A) Age ~ Q_prior + Season FE ---\n")
        f.write(age_model.summary().as_text())
        f.write("\n\n--- (B) HotIndustry ~ Q_prior + Season FE (Logit) ---\n")
        f.write(hot_model.summary().as_text())


# --------------------------------------------
# 4) Add nonlinear age: quadratic + splines
# --------------------------------------------

def add_age_nonlinearity(df: pd.DataFrame, spline_df: int = 5) -> pd.DataFrame:
    """
    Add:
    - centered age (age_c)
    - quadratic term (age_c2)
    - natural cubic spline basis bs1..bsK (optional features for RF or regression)

    Returns new df with added columns.
    """
    out = df.copy()
    out["age_c"] = out["age"] - out["age"].mean()
    out["age_c2"] = out["age_c"] ** 2

    # Natural cubic splines basis via patsy (bs)
    # Note: include_intercept=False keeps base out; we can include in model as needed
    bs = dmatrix(f"bs(age, df={spline_df}, degree=3, include_intercept=False)",
                 data=out, return_type="dataframe")
    bs_cols = [f"bs{i+1}" for i in range(bs.shape[1])]
    bs.columns = bs_cols
    out = pd.concat([out.reset_index(drop=True), bs.reset_index(drop=True)], axis=1)
    return out


# --------------------------------------------
# 5) Mixed effects: Judges channel (YJ)
# --------------------------------------------
def fit_mixedlm_judges(dfC: pd.DataFrame, out_fixed_csv: str, out_re_csv: str) -> None:
    """
    Judges channel model:
      YJ ~ era:week + age_c + age_c2 + industry + tenure + muJ_loso + (random intercept by dancer) + (vc by season)
    """
    # 1. 显式清洗数据
    req_cols = ["YJ", "era", "week", "age_c", "age_c2", "industry", "tenure", "muJ_loso", "dancer_id", "season"]
    existing_cols = [c for c in req_cols if c in dfC.columns]
    d = dfC.dropna(subset=existing_cols).copy()

    print(f"   -> Fitting Judges Model on {len(d)} rows...")

    formula = "YJ ~ C(era)*C(week) + age_c + age_c2 + C(industry) + tenure + muJ_loso"
    vc = {"season": "0 + C(season)"}

    md = MixedLM.from_formula(formula, groups="dancer_id", vc_formula=vc, data=d, re_formula="1")

    # 2. 增强拟合逻辑 (强制检测收敛性)
    try:
        # 优先尝试 lbfgs/cg
        res = md.fit(reml=True, method=["lbfgs", "cg"], maxiter=5000)
        # 🔥 如果没收敛，主动抛出异常以触发 Powell 算法
        if not res.converged:
            print("      [Warning] LBFGS/CG failed to converge. Switching to Powell...")
            raise RuntimeError("Model did not converge")
    except Exception as e:
        print(f"      [Log] Retrying with Powell method (robust but slow)...")
        # 备选方案：Powell 算法
        res = md.fit(reml=True, method="powell", maxiter=3000)

    # 3. 🔥 关键修复：使用 Pandas 索引对齐构建结果表
    # 不再使用 .values，而是直接传入 Series，Pandas 会自动处理长度不一致的问题 (填充 NaN)
    fe = pd.DataFrame({
        "coef": res.fe_params,
        "se": res.bse,
        "pval": res.pvalues
    })

    # 整理表格格式
    fe.reset_index(inplace=True)
    fe.rename(columns={"index": "term"}, inplace=True)

    # 填充可能出现的 NaN (防止后续处理报错)
    fe.fillna(0, inplace=True)

    fe.to_csv(out_fixed_csv, index=False)

    # 4. 提取 Dancer 随机效应
    re_list = []
    if hasattr(res, 'random_effects'):
        for dancer, vec in res.random_effects.items():
            try:
                # 尝试获取第一个元素
                val = vec[0] if isinstance(vec, (list, np.ndarray)) else vec.iloc[0]
                re_list.append({"dancer_id": dancer, "bJ": float(val)})
            except:
                continue

    if re_list:
        re_df = pd.DataFrame(re_list).sort_values("bJ", ascending=False)
        re_df.to_csv(out_re_csv, index=False)
    else:
        # 如果提取失败，创建一个空的占位文件
        pd.DataFrame(columns=["dancer_id", "bJ"]).to_csv(out_re_csv, index=False)

    print(f"      [Done] Judges model saved to {out_fixed_csv}")


# ----------------------------------------------------------
# 6) Vote samples -> Fan channel response: clr + standardize
# ----------------------------------------------------------

def load_vote_samples(path: str) -> pd.DataFrame:
    """
    Load long-form vote samples:
      season, week, pair_id, m, v
    """
    if path.lower().endswith(".parquet"):
        vs = pd.read_parquet(path)
    else:
        vs = pd.read_csv(path)

    required = {"season", "week", "pair_id", "m", "v"}
    miss = required - set(vs.columns)
    if miss:
        raise ValueError(f"vote_samples missing columns: {miss}. Expected {required}.")
    vs["season"] = vs["season"].astype(int)
    vs["week"] = vs["week"].astype(int)
    vs["m"] = vs["m"].astype(int)
    vs["v"] = pd.to_numeric(vs["v"], errors="coerce")
    return vs


def validate_and_fix_simplex(vs: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    """
    Ensure v in (0,1], sum_i v = 1 for each (season,week,m).
    We clip to eps for log transform stability, then renormalize.
    This is a numerical device; does NOT change ordering when eps is tiny.
    """
    out = vs.copy()
    out["v"] = out["v"].clip(lower=0.0)
    out["v"] = out["v"].clip(lower=eps)
    # renormalize per (s,t,m)
    g = out.groupby(["season", "week", "m"])["v"].transform("sum")
    out["v"] = out["v"] / g
    return out


def compute_fan_response_YV(dfC: pd.DataFrame, vs: pd.DataFrame) -> pd.DataFrame:
    """
    Merge vote samples with weekly panel and compute:
      u = log(v) - mean_i log(v) within (s,t,m)  (clr)
      YV = u / sd(u) within (s,t,m)

    Returns long-form panel with additional columns:
      m, v, u, YV
    """
    # Keep only pairs in dfC (C_{s,t})
    keys = dfC[["season", "week", "pair_id"]].drop_duplicates()
    vs2 = vs.merge(keys, on=["season", "week", "pair_id"], how="inner")

    # clr
    vs2["logv"] = np.log(vs2["v"])
    mean_logv = vs2.groupby(["season", "week", "m"])["logv"].transform("mean")
    vs2["u"] = vs2["logv"] - mean_logv

    # standardize u within (s,t,m)
    sd_u = vs2.groupby(["season", "week", "m"])["u"].transform("std").replace(0.0, np.nan)
    vs2["YV"] = (vs2["u"]) / sd_u
    vs2["YV"] = vs2["YV"].fillna(0.0)
    vs2["YV"] = vs2["YV"].clip(-6.0, 6.0)

    # Merge back static covariates from dfC
    base_cols = [
        "season", "week", "era", "pair_id", "dancer_id", "celebrity_id",
        "age", "industry", "age_c", "age_c2", "tenure", "muJ_loso", "YJ"
    ]
    base = dfC[base_cols].drop_duplicates(["season", "week", "pair_id"])
    out = vs2.merge(base, on=["season", "week", "pair_id"], how="left")

    return out


# ------------------------------------------------------------
# 7) MI (Rubin) aggregation for fixed effects across m draws
# ------------------------------------------------------------

def rubin_pool(theta_list: List[np.ndarray], cov_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rubin rules:
      bar = mean(theta_m)
      W = mean(cov_m)
      B = sample covariance of theta_m
      T = W + (1 + 1/M) B
    Returns:
      bar, se_total
    """
    M = len(theta_list)
    theta = np.vstack(theta_list)  # (M,p)
    bar = theta.mean(axis=0)

    W = np.mean(np.stack(cov_list, axis=0), axis=0)
    B = np.cov(theta.T, ddof=1) if M > 1 else np.zeros_like(W)
    T = W + (1.0 + 1.0 / M) * B
    se = np.sqrt(np.diag(T))
    return bar, se


# --------------------------------------------
# 8) Fan channel MixedLM with MI over m draws
# --------------------------------------------

def fit_fan_mixedlm_MI(dfV: pd.DataFrame, out_fixed_csv: str, out_re_csv: str,
                       M_fit: int = 300, random_state: int = 0) -> None:
    """
    Fan channel model (Full Feature Version):
    1. Includes 'C(industry)' as requested.
    2. Robust error handling to prevent crashes when convergence is low.
    3. Uses LBFGS -> CG -> Powell cascade to maximize success rate on complex data.
    """
    rng = np.random.default_rng(random_state)

    # 1. 基础清洗
    dfV_clean = dfV.dropna(subset=["m"]).copy()
    all_m = np.sort(dfV_clean["m"].unique())

    if len(all_m) == 0:
        raise ValueError("No m draws in dfV.")

    # 采样
    if M_fit >= len(all_m):
        m_used = all_m
    else:
        m_used = np.sort(rng.choice(all_m, size=M_fit, replace=False))

    print(f"   -> Fitting Fan Model MI on {len(m_used)} draws (With Industry Feature)...")
    print("      (Targeting 50 successful converges)")

    # 🔥 1. 保留所有特征，包括 Industry
    formula = "YV ~ C(era) + C(week) + age_c + age_c2 + C(industry) + tenure + muJ_loso + YJ"

    theta_list, cov_list = [], []
    dancer_re_accum: Dict[str, List[float]] = {}

    success_count = 0
    target_success = M_fit+1

    # 🔥 2. 在循环外初始化变量 (修复 UnboundLocalError 的关键!)
    fe_names = None
    last_params = None

    for i, mm in enumerate(m_used):
        # 早停机制
        if success_count >= target_success:
            print(f"      [Info] Reached {target_success} successful models. Stopping early.")
            break

        # 进度打印
        if i % 10 == 0:
            print(f"      Processing Fan draw {i}/{len(m_used)} (m={mm})... Success: {success_count}")

        dmm = dfV_clean[dfV_clean["m"] == mm].copy()

        # 深度清洗
        dmm = dmm.replace([np.inf, -np.inf], np.nan)
        # 确保 industry 也在检查范围内
        req_cols = ["YV", "era", "week", "age_c", "age_c2", "industry", "tenure", "muJ_loso", "YJ", "dancer_id"]
        dmm = dmm.dropna(subset=[c for c in req_cols if c in dmm.columns])

        # 强制类型转换，防止 Patsy 报错
        dmm["industry"] = dmm["industry"].astype(str).str.strip().str.lower()
        dmm["dancer_id"] = dmm["dancer_id"].astype(str)

        if len(dmm) < 50: continue

        try:
            # 建立模型
            md = MixedLM.from_formula(formula, groups="dancer_id", data=dmm, re_formula="1")

            res = None

            # --- 优化算法级联 (Cascade) ---

            # 1) LBFGS (最快，但对 Industry 这种复杂特征容易失败)
            try:
                start_p = last_params if last_params is not None else None
                res = md.fit(reml=False, method="lbfgs", maxiter=200, start_params=start_p, disp=False)
                if not res.converged: res = None
            except: res = None

            # 2) CG (中等)
            if res is None:
                try:
                    res = md.fit(reml=False, method="cg", maxiter=500, disp=False)
                    if not res.converged: res = None
                except: res = None

            # 3) Powell (慢且稳，专门用来啃 Industry 这块硬骨头)
            # 我们限制 maxiter=300，防止它卡死几小时
            if res is None:
                try:
                    res = md.fit(reml=False, method="powell", maxiter=300, disp=False)
                    if not res.converged: res = None
                except: res = None

            # 如果三招都败了，跳过
            if res is None:
                continue

            # --- 成功后的数据收集 ---

            # 更新热启动参数
            last_params = res.params

            # 收集参数
            theta_list.append(res.fe_params.values)
            cov_list.append(res.cov_params().loc[res.fe_params.index, res.fe_params.index].values)

            # 🔥 3. 只要成功一次，立刻抓取列名 (修复 AttributeError/UnboundError)
            if fe_names is None:
                fe_names = res.fe_params.index.tolist()

            # 收集随机效应
            if hasattr(res, 'random_effects'):
                for dancer, vec in res.random_effects.items():
                    try:
                        val = vec[0] if isinstance(vec, (list, np.ndarray)) else vec.iloc[0]
                        dancer_re_accum.setdefault(dancer, []).append(float(val))
                    except: pass

            success_count += 1

        except Exception:
            continue

    # --- 汇总与保存 ---
    print(f"   -> Fan Pooling results from {success_count} successful draws.")

    # 如果全军覆没
    if len(theta_list) == 0:
        print("      [Error] No Fan models converged. Saving empty files.")
        pd.DataFrame(columns=["term", "coef_MI", "se_MI"]).to_csv(out_fixed_csv, index=False)
        pd.DataFrame(columns=["dancer_id", "bV_mean"]).to_csv(out_re_csv, index=False)
        return

    # 🔥 4. 双重保险：防止 fe_names 依然为空 (虽然理论上不可能了)
    if fe_names is None:
        print("      [Warning] Generating dummy feature names.")
        fe_names = [f"Term_{k}" for k in range(len(theta_list[0]))]

    # Rubin Pooling
    try:
        bar, se = rubin_pool(theta_list, cov_list)
        fe = pd.DataFrame({"term": fe_names, "coef_MI": bar, "se_MI": se})
        fe.to_csv(out_fixed_csv, index=False)
        print(f"      [Success] Saved Fixed Effects to {out_fixed_csv}")
    except Exception as e:
        print(f"      [Error] Pooling failed: {e}")

    # 保存 Dancer Random Effects
    re_rows = []
    for dancer, vals in dancer_re_accum.items():
        re_rows.append({
            "dancer_id": dancer,
            "bV_mean": float(np.mean(vals)),
            "bV_sd_across_m": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n_m": len(vals),
        })

    if len(re_rows) > 0:
        re_df = pd.DataFrame(re_rows).sort_values("bV_mean", ascending=False)
        re_df.to_csv(out_re_csv, index=False)
    else:
        pd.DataFrame(columns=["dancer_id", "bV_mean"]).to_csv(out_re_csv, index=False)

    print(f"      [Done] Fan model saved.")
# ---------------------------------------------------------
# 9) Hazard (weekly elimination) + era interactions (GEE)
# ---------------------------------------------------------

def add_elimination_label(dfC: pd.DataFrame, week_elim_col: Optional[str]) -> pd.DataFrame:
    """
    Create y_elim = 1(i eliminated this week) for i in C_{s,t}.
    Requires a column for elimination week; if missing, user must merge Q2 outputs.

    If week_elim_col is None -> raise with instruction.
    """
    if week_elim_col is None:
        raise ValueError("No week_eliminated column found. "
                         "Please provide a column in cleaned_data.csv or merge Q2 outputs to create it.")

    out = dfC.copy()
    out["week_eliminated"] = pd.to_numeric(out[week_elim_col], errors="coerce")
    # y=1 if current week equals elimination week
    out["y_elim"] = ((out["week"] == out["week_eliminated"]) & out["week_eliminated"].notna()).astype(int)
    # For finalists (censored), week_eliminated may be max-week or NaN; y will be 0.
    return out


def fit_hazard_GEE_MI(dfVH: pd.DataFrame, out_fixed_csv: str, out_re_csv: str,
                      M_fit: int = 300, random_state: int = 0) -> None:
    """
    Discrete-time hazard approximation with GEE.
    Includes 'industry' feature but filters out runs that result in NaN/Inf coefficients.
    """
    rng = np.random.default_rng(random_state)

    # 1. 基础清洗
    df_clean = dfVH.dropna(subset=["m"]).copy()

    K_INDUSTRY = 20

 # ---------------------------------------------------------
  # Fallback: ensure industry_grp exists for hazard stability
  # ---------------------------------------------------------

    if "industry_grp" not in df_clean.columns:

        if "industry" not in df_clean.columns:

            raise KeyError("Neither 'industry_grp' nor 'industry' exists in dfVH; cannot build hazard covariate.")

            K_INDUSTRY = 20  # frequency threshold; tune 10/20/30 if needed
  # normalize raw industry label
        df_clean["industry_clean"] = df_clean["industry"].astype(str).str.strip().str.lower()

  # count on unique pair-week panel to avoid m-replication inflating frequencies

        if all(c in df_clean.columns for c in ["season", "week", "pair_id"]):
                base = df_clean.drop_duplicates(subset=["season", "week", "pair_id"]).copy()
        else:
            base = df_clean.copy()

        freq = base["industry_clean"].value_counts(dropna=False)
        rare = freq[freq < K_INDUSTRY].index
        df_clean["industry_grp"] = df_clean["industry_clean"].where(~df_clean["industry_clean"].isin(rare),
                                                                     other="other")
    all_m = np.sort(df_clean["m"].unique())

    if len(all_m) == 0:
        raise ValueError("No m draws in dfVH.")

    # 采样
    if M_fit >= len(all_m):
        m_used = all_m
    else:
        m_used = np.sort(rng.choice(all_m, size=M_fit, replace=False))

    print(f"   -> Fitting Hazard GEE Model on {len(m_used)} draws (With Industry)...")
    print("      (Targeting 50 successful converges with VALID numbers)")

    # 准备变量
    df_clean["is_percent"] = (df_clean["era"] == 2).astype(int)

    # 🔥 1. 保留 industry 特征
    formula = ("y_elim ~ C(era) + C(week) + YJ + YV + age_c + age_c2 + "
               "C(industry_grp) + tenure + muJ_loso")

    theta_list, cov_list = [], []
    coef_names = None
    dancer_eff_accum: Dict[str, List[float]] = {}

    success_count = 0
    target_success = M_fit+1

    for i, mm in enumerate(m_used):
        if success_count >= target_success:
            print(f"      [Info] Reached {target_success} successful models. Stopping early.")
            break

        dmm = df_clean[df_clean["m"] == mm].copy()

        # 数据清洗
        dmm["industry_grp"] = dmm["industry_grp"].astype(str).str.strip().str.lower().fillna("other")
        dmm["pair_id"] = dmm["pair_id"].astype(str)
        dmm = dmm.replace([np.inf, -np.inf], np.nan)

        # 确保 industry 也在检查范围内
        req_cols = ["y_elim", "YJ", "YV", "age_c", "age_c2", "tenure", "muJ_loso", "industry_grp"]
        dmm = dmm.dropna(subset=[c for c in req_cols if c in dmm.columns])

        # 重置索引，防止预测对齐错误
        dmm = dmm.reset_index(drop=True)

        if len(dmm) < 50: continue

        try:
            # GEE 模型
            model = smf.gee(formula=formula, groups="pair_id", data=dmm,
                            family=Binomial(), cov_struct=Exchangeable())

            # 拟合
            try:
                res = model.fit(maxiter=100)

                # 🔥 2. 关键质量检测：检查系数是否包含 NaN 或 Inf
                params_val = res.params.values
                if np.any(np.isnan(params_val)) or np.any(np.isinf(params_val)):
                    # 如果结果里有 NaN，说明这个样本算废了，直接跳过
                    continue

            except Exception:
                continue

            if coef_names is None:
                coef_names = res.params.index.tolist()

            # 只有通过了质量检测，才收集参数
            theta_list.append(res.params.values)
            cov_list.append(res.cov_params().values)

            # 计算选手残差效应
            try:
                # 直接获取 numpy 数组，不加 .values
                linpred = res.predict(linear=True)

                if len(linpred) == len(dmm):
                    # 再次检查预测值是否有 NaN
                    linpred_arr = np.asarray(linpred)
                    if np.any(np.isnan(linpred_arr)):
                        continue  # 预测值有 NaN 也不行

                    dmm["_eta"] = linpred_arr
                    by_d = dmm.groupby("dancer_id")["_eta"].mean()
                    global_eta = float(np.mean(linpred_arr))

                    for dancer, eta_bar in by_d.items():
                        dancer_eff_accum.setdefault(dancer, []).append(float(eta_bar - global_eta))
            except Exception as e:
                pass

            success_count += 1
            if success_count % 10 == 0:
                print(f"      Processing Hazard draw {i + 1}/{len(m_used)} (m={mm})... Success: {success_count}")

        except Exception as e:
            continue

    print(f"   -> Hazard Pooling results from {success_count} successful draws.")

    if len(theta_list) == 0:
        print("      [Error] No valid models converged. Please check data quality.")
        pd.DataFrame(columns=["term", "coef_MI", "se_MI"]).to_csv(out_fixed_csv, index=False)
        pd.DataFrame(columns=["dancer_id", "bH_eta_mean"]).to_csv(out_re_csv, index=False)
        return

    # 1. 保存 Fixed Effects
    try:
        bar, se = rubin_pool(theta_list, cov_list)
        fe = pd.DataFrame({"term": coef_names, "coef_MI": bar, "se_MI": se})
        fe.to_csv(out_fixed_csv, index=False)
        print(f"      [Success] Saved Fixed Effects (Rows: {len(fe)})")
    except Exception as e:
        print(f"      [Error] Fixed effects pooling failed: {e}")

    # 2. 保存 Dancer Effects
    rows = []
    for dancer, vals in dancer_eff_accum.items():
        if len(vals) > 0:
            rows.append({
                "dancer_id": dancer,
                "bH_eta_mean": float(np.mean(vals)),
                "bH_eta_sd_across_m": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "n_m": len(vals)
            })

    if len(rows) > 0:
        re_df = pd.DataFrame(rows).sort_values("bH_eta_mean", ascending=False)
        re_df.to_csv(out_re_csv, index=False)
        print(f"      [Success] Saved Dancer Effects (Rows: {len(re_df)})")
    else:
        print("      [Warning] No dancer effects could be extracted.")
        pd.DataFrame(columns=["dancer_id", "bH_eta_mean", "bH_eta_sd_across_m", "n_m"]).to_csv(out_re_csv, index=False)

    print(f"      [Done] Hazard model saved.")
# ---------------------------------------------------------
# 10) Quadrant mapping + plot
# ---------------------------------------------------------

def plot_quadrant(dancer_df: pd.DataFrame, out_png: str, label_top: int = 12) -> None:
    """
    Quadrant plot:
      x = bJ (technical value add)
      y = bV_mean (popularity value add)

    Label top dancers by distance from origin.
    """
    d = dancer_df.copy()
    d = d.dropna(subset=["bJ", "bV_mean"])

    x = d["bJ"].values
    y = d["bV_mean"].values
    dist = np.sqrt(x**2 + y**2)
    d["_dist"] = dist
    d_lab = d.sort_values("_dist", ascending=False).head(label_top)

    plt.figure(figsize=(8, 6))
    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)
    plt.scatter(x, y, s=30)

    for _, r in d_lab.iterrows():
        plt.text(r["bJ"], r["bV_mean"], str(r["dancer_id"]), fontsize=8)

    plt.xlabel("Technical Value Add (b_d^(J))")
    plt.ylabel("Popularity Value Add (b_d^(V))")
    plt.title("Pro Dancer Quadrant Mapping")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ---------------------------------------------------------
# 11) Random Forest + PDP + SHAP
# ---------------------------------------------------------

def rf_shap_pipeline(dfVH: pd.DataFrame, out_dir: str, random_state: int = 0) -> None:
    """
    Fit RandomForestClassifier for weekly elimination prediction.
    Features:
      age_c, age_c2, industry dummies, tenure, muJ_loso, YJ, YV, era, week
    Output:
      - feature_importance.csv
      - PDP plots for age
      - SHAP summary plot (if shap available)
    """
    d = dfVH.copy()

    # Basic feature set (you can extend with spline bases bs1..)
    # One-hot encode industry, era, week
    X = pd.get_dummies(
        d[["age_c", "age_c2", "tenure", "muJ_loso", "YJ", "YV", "era", "week", "industry"]],
        columns=["era", "week", "industry"],
        drop_first=True
    )
    y = d["y_elim"].astype(int).values

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
    )

    rf = RandomForestClassifier(
        n_estimators=600,
        random_state=random_state,
        class_weight="balanced_subsample",
        min_samples_leaf=3
    )
    rf.fit(X_train, y_train)

    # Feature importances
    imp = pd.DataFrame({"feature": X.columns, "importance": rf.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    imp.to_csv(os.path.join(out_dir, "14_rf_feature_importance.csv"), index=False)

    # PDP for (centered) age_c (or raw age if you prefer)
    plt.figure()
    PartialDependenceDisplay.from_estimator(rf, X_test, ["age_c"])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "12_pdp_age_hazard.png"), dpi=200)
    plt.close()

    # SHAP summary plot
    if _HAS_SHAP:
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_test)
        # class 1 corresponds to y=1 (eliminated) in binary classification
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        shap.summary_plot(sv, X_test, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "13_rf_shap_summary.png"), dpi=200)
        plt.close()
    else:
        with open(os.path.join(out_dir, "13_rf_shap_summary.txt"), "w", encoding="utf-8") as f:
            f.write("shap is not available. Install with: pip install shap\n")


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv_path = os.path.join(current_dir, "data", "processed_data", "cleaned_data.csv")
    output_folder = os.path.join("data", "Q3data")
    vote_samples_path= os.path.join(current_dir, "data", "Q3data", "vote_samples.csv")
    ap.add_argument("--cleaned_csv", default=default_csv_path,type=str, help="Path to cleaned_data.csv")
    ap.add_argument("--vote_samples", default=vote_samples_path,type=str, help="Path to vote_samples.parquet/csv (Q1 posterior)")
    ap.add_argument("--out_dir", type=str, default=output_folder, help="Output directory")
    ap.add_argument("--M_target", type=int, default=2000, help="Target M from Q1/Q2 definition")
    ap.add_argument("--M_fit", type=int, default=2000, help="Number of m draws used for regression fitting (speed)")
    ap.add_argument("--colmap_json", type=str, default=None,
                    help="Optional JSON string/file path for manual column mapping")
    ap.add_argument("--hot_top_k", type=int, default=3, help="Top-K industries treated as 'hot' in matching bias test")
    ap.add_argument("--spline_df", type=int, default=5, help="df for spline basis (optional features)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading raw data from {args.cleaned_csv}...")
    df_wide = pd.read_csv(args.cleaned_csv)

    # 2. 宽表转长表 (Wide -> Long)
    # 这一步会自动生成 'week', 'season', 'pair_id' 等列
    try:
        # 1. 使用 Q1 识别周次
        weeks = Q1.detect_weeks(df_wide, col_suffix="_total")
        base = Q1.build_base_table(df_wide, weeks, score_template=Q1.SCORE_TEMPLATE)
        df_long = Q1.wide_to_long(base, weeks, score_template=Q1.SCORE_TEMPLATE)

        # ================= 🔥 关键修复：添加 eliminated_week =================
        # 定义我们需要保留的静态列 (增加了 eliminated_week)
        static_cols_to_keep = [
            "celebrity_age_during_season",
            "celebrity_industry",
            "celebrity_homestate",
            "celebrity_homecountry_region",
            "eliminated_week"  # <--- 新增这一行！
        ]

        # 检查原始数据里有哪些列是存在的
        valid_static_cols = [c for c in static_cols_to_keep if c in df_wide.columns]

        # 提取静态特征表 (按 season 和 celebrity_name 去重)
        if valid_static_cols:
            print(f"Restoring static columns: {valid_static_cols}")
            # 注意：eliminated_week 在原始数据中可能每行都有，去重时需要保留
            static_df = df_wide[["season", "celebrity_name"] + valid_static_cols].drop_duplicates()

            # 将静态特征合并进长表
            df_long = pd.merge(df_long, static_df, on=["season", "celebrity_name"], how="left")
        # =====================================================================

        # 2. 列名适配修复
        # 把 'J' 改名为 'wt_avg'
        if 'J' in df_long.columns:
            df_long.rename(columns={'J': 'wt_avg'}, inplace=True)

        # 🔥 把 'eliminated_week' 改名为 'week_eliminated' 以便后续代码识别
        if 'eliminated_week' in df_long.columns:
            print("Renaming 'eliminated_week' to 'week_eliminated'...")
            df_long.rename(columns={'eliminated_week': 'week_eliminated'}, inplace=True)

        # 3. 只保留 active 的行
        df_raw = df_long[df_long["active"] == True].copy()

        # 4. 构造 pair_id
        df_raw["pair_id"] = (
                df_raw["season"].astype(int).astype(str) + "||" +
                df_raw["celebrity_name"].astype(str) + "||" +
                df_raw["ballroom_partner"].astype(str)
        )

        logging.info(f"Conversion successful. New Long Format Shape: {df_raw.shape}")

    except Exception as e:
        logging.error(f"Failed to convert Wide to Long format: {e}")
        raise RuntimeError("Data conversion failed. Please check input format.")
    # Column mapping
    if args.colmap_json is not None:
        # If path exists, load file; else interpret as JSON string
        if os.path.exists(args.colmap_json):
            with open(args.colmap_json, "r", encoding="utf-8") as f:
                cm_dict = json.load(f)
        else:
            cm_dict = json.loads(args.colmap_json)
        cm = ColMap(**cm_dict)
    else:
        cm = infer_columns(df_raw)

    # Build panel and features
    dfC = build_panel_weekly(df_raw, cm)
    df_profile, dfC_aug = compute_pro_dancer_profile(dfC)
    dfC_aug = add_age_nonlinearity(dfC_aug, spline_df=args.spline_df)

    # Save panel
    dfC_aug.to_parquet(os.path.join(args.out_dir, "01_panel_weekly.parquet"), index=False)
    df_profile.to_csv(os.path.join(args.out_dir, "02_pro_dancer_profile.csv"), index=False)

    # Matching bias tests
    matching_bias_tests(
        dfC_aug,
        out_txt_path=os.path.join(args.out_dir, "03_matching_bias_tests.txt"),
        hot_top_k=args.hot_top_k
    )

    # Judges mixed model
    fit_mixedlm_judges(
        dfC_aug,
        out_fixed_csv=os.path.join(args.out_dir, "04_mixed_judges_fixed.csv"),
        out_re_csv=os.path.join(args.out_dir, "05_mixed_judges_dancer_effects.csv")
    )

    # If no vote samples -> stop here
    if args.vote_samples is None:
        print("No vote_samples provided. Finished judges + features + matching bias tests.")
        return

    # Vote samples pipeline
    vs = load_vote_samples(args.vote_samples)
    vs = validate_and_fix_simplex(vs)

    # Compute fan response YV
    dfV = compute_fan_response_YV(dfC_aug, vs)

    # Create elimination label (requires week_eliminated)
    dfC_elim = add_elimination_label(dfC_aug, cm.week_eliminated)
    # Merge y_elim to dfV (by season,week,pair_id)
    dfV = dfV.merge(dfC_elim[["season", "week", "pair_id", "y_elim"]], on=["season", "week", "pair_id"], how="left")
    dfV["y_elim"] = dfV["y_elim"].fillna(0).astype(int)

    # Fan mixed model MI
    fit_fan_mixedlm_MI(
        dfV,
        out_fixed_csv=os.path.join(args.out_dir, "06_mixed_fan_fixed_MI.csv"),
        out_re_csv=os.path.join(args.out_dir, "07_mixed_fan_dancer_MI.csv"),
        M_fit=args.M_fit
    )

    # Hazard model MI (GEE approximation)
    fit_hazard_GEE_MI(
        dfV,
        out_fixed_csv=os.path.join(args.out_dir, "08_hazard_fixed_MI.csv"),
        out_re_csv=os.path.join(args.out_dir, "09_hazard_dancer_MI.csv"),
        M_fit=args.M_fit
    )

    # Quadrant mapping (merge dancer effects)
    bJ = pd.read_csv(os.path.join(args.out_dir, "05_mixed_judges_dancer_effects.csv"))
    bV = pd.read_csv(os.path.join(args.out_dir, "07_mixed_fan_dancer_MI.csv"))
    dancer_quad = df_profile.merge(bJ, on="dancer_id", how="left").merge(bV, on="dancer_id", how="left")
    dancer_quad.to_csv(os.path.join(args.out_dir, "10_quadrant_data.csv"), index=False)
    plot_quadrant(dancer_quad, out_png=os.path.join(args.out_dir, "10_quadrant_plot.png"))

    # Random forest + PDP + SHAP
    # Use one m draw (median) for ML demonstration (or you can filter dfV to a chosen mm)
    mm0 = np.median(dfV["m"].unique()).astype(int)
    dfVH_one = dfV[dfV["m"] == mm0].copy()
    rf_shap_pipeline(dfVH_one, out_dir=args.out_dir)

    print("Q3 coding pipeline completed. Outputs saved to:", args.out_dir)


if __name__ == "__main__":
    main()
