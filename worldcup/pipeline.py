"""端到端管线: 数据 -> Elo -> 特征 (75 维) -> 训练/预测。

外部商业数据 (data/external/) 自动发现并通过两条通路生效:
  1. 特征通路: as-of 对齐进特征表, 历史快照积累后被 GBM 学习;
  2. 即时调整通路: 伤停/状态 -> Elo 当量修正, 仅作用于未来比赛的预测行
     (历史训练行不修正, 避免与特征通路重复计入)。
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

from .data import (CITY_ALTITUDES, K_WORLD_CUP, load_centroids,
                   load_goal_events, load_matches, outcome_labels)
from .dixon_coles import DixonColes
from .elo import compute_elo_history, expected_score
from .external import ExternalData
from .features import build_features
from .model import Ensemble


def _cache_key() -> str:
    import hashlib
    from .data import DATA_PATH
    from .external import EXTERNAL_DIR
    parts = []
    for p in [DATA_PATH] + sorted(
            os.path.join(EXTERNAL_DIR, f)
            for f in (os.listdir(EXTERNAL_DIR)
                      if os.path.isdir(EXTERNAL_DIR) else [])
            if f.endswith(".csv")):
        st = os.stat(p)
        parts.append(f"{p}:{st.st_size}:{st.st_mtime_ns}")
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:16]


def prepare(path: str | None = None,
            external: ExternalData | None = None,
            use_cache: bool = True) -> tuple[pd.DataFrame, dict[str, float]]:
    """加载全部数据源并构建 Elo + 全部特征。返回 (特征表, 当前 Elo)。

    结果按数据文件指纹缓存到磁盘 (数据或外部快照变化自动失效)。
    """
    cache = None
    if use_cache and path is None:
        cache = os.path.join(os.path.dirname(__file__), "..", "data",
                             f".cache_feat_{_cache_key()}.pkl")
        if os.path.exists(cache):
            with open(cache, "rb") as f:
                return pickle.load(f)
    if external is None:
        external = ExternalData.discover()
    df = load_matches(path) if path else load_matches()
    df, ratings = compute_elo_history(df)
    feat = build_features(df, goal_events=load_goal_events(),
                          centroids=load_centroids(), external=external,
                          city_alt=CITY_ALTITUDES)
    if cache:
        for old in os.listdir(os.path.dirname(cache)):
            if old.startswith(".cache_feat_"):
                os.remove(os.path.join(os.path.dirname(cache), old))
        with open(cache, "wb") as f:
            pickle.dump((feat, ratings), f)
    return feat, ratings


def apply_external_adjustments(rows: pd.DataFrame,
                               external: ExternalData | None) -> pd.DataFrame:
    """即时调整通路: 伤停/状态的 Elo 当量修正, 并重算派生列。"""
    if external is None or external.fm is None:
        return rows
    rows = rows.copy()
    d = rows["date"].to_numpy("datetime64[D]")
    adj_h = np.array([external.elo_adjustment(t, dd)
                      for t, dd in zip(rows["home_team"], d)])
    adj_a = np.array([external.elo_adjustment(t, dd)
                      for t, dd in zip(rows["away_team"], d)])
    rows["elo_home"] = rows["elo_home"] + adj_h
    rows["elo_away"] = rows["elo_away"] + adj_a
    rows["elo_diff"] = rows["elo_home"] - rows["elo_away"] \
        + np.where(rows["neutral"], 0.0, 80.0)
    rows["elo_sum"] = rows["elo_home"] + rows["elo_away"]
    rows["elo_exp_home"] = [
        expected_score(eh, ea, ne) for eh, ea, ne in
        zip(rows["elo_home"], rows["elo_away"], rows["neutral"])]
    return rows


def hypothetical_rows(raw: pd.DataFrame, pairs: list[tuple[str, str]],
                      date: pd.Timestamp, neutral: bool = True,
                      country: str = "United States",
                      external: ExternalData | None = None) -> pd.DataFrame:
    """为任意对阵生成"截至 date"的特征行 (用于淘汰赛等未排程比赛)。"""
    if external is None:
        external = ExternalData.discover()
    fix = pd.DataFrame({
        "date": date, "home_team": [p[0] for p in pairs],
        "away_team": [p[1] for p in pairs],
        "home_score": np.nan, "away_score": np.nan,
        "tournament": "FIFA World Cup", "city": "", "country": country,
        "neutral": neutral, "played": False, "importance": K_WORLD_CUP,
    })
    base = raw[raw["date"] < date]
    alld = pd.concat([base, fix], ignore_index=True).sort_values(
        "date", kind="stable").reset_index(drop=True)
    alld, _ = compute_elo_history(alld)
    feat = build_features(alld, goal_events=load_goal_events(),
                          centroids=load_centroids(), external=external,
                          city_alt=CITY_ALTITUDES)
    rows = feat[~feat["played"]].tail(len(pairs)).reset_index(drop=True)
    return apply_external_adjustments(rows, external)


def train_full(feat: pd.DataFrame, valid_years: int = 2) -> Ensemble:
    """用全部已完赛数据训练线上模型。

    基模型用 2002 年以来除最近 valid_years 年外的数据训练, 最近数据训练
    元学习器与温度; 之后基模型在全样本重训 (元学习器系数沿用)。
    """
    done = feat[feat["played"]].reset_index(drop=True)
    y = outcome_labels(done)
    cut = done["date"].max() - pd.DateOffset(years=valid_years)
    tr = (done["date"] >= "2002-01-01") & (done["date"] < cut)
    va = done["date"] >= cut

    ens = Ensemble()
    ens.fit_base(done[tr], y[tr])
    ens.dc = DixonColes().fit(done[done["date"] < cut])
    ens.fit_meta(done[va].reset_index(drop=True), y[va])

    # 调参完成后用全量数据重训基模型
    full = done["date"] >= "2002-01-01"
    ens.fit_base(done[full], y[full])
    ens.dc = DixonColes().fit(done)
    return ens
