"""端到端管线: 数据 -> Elo -> 特征 -> 训练/预测。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import K_WORLD_CUP, load_matches, outcome_labels
from .dixon_coles import DixonColes
from .elo import compute_elo_history
from .features import build_features
from .model import Ensemble


def prepare(path: str | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
    """加载数据并构建 Elo + 全部特征。返回 (特征表, 当前 Elo 评分表)。"""
    df = load_matches(path) if path else load_matches()
    df, ratings = compute_elo_history(df)
    return build_features(df), ratings


def hypothetical_rows(raw: pd.DataFrame, pairs: list[tuple[str, str]],
                      date: pd.Timestamp, neutral: bool = True) -> pd.DataFrame:
    """为任意对阵生成"截至 date"的特征行 (用于淘汰赛等未排程比赛)。"""
    fix = pd.DataFrame({
        "date": date, "home_team": [p[0] for p in pairs],
        "away_team": [p[1] for p in pairs],
        "home_score": np.nan, "away_score": np.nan,
        "tournament": "FIFA World Cup", "city": "", "country": "",
        "neutral": neutral, "played": False, "importance": K_WORLD_CUP,
    })
    base = raw[raw["date"] < date]
    alld = pd.concat([base, fix], ignore_index=True).sort_values(
        "date", kind="stable").reset_index(drop=True)
    alld, _ = compute_elo_history(alld)
    feat = build_features(alld)
    return feat[~feat["played"]].tail(len(pairs)).reset_index(drop=True)


def train_full(feat: pd.DataFrame, valid_years: int = 2) -> Ensemble:
    """用全部已完赛数据训练线上模型。

    GBM 用 2002 年以来除最近 valid_years 年外的数据训练, 最近数据调融合
    参数; DC 拟合至最新。最后 GBM 在全样本上重训 (参数沿用)。
    """
    done = feat[feat["played"]].reset_index(drop=True)
    y = outcome_labels(done)
    cut = done["date"].max() - pd.DateOffset(years=valid_years)
    tr = (done["date"] >= "2002-01-01") & (done["date"] < cut)
    va = done["date"] >= cut

    ens = Ensemble()
    ens.fit_gbm(done[tr], y[tr])
    ens.dc = DixonColes().fit(done[done["date"] < cut])
    ens.tune(ens.dc_probs_rows(done[va]), ens.gbm_probs(done[va]), y[va])

    # 调参完成后用全量数据重训两个组件
    full = done["date"] >= "2002-01-01"
    ens.fit_gbm(done[full], y[full])
    ens.dc = DixonColes().fit(done)
    return ens
