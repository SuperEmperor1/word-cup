"""加权 Elo 评分体系 (World Football Elo Ratings 方法)。

R' = R + K * G * (W - W_e)
  K   : 赛事重要性 (世界杯 60 / 洲际杯 50 / 预选赛 40 / 友谊赛 20)
  G   : 净胜球倍增  1 球=1, 2 球=1.5, N>=3 球=(11+N)/8
  W_e : 期望胜率 = 1 / (1 + 10^(-dr/400)), 非中立场地主队 dr 加 HOME_ADV
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HOME_ADV = 80.0   # 主场优势折算的 Elo 分
BASE_RATING = 1500.0


def _goal_multiplier(margin: float) -> float:
    m = abs(margin)
    if m <= 1:
        return 1.0
    if m == 2:
        return 1.5
    return (11.0 + m) / 8.0


def expected_score(elo_home: float, elo_away: float, neutral: bool) -> float:
    dr = elo_home - elo_away + (0.0 if neutral else HOME_ADV)
    return 1.0 / (1.0 + 10.0 ** (-dr / 400.0))


def compute_elo_history(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """按时间顺序回放全部已完赛比赛, 计算每场赛前 Elo。

    返回 (带 elo_home/elo_away/elo_exp 列的副本, 最终评分表)。
    未完赛行也会得到其时点的赛前 Elo (即当前最新评分)。
    """
    ratings: dict[str, float] = {}
    n = len(df)
    elo_h = np.empty(n)
    elo_a = np.empty(n)
    exp_h = np.empty(n)

    home = df["home_team"].to_numpy()
    away = df["away_team"].to_numpy()
    hs = df["home_score"].to_numpy()
    as_ = df["away_score"].to_numpy()
    neutral = df["neutral"].to_numpy()
    kfac = df["importance"].to_numpy()
    is_played = df["played"].to_numpy()

    for i in range(n):
        rh = ratings.get(home[i], BASE_RATING)
        ra = ratings.get(away[i], BASE_RATING)
        elo_h[i], elo_a[i] = rh, ra
        we = expected_score(rh, ra, neutral[i])
        exp_h[i] = we
        if not is_played[i]:
            continue
        margin = hs[i] - as_[i]
        w = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
        delta = kfac[i] * _goal_multiplier(margin) * (w - we)
        ratings[home[i]] = rh + delta
        ratings[away[i]] = ra - delta

    out = df.copy()
    out["elo_home"] = elo_h
    out["elo_away"] = elo_a
    out["elo_exp_home"] = exp_h
    return out, ratings
