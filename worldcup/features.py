"""防泄漏的滚动特征工程。

所有特征只使用该场比赛之前的信息 (单次时间正序扫描), 涵盖:
  - 实力: 赛前 Elo (双方/差值/和值), Elo 期望胜率
  - 状态: 指数加权近 10 场积分率、场均进球/失球
  - 日程: 休息天数 (疲劳), 近 30 天比赛密度
  - 交锋: 历史 H2H 场均净胜球 (近 10 次)
  - 情境: 中立场地、赛事重要性、是否大赛淘汰期
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

FORM_N = 10          # 状态窗口场次
FORM_DECAY = 0.85    # 指数衰减 (最近一场权重最高)
H2H_N = 10

FEATURE_COLS = [
    "elo_home", "elo_away", "elo_diff", "elo_sum", "elo_exp_home",
    "form_pts_h", "form_pts_a", "gf_h", "ga_h", "gf_a", "ga_a",
    "rest_h", "rest_a", "density_h", "density_a",
    "h2h_gd", "neutral_f", "importance",
]


def _form_stats(hist: deque) -> tuple[float, float, float]:
    """指数加权 (积分率, 场均进球, 场均失球); 无历史时返回先验。"""
    if not hist:
        return 1.0, 1.3, 1.3
    wts = np.array([FORM_DECAY ** k for k in range(len(hist) - 1, -1, -1)])
    arr = np.array(hist)  # 列: pts, gf, ga
    s = wts.sum()
    return (float(wts @ arr[:, 0] / s), float(wts @ arr[:, 1] / s),
            float(wts @ arr[:, 2] / s))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df 须已含 elo_home/elo_away/elo_exp_home (来自 elo.compute_elo_history)。

    对未完赛行同样生成特征 (使用截至当前的全部历史), 不更新状态。
    """
    n = len(df)
    home = df["home_team"].to_numpy()
    away = df["away_team"].to_numpy()
    hs = df["home_score"].to_numpy(float)
    as_ = df["away_score"].to_numpy(float)
    dates = df["date"].to_numpy("datetime64[D]")
    is_played = df["played"].to_numpy()

    form: dict[str, deque] = defaultdict(lambda: deque(maxlen=FORM_N))
    last_date: dict[str, np.datetime64] = {}
    recent_dates: dict[str, deque] = defaultdict(lambda: deque(maxlen=15))
    h2h: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=H2H_N))

    cols = {c: np.empty(n) for c in
            ("form_pts_h", "form_pts_a", "gf_h", "ga_h", "gf_a", "ga_a",
             "rest_h", "rest_a", "density_h", "density_a", "h2h_gd")}

    for i in range(n):
        h, a, d = home[i], away[i], dates[i]
        ph, gfh, gah = _form_stats(form[h])
        pa, gfa, gaa = _form_stats(form[a])
        cols["form_pts_h"][i], cols["gf_h"][i], cols["ga_h"][i] = ph, gfh, gah
        cols["form_pts_a"][i], cols["gf_a"][i], cols["ga_a"][i] = pa, gfa, gaa

        cols["rest_h"][i] = min(int((d - last_date[h]) / np.timedelta64(1, "D")), 60) \
            if h in last_date else 30
        cols["rest_a"][i] = min(int((d - last_date[a]) / np.timedelta64(1, "D")), 60) \
            if a in last_date else 30
        cols["density_h"][i] = sum(1 for x in recent_dates[h]
                                   if (d - x) / np.timedelta64(1, "D") <= 30)
        cols["density_a"][i] = sum(1 for x in recent_dates[a]
                                   if (d - x) / np.timedelta64(1, "D") <= 30)

        key = (h, a) if h < a else (a, h)
        past = h2h[key]
        if past:
            gd = float(np.mean(past))
            cols["h2h_gd"][i] = gd if h < a else -gd
        else:
            cols["h2h_gd"][i] = 0.0

        if not is_played[i]:
            continue
        # 更新状态 (仅完赛)
        margin = hs[i] - as_[i]
        pts_h = 3.0 if margin > 0 else (1.0 if margin == 0 else 0.0)
        form[h].append((pts_h, hs[i], as_[i]))
        form[a].append((3.0 - pts_h if margin != 0 else 1.0, as_[i], hs[i]))
        last_date[h] = last_date[a] = d
        recent_dates[h].append(d)
        recent_dates[a].append(d)
        h2h[key].append(margin if h < a else -margin)

    out = df.copy()
    for c, v in cols.items():
        out[c] = v
    out["elo_diff"] = out["elo_home"] - out["elo_away"] \
        + np.where(out["neutral"], 0.0, 80.0)
    out["elo_sum"] = out["elo_home"] + out["elo_away"]
    out["neutral_f"] = out["neutral"].astype(float)
    return out
