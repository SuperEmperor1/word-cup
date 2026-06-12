"""防泄漏的滚动特征工程 (v2, 52 维)。

所有特征只使用该场比赛之前的信息 (单次时间正序扫描)。特征族:

实力评级
  - 赛前 Elo (双方/差值/和值)、Elo 期望胜率
  - Elo 一年趋势 (上升期/衰退期)
  - 分数驱动攻防评级 (score-driven dynamic Poisson, Koopman-Lit 简化版):
    λ̂ = exp(c + att_i + def_j), 每场按 (实际进球 − λ̂) 在线更新
状态与赛程强度
  - 指数加权近 10 场与近 5 场积分率、场均进球/失球
  - 对手平均 Elo (赛程强度)、Elo 调整超额表现 (W − W_e 均值)
  - 不败场次连击、近 10 场零封率
日程与地理
  - 休息天数、近 30 天比赛密度
  - 旅行距离 (球队所在国质心 -> 举办国质心, haversine)、时区差
球员级 (来自进球明细数据)
  - 75 分钟后进球/失球占比 (体能与终局强度)
  - 点球依赖度、射手集中度 HHI (近 2 年, 阵容深度代理)
经验与情境
  - 近 8 年大赛 (世界杯/洲际杯) 场次
  - 洲足联归属 (由参加的洲际赛事自动推断)、是否同洲对决
  - 交锋历史净胜球、中立场地、赛事重要性
"""
from __future__ import annotations

from collections import defaultdict, deque
from math import asin, cos, radians, sin, sqrt

import numpy as np
import pandas as pd

FORM_N = 10
FORM_DECAY = 0.85
H2H_N = 10
GOALEV_N = 25          # 球员级特征回看场次
SD_K = 0.10            # 分数驱动评级学习率
SD_BASE = 0.20         # log 基准进球率
LATE_MIN = 75

CONFED = {"UEFA": 1, "CONMEBOL": 2, "CONCACAF": 3, "CAF": 4, "AFC": 5, "OFC": 6}
_CONFED_HINTS = [
    ("UEFA", "UEFA"), ("Copa América", "CONMEBOL"), ("CONMEBOL", "CONMEBOL"),
    ("CONCACAF", "CONCACAF"), ("Gold Cup", "CONCACAF"),
    ("African", "CAF"), ("Africa", "CAF"), ("CECAFA", "CAF"), ("COSAFA", "CAF"),
    ("WAFF", "AFC"), ("AFC", "AFC"), ("Asian", "AFC"), ("Gulf Cup", "AFC"),
    ("OFC", "OFC"), ("Oceania", "OFC"), ("Pacific", "OFC"),
]

FEATURE_COLS = [
    # 实力评级
    "elo_home", "elo_away", "elo_diff", "elo_sum", "elo_exp_home",
    "elo_trend_h", "elo_trend_a",
    "sd_att_h", "sd_def_h", "sd_att_a", "sd_def_a", "sd_xg_h", "sd_xg_a",
    # 状态与赛程强度
    "form_pts_h", "form_pts_a", "form5_pts_h", "form5_pts_a",
    "gf_h", "ga_h", "gf_a", "ga_a",
    "opp_elo_h", "opp_elo_a", "overperf_h", "overperf_a",
    "unbeaten_h", "unbeaten_a", "cleansheet_h", "cleansheet_a",
    # 日程与地理
    "rest_h", "rest_a", "density_h", "density_a",
    "travel_h", "travel_a", "travel_diff", "tz_h", "tz_a",
    # 球员级
    "late_goal_h", "late_goal_a", "late_concede_h", "late_concede_a",
    "pen_share_h", "pen_share_a", "scorer_hhi_h", "scorer_hhi_a",
    "star_form_h", "star_form_a",
    # 经验与情境
    "major_exp_h", "major_exp_a", "confed_h", "confed_a", "same_confed",
    "h2h_gd", "neutral_f", "importance",
    # 外部商业数据 (FM2026 阵容 / 转会身价 / 博彩市场), 缺失时为 NaN
    "fm_xi_h", "fm_xi_a", "fm_xi_diff", "fm_depth_h", "fm_depth_a",
    "fm_gk_h", "fm_gk_a", "fm_age_h", "fm_age_a",
    "fm_stardep_h", "fm_stardep_a", "fm_cond_h", "fm_cond_a",
    "fm_inj_h", "fm_inj_a",
    "mv_log_h", "mv_log_a", "mv_diff",
    "mkt_ph", "mkt_pd", "mkt_pa",
]

STAR_WINDOW = 120        # 核心射手「近期」窗口 (天)
STAR_BASELINE = 730      # 基准期 (天)


def _haversine(lat1, lon1, lat2, lon2):
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 6371.0 * 2 * asin(sqrt(a))


def _form_stats(hist: deque, n: int | None = None) -> tuple[float, float, float]:
    rows = list(hist)[-n:] if n else list(hist)
    if not rows:
        return 1.0, 1.3, 1.3
    wts = np.array([FORM_DECAY ** k for k in range(len(rows) - 1, -1, -1)])
    arr = np.array(rows)
    s = wts.sum()
    return (float(wts @ arr[:, 0] / s), float(wts @ arr[:, 1] / s),
            float(wts @ arr[:, 2] / s))


def _confed_of(tournament: str) -> str | None:
    for hint, conf in _CONFED_HINTS:
        if hint in tournament:
            return conf
    return None


class _GoalEventIndex:
    """(date, home, away) -> 该场进球事件, 供时间正序扫描查询。"""

    def __init__(self, goals: pd.DataFrame | None):
        self.idx = defaultdict(list)
        if goals is None:
            return
        for r in goals.itertuples(index=False):
            self.idx[(r.date, r.home_team, r.away_team)].append(
                (r.team, r.minute, r.penalty, r.scorer))

    def get(self, date_str, home, away):
        return self.idx.get((date_str, home, away), ())


def build_features(df: pd.DataFrame,
                   goal_events: pd.DataFrame | None = None,
                   centroids: dict | None = None,
                   external=None) -> pd.DataFrame:
    """df 须已含 elo_home/elo_away/elo_exp_home (来自 elo.compute_elo_history)。

    external: worldcup.external.ExternalData, 缺省时外部特征列为 NaN
    (HistGradientBoosting 原生处理缺失, 历史无快照不影响其余特征的学习)。
    """
    n = len(df)
    home = df["home_team"].to_numpy()
    away = df["away_team"].to_numpy()
    hs = df["home_score"].to_numpy(float)
    as_ = df["away_score"].to_numpy(float)
    dates = df["date"].to_numpy("datetime64[D]")
    date_strs = df["date"].dt.strftime("%Y-%m-%d").to_numpy()
    country = df["country"].to_numpy()
    tournament = df["tournament"].to_numpy()
    imp = df["importance"].to_numpy()
    elo_h_arr = df["elo_home"].to_numpy()
    elo_a_arr = df["elo_away"].to_numpy()
    exp_h_arr = df["elo_exp_home"].to_numpy()
    neutral = df["neutral"].to_numpy()
    is_played = df["played"].to_numpy()
    gev = _GoalEventIndex(goal_events)
    centroids = centroids or {}

    # -------- 球队状态 --------
    form: dict[str, deque] = defaultdict(lambda: deque(maxlen=FORM_N))
    sched: dict[str, deque] = defaultdict(lambda: deque(maxlen=FORM_N))  # (opp_elo, w-we, clean)
    last_date: dict = {}
    recent_dates: dict[str, deque] = defaultdict(lambda: deque(maxlen=15))
    h2h: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=H2H_N))
    elo_hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=60))
    sd_att: dict[str, float] = defaultdict(float)
    sd_def: dict[str, float] = defaultdict(float)
    unbeaten: dict[str, int] = defaultdict(int)
    goalrec: dict[str, deque] = defaultdict(lambda: deque(maxlen=GOALEV_N))
    scorers: dict[str, deque] = defaultdict(deque)   # (date, scorer)
    majors: dict[str, deque] = defaultdict(deque)    # 大赛日期
    confed: dict[str, int] = defaultdict(int)

    _derived = ("elo_home", "elo_away", "elo_diff", "elo_sum",
                "elo_exp_home", "neutral_f", "importance",
                "fm_xi_diff", "mv_diff", "travel_diff")
    cols = {c: (np.full(n, np.nan) if c.startswith(("fm_", "mv_", "mkt_"))
                else np.zeros(n))
            for c in FEATURE_COLS if c not in _derived}

    def elo_then(t, d):
        """t 在 d-365 天时的 Elo (无记录则取现值 -> 趋势 0)。"""
        cut = d - np.timedelta64(365, "D")
        val = None
        for dd, e in elo_hist[t]:
            if dd <= cut:
                val = e
            else:
                break
        return val

    def goal_features(t):
        rec = goalrec[t]
        if not rec:
            return 0.25, 0.25, 0.10, 0.15
        arr = np.array(rec, float)  # gf, late_gf, ga, late_ga, pens
        gf, lgf, ga, lga, pen = arr.sum(0)
        return (lgf / gf if gf else 0.25, lga / ga if ga else 0.25,
                pen / gf if gf else 0.10, 0.0)

    def scorer_stats(t, d):
        """(射手集中度 HHI, 核心射手近期状态)。

        star_form: 近 2 年队内前 3 射手在最近 120 天的进球占其总进球的
        比例, 除以时间占比基准 —— >1 核心射手火热, <<1 核心哑火/缺阵。
        这是球员可用性与状态的内生代理 (无需外部数据)。
        """
        dq = scorers[t]
        cut = d - np.timedelta64(STAR_BASELINE, "D")
        while dq and dq[0][0] < cut:
            dq.popleft()
        if len(dq) < 5:
            return 0.15, 1.0
        cnt = defaultdict(int)
        for _, s in dq:
            cnt[s] += 1
        tot = sum(cnt.values())
        hhi = sum((v / tot) ** 2 for v in cnt.values())
        top3 = {s for s, _ in sorted(cnt.items(), key=lambda kv: -kv[1])[:3]}
        star_tot = sum(cnt[s] for s in top3)
        recent_cut = d - np.timedelta64(STAR_WINDOW, "D")
        star_recent = sum(1 for dd, s in dq if s in top3 and dd >= recent_cut)
        form = (star_recent / star_tot) / (STAR_WINDOW / STAR_BASELINE) \
            if star_tot else 1.0
        return hhi, min(form, 3.0)

    for i in range(n):
        h, a, d = home[i], away[i], dates[i]
        eh, ea = elo_h_arr[i], elo_a_arr[i]

        for side, t, opp_elo in (("h", h, ea), ("a", a, eh)):
            p10, gf, ga = _form_stats(form[t])
            p5, _, _ = _form_stats(form[t], 5)
            cols[f"form_pts_{side}"][i] = p10
            cols[f"form5_pts_{side}"][i] = p5
            cols[f"gf_{side}"][i], cols[f"ga_{side}"][i] = gf, ga
            sc = sched[t]
            cols[f"opp_elo_{side}"][i] = np.mean([r[0] for r in sc]) if sc else 1500.0
            cols[f"overperf_{side}"][i] = np.mean([r[1] for r in sc]) if sc else 0.0
            cols[f"cleansheet_{side}"][i] = np.mean([r[2] for r in sc]) if sc else 0.3
            cols[f"unbeaten_{side}"][i] = min(unbeaten[t], 15)
            prev = elo_then(t, d)
            now = eh if side == "h" else ea
            cols[f"elo_trend_{side}"][i] = now - prev if prev is not None else 0.0
            cols[f"rest_{side}"][i] = min(int((d - last_date[t]) / np.timedelta64(1, "D")), 60) \
                if t in last_date else 30
            cols[f"density_{side}"][i] = sum(
                1 for x in recent_dates[t] if (d - x) / np.timedelta64(1, "D") <= 30)
            # 地理
            tc, mc = centroids.get(t), centroids.get(country[i])
            if tc and mc:
                cols[f"travel_{side}"][i] = _haversine(*tc, *mc)
                cols[f"tz_{side}"][i] = abs(tc[1] - mc[1]) / 15.0
            else:
                cols[f"travel_{side}"][i] = np.nan
                cols[f"tz_{side}"][i] = np.nan
            # 球员级
            lg, lc, ps, _ = goal_features(t)
            cols[f"late_goal_{side}"][i] = lg
            cols[f"late_concede_{side}"][i] = lc
            cols[f"pen_share_{side}"][i] = ps
            hhi, star = scorer_stats(t, d)
            cols[f"scorer_hhi_{side}"][i] = hhi
            cols[f"star_form_{side}"][i] = star
            # 外部商业数据 (as-of 防泄漏查询)
            if external is not None:
                for k, v in external.fm_features(t, d).items():
                    cols[f"{k}_{side}"][i] = v
                cols[f"mv_log_{side}"][i] = external.mv_log(t, d)
            # 经验 / 洲足联
            mj = majors[t]
            cut8 = d - np.timedelta64(365 * 8, "D")
            while mj and mj[0] < cut8:
                mj.popleft()
            cols[f"major_exp_{side}"][i] = len(mj)
            cols[f"confed_{side}"][i] = confed[t]

        # 分数驱动攻防评级 (赛前值)
        cols["sd_att_h"][i], cols["sd_def_h"][i] = sd_att[h], sd_def[h]
        cols["sd_att_a"][i], cols["sd_def_a"][i] = sd_att[a], sd_def[a]
        lam_h = np.exp(SD_BASE + sd_att[h] + sd_def[a])
        lam_a = np.exp(SD_BASE + sd_att[a] + sd_def[h])
        cols["sd_xg_h"][i], cols["sd_xg_a"][i] = lam_h, lam_a

        cols["same_confed"][i] = float(confed[h] == confed[a] and confed[h] > 0)

        if external is not None:
            mkt = external.match_odds(date_strs[i], h, a)
            cols["mkt_ph"][i], cols["mkt_pd"][i], cols["mkt_pa"][i] = mkt

        key = (h, a) if h < a else (a, h)
        past = h2h[key]
        cols["h2h_gd"][i] = (float(np.mean(past)) * (1 if h < a else -1)) if past else 0.0

        # 洲足联推断 (赛前即可更新, 与结果无关)
        cf = _confed_of(tournament[i])
        if cf:
            confed[h] = confed[h] or CONFED[cf]
            confed[a] = confed[a] or CONFED[cf]

        if not is_played[i]:
            continue

        # -------- 用本场结果更新状态 --------
        margin = hs[i] - as_[i]
        pts_h = 3.0 if margin > 0 else (1.0 if margin == 0 else 0.0)
        w_h = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
        we = exp_h_arr[i]
        form[h].append((pts_h, hs[i], as_[i]))
        form[a].append((3.0 - pts_h if margin != 0 else 1.0, as_[i], hs[i]))
        sched[h].append((ea, w_h - we, float(as_[i] == 0)))
        sched[a].append((eh, (1 - w_h) - (1 - we), float(hs[i] == 0)))
        unbeaten[h] = unbeaten[h] + 1 if margin >= 0 else 0
        unbeaten[a] = unbeaten[a] + 1 if margin <= 0 else 0
        sd_att[h] += SD_K * (hs[i] - lam_h)
        sd_def[a] += SD_K * (hs[i] - lam_h)
        sd_att[a] += SD_K * (as_[i] - lam_a)
        sd_def[h] += SD_K * (as_[i] - lam_a)
        last_date[h] = last_date[a] = d
        recent_dates[h].append(d)
        recent_dates[a].append(d)
        elo_hist[h].append((d, eh))
        elo_hist[a].append((d, ea))
        h2h[key].append(margin if h < a else -margin)
        if imp[i] >= 50:
            majors[h].append(d)
            majors[a].append(d)

        ev = gev.get(date_strs[i], h, a)
        stats = {h: [0, 0, 0], a: [0, 0, 0]}  # goals, late, pens
        for team, minute, pen, scorer in ev:
            if team not in stats:
                continue
            stats[team][0] += 1
            if minute == minute and minute >= LATE_MIN:
                stats[team][1] += 1
            if pen:
                stats[team][2] += 1
            scorers[team].append((d, scorer))
        if ev:
            gh, ga_ev = stats[h], stats[a]
            goalrec[h].append((gh[0], gh[1], ga_ev[0], ga_ev[1], gh[2]))
            goalrec[a].append((ga_ev[0], ga_ev[1], gh[0], gh[1], ga_ev[2]))

    out = df.copy()
    for c, v in cols.items():
        out[c] = v
    out["elo_diff"] = out["elo_home"] - out["elo_away"] \
        + np.where(out["neutral"], 0.0, 80.0)
    out["elo_sum"] = out["elo_home"] + out["elo_away"]
    out["travel_diff"] = out["travel_h"] - out["travel_a"]
    out["fm_xi_diff"] = out["fm_xi_h"] - out["fm_xi_a"]
    out["mv_diff"] = out["mv_log_h"] - out["mv_log_a"]
    out["neutral_f"] = out["neutral"].astype(float)
    return out
