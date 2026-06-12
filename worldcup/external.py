"""外部商业数据适配层: FM2026 球员导出 / 转会市场身价 / 博彩赔率。

设计原则
  1. 标准化 schema + 严格校验: 任何来源 (FM 游戏内编辑器导出、Transfermarkt
     导出、赔率商 API) 只要落成约定 CSV 即可接入, 见 data/external/README。
  2. 防泄漏 as-of 对齐: 每条快照带 date, 对任一比赛只取「该比赛之前最近
     一次」快照, 且超过陈旧上限 (staleness) 视为缺失 -> NaN (GBM 原生处理)。
  3. 双通路生效:
     - 特征通路: 进入 GBM 特征表, 随历史快照积累自动被学习;
     - 即时调整通路: 伤停/状态 -> Elo 当量修正, 无需重训立即影响预测。

schema (data/external/*.csv):
  fm_players.csv : date,nation,player,position,age,current_ability,
                   condition,injured
                   (position ∈ GK/DEF/MID/ATT; current_ability 0-200 FM 标度;
                    condition 0-1; injured 0/1)
  market_values.csv : date,team,total_value_eur,top11_value_eur
  odds.csv : date,home_team,away_team,odds_h,odds_d,odds_a  (欧赔小数)
"""
from __future__ import annotations

import os
from bisect import bisect_right
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EXTERNAL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "external")

FM_STALE_DAYS = 120      # FM 快照陈旧上限 (阵容/状态变化快)
MV_STALE_DAYS = 400      # 身价陈旧上限 (变化慢)

# 即时调整通路的 Elo 当量系数 (可按自有数据重标定):
# 伤停负担每损失 10% 阵容总 CA ≈ -40 Elo; 平均状态相对基准 0.92 每 ±0.01 ≈ ±4 Elo
INJURY_ELO_PER_BURDEN = -400.0
CONDITION_ELO_PER_UNIT = 400.0
CONDITION_BASELINE = 0.92

_FM_COLS = {"date", "nation", "player", "position", "age",
            "current_ability", "condition", "injured"}
_MV_COLS = {"date", "team", "total_value_eur", "top11_value_eur"}
_ODDS_COLS = {"date", "home_team", "away_team", "odds_h", "odds_d", "odds_a"}

FM_FEATURES = ["fm_xi", "fm_depth", "fm_gk", "fm_age", "fm_stardep",
               "fm_cond", "fm_inj"]


def _validate(df: pd.DataFrame, required: set, name: str) -> pd.DataFrame:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} 缺少必需列: {sorted(missing)}")
    return df


# ------------------------------------------------------------ FM 阵容聚合
def aggregate_squad(squad: pd.DataFrame) -> dict[str, float]:
    """单国家单快照 -> 阵容强度指标。

    fm_xi      : 最佳 11 人 (1 门将 + 10 outfield) 平均 CA
    fm_depth   : 第 12-23 人平均 CA (轮换深度)
    fm_gk      : 最佳门将 CA
    fm_age     : 首发平均年龄
    fm_stardep : 头牌 CA / 首发均值 (核心依赖度)
    fm_cond    : 首发可用球员的状态加权均值 (伤员剔除)
    fm_inj     : 伤停负担 = 伤员 CA 占 23 人总 CA 比例
    """
    s = squad.sort_values("current_ability", ascending=False)
    gk = s[s["position"] == "GK"]
    outf = s[s["position"] != "GK"]
    xi = pd.concat([gk.head(1), outf.head(10)])
    bench = pd.concat([gk.iloc[1:2], outf.iloc[10:21]])
    top23 = pd.concat([xi, bench])
    fit_xi = pd.concat([gk[~gk["injured"].astype(bool)].head(1),
                        outf[~outf["injured"].astype(bool)].head(10)])
    inj_ca = top23.loc[top23["injured"].astype(bool), "current_ability"].sum()
    return {
        "fm_xi": float(xi["current_ability"].mean()),
        "fm_depth": float(bench["current_ability"].mean()) if len(bench) else np.nan,
        "fm_gk": float(gk["current_ability"].max()) if len(gk) else np.nan,
        "fm_age": float(xi["age"].mean()),
        "fm_stardep": float(s["current_ability"].iloc[0] / xi["current_ability"].mean()),
        "fm_cond": float(fit_xi["condition"].mean()) if len(fit_xi) else np.nan,
        "fm_inj": float(inj_ca / max(top23["current_ability"].sum(), 1e-9)),
    }


# ------------------------------------------------------------ 赔率去水
def devig_shin(odds: np.ndarray, iters: int = 50) -> np.ndarray:
    """Shin (1993) 去水: 比按比例归一更好地校正长热偏差 (favorite-longshot)。

    odds: (n,3) 欧赔 -> 返回 (n,3) 无水分隐含概率。
    """
    pi = 1.0 / odds
    booksum = pi.sum(axis=1, keepdims=True)
    z = np.full((len(odds), 1), 0.02)
    for _ in range(iters):
        num = np.sqrt(z ** 2 + 4 * (1 - z) * pi ** 2 / booksum) - z
        p = num / (2 * (1 - z))
        z = z + 0.5 * (p.sum(axis=1, keepdims=True) - 1)
        z = np.clip(z, 0.0, 0.2)
    p = (np.sqrt(z ** 2 + 4 * (1 - z) * pi ** 2 / booksum) - z) / (2 * (1 - z))
    return p / p.sum(axis=1, keepdims=True)


# ------------------------------------------------------------ as-of 查询
class _AsOf:
    """team -> 按日期排序的快照序列, O(log n) 取「比赛前最近且未过期」的一条。"""

    def __init__(self, df: pd.DataFrame, key: str, stale_days: int):
        self.stale = np.timedelta64(stale_days, "D")
        self.dates: dict[str, list] = {}
        self.rows: dict[str, list] = {}
        for team, grp in df.groupby(key):
            g = grp.sort_values("date")
            self.dates[team] = list(g["date"].to_numpy("datetime64[D]"))
            self.rows[team] = g.drop(columns=["date", key]).to_dict("records")

    def get(self, team: str, date: np.datetime64) -> dict | None:
        ds = self.dates.get(team)
        if not ds:
            return None
        k = bisect_right(ds, date) - 1
        if k < 0 or date - ds[k] > self.stale:
            return None
        return self.rows[team][k]


# ------------------------------------------------------------ 总入口
@dataclass
class ExternalData:
    fm: _AsOf | None = None
    mv: _AsOf | None = None
    odds: dict | None = None
    sources: list[str] = field(default_factory=list)

    @classmethod
    def discover(cls, directory: str = EXTERNAL_DIR) -> "ExternalData":
        """自动发现 data/external/ 下的标准文件, 缺失则优雅降级。"""
        ext = cls()
        p = os.path.join(directory, "fm_players.csv")
        if os.path.exists(p):
            raw = _validate(pd.read_csv(p, parse_dates=["date"]), _FM_COLS,
                            "fm_players.csv")
            agg = (raw.groupby(["nation", "date"])
                      .apply(lambda g: pd.Series(aggregate_squad(g)),
                             include_groups=False).reset_index())
            ext.fm = _AsOf(agg, "nation", FM_STALE_DAYS)
            ext.sources.append("fm_players")
        p = os.path.join(directory, "market_values.csv")
        if os.path.exists(p):
            mv = _validate(pd.read_csv(p, parse_dates=["date"]), _MV_COLS,
                           "market_values.csv")
            ext.mv = _AsOf(mv, "team", MV_STALE_DAYS)
            ext.sources.append("market_values")
        p = os.path.join(directory, "odds.csv")
        if os.path.exists(p):
            od = _validate(pd.read_csv(p, parse_dates=["date"]), _ODDS_COLS,
                           "odds.csv")
            probs = devig_shin(od[["odds_h", "odds_d", "odds_a"]].to_numpy(float))
            ext.odds = {(r.date.strftime("%Y-%m-%d"), r.home_team, r.away_team):
                        probs[i] for i, r in enumerate(od.itertuples())}
            ext.sources.append("odds")
        return ext

    # ---- 特征通路 ----
    def fm_features(self, team: str, date: np.datetime64) -> dict[str, float]:
        rec = self.fm.get(team, date) if self.fm else None
        return rec if rec else {k: np.nan for k in FM_FEATURES}

    def mv_log(self, team: str, date: np.datetime64) -> float:
        rec = self.mv.get(team, date) if self.mv else None
        return float(np.log(rec["top11_value_eur"])) if rec else np.nan

    def match_odds(self, date_str: str, home: str, away: str) -> np.ndarray:
        if self.odds:
            p = self.odds.get((date_str, home, away))
            if p is not None:
                return p
        return np.full(3, np.nan)

    # ---- 即时调整通路 ----
    def elo_adjustment(self, team: str, date: np.datetime64) -> float:
        """伤停 + 状态 -> Elo 当量修正 (无外部数据时为 0)。"""
        rec = self.fm.get(team, date) if self.fm else None
        if not rec:
            return 0.0
        adj = 0.0
        if rec["fm_inj"] == rec["fm_inj"]:
            adj += INJURY_ELO_PER_BURDEN * rec["fm_inj"]
        if rec["fm_cond"] == rec["fm_cond"]:
            adj += CONDITION_ELO_PER_UNIT * (rec["fm_cond"] - CONDITION_BASELINE)
        return float(np.clip(adj, -120, 60))
