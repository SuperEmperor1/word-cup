"""赛前首发阵容与阵型预测。

输入 (data/external/, 均为可选, 缺失优雅降级):
  lineups.csv : date,team,formation,player,position,started
                (历史出场记录, position ∈ GK/DEF/MID/ATT; started 1=首发)
  squads.csv  : team,player,position,injured  (当前大名单与伤病)
  coaches.csv : team,coach,since              (现任教练任期起点)

预测逻辑 ("伤病 + 教练喜好 + 出场历史"):
  1. 每场历史权重 = exp(-Δt/180天) x 现任教练任期内 x3 (教练喜好)
  2. P(首发) = 加权首发率; 伤员置 0; 大名单外置 0
  3. 阵型 = 加权众数 (如 4-3-3 -> GK1/DEF4/MID3/ATT3)
  4. XI = 各位置组按 P(首发) 取前 N (位置约束的贪心)
  5. 核心缺阵 -> Elo 当量修正: 基线 P(首发)>=0.6 的伤员每人 -8, 上限 -40

输出附 "阵容确定度" (XI 平均 P(首发)) —— 名单数据稀疏时如实给低确定度。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

EXTERNAL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "external")
HALF_LIFE_DAYS = 180.0
COACH_BONUS = 3.0
ELO_PER_MISSING_STARTER = -8.0
MAX_LINEUP_ADJ = -40.0

_FORMATION_SLOTS = {  # 阵型 -> (DEF, MID, ATT); GK 恒为 1
    "4-3-3": (4, 3, 3), "4-2-3-1": (4, 5, 1), "4-4-2": (4, 4, 2),
    "3-5-2": (3, 5, 2), "3-4-3": (3, 4, 3), "5-4-1": (5, 4, 1),
    "5-3-2": (5, 3, 2), "4-5-1": (4, 5, 1), "4-1-4-1": (4, 5, 1),
}


def _slots(formation: str) -> tuple[int, int, int]:
    if formation in _FORMATION_SLOTS:
        return _FORMATION_SLOTS[formation]
    parts = [int(x) for x in formation.split("-") if x.isdigit()]
    if len(parts) >= 2 and sum(parts) == 10:
        return parts[0], sum(parts[1:-1]), parts[-1]
    return 4, 4, 2


PROJECTED_FRESH_DAYS = 4      # 媒体预测 XI 的新鲜度窗口
PROJECTED_CERTAINTY = 0.85    # 赛前媒体预测 XI 的经验准确率


@dataclass
class LineupPredictor:
    lineups: pd.DataFrame | None = None
    squads: pd.DataFrame | None = None
    coaches: pd.DataFrame | None = None
    projected: pd.DataFrame | None = None  # 媒体共识预测 XI (快速通道)

    @classmethod
    def discover(cls, directory: str = EXTERNAL_DIR) -> "LineupPredictor":
        def _read(name, dates=()):
            p = os.path.join(directory, name)
            return pd.read_csv(p, parse_dates=list(dates)) \
                if os.path.exists(p) else None
        return cls(lineups=_read("lineups.csv", ["date"]),
                   squads=_read("squads.csv"),
                   coaches=_read("coaches.csv", ["since"]),
                   projected=_read("projected_lineups.csv", ["date"]))

    @property
    def available(self) -> bool:
        return (self.lineups is not None and len(self.lineups) > 0) \
            or (self.projected is not None and len(self.projected) > 0)

    def _from_projected(self, team: str, as_of: pd.Timestamp) -> dict | None:
        """快速通道: 新鲜的媒体共识预测 XI 优先于历史推断。"""
        if self.projected is None:
            return None
        p = self.projected[(self.projected["team"] == team)
                           & ((self.projected["date"] - as_of).dt.days
                              .abs() <= PROJECTED_FRESH_DAYS)]
        if len(p) < 11:
            return None
        p = p.sort_values("date").tail(11)
        injured = self._injured(team)
        missing = []
        if self.squads is not None:
            s = self.squads[self.squads["team"] == team]
            if "key_player" in s.columns:
                missing = list(s[(s["injured"].astype(bool))
                                 & (s["key_player"].astype(bool))]["player"])
        adj = float(np.clip(len(missing) * ELO_PER_MISSING_STARTER,
                            MAX_LINEUP_ADJ, 0))
        return {
            "formation": str(p.iloc[0]["formation"]),
            "xi": [(r["player"], r["position"],
                    PROJECTED_CERTAINTY * (0.6 if r["player"] in injured else 1))
                   for _, r in p.iterrows()],
            "certainty": PROJECTED_CERTAINTY,
            "missing_starters": missing,
            "elo_adjustment": adj,
            "source": "projected",
        }

    # ------------------------------------------------------------ 核心
    def _team_history(self, team: str, as_of: pd.Timestamp) -> pd.DataFrame:
        h = self.lineups[(self.lineups["team"] == team)
                         & (self.lineups["date"] < as_of)].copy()
        if h.empty:
            return h
        dt = (as_of - h["date"]).dt.days
        h["w"] = np.exp(-dt * np.log(2) / HALF_LIFE_DAYS)
        if self.coaches is not None:
            row = self.coaches[self.coaches["team"] == team]
            if len(row):
                h.loc[h["date"] >= row.iloc[0]["since"], "w"] *= COACH_BONUS
        return h

    def _injured(self, team: str) -> set[str]:
        if self.squads is None:
            return set()
        s = self.squads[self.squads["team"] == team]
        return set(s.loc[s["injured"].astype(bool), "player"])

    def _squad(self, team: str) -> set[str] | None:
        if self.squads is None:
            return None
        s = self.squads[self.squads["team"] == team]
        return set(s["player"]) if len(s) else None

    def predict(self, team: str, as_of: pd.Timestamp) -> dict | None:
        """返回 {formation, xi: [(player, pos, p_start)], certainty,
        missing_starters, elo_adjustment}; 无数据返回 None。"""
        if not self.available:
            return None
        proj = self._from_projected(team, as_of)
        if proj:
            return proj
        if self.lineups is None:
            return None
        h = self._team_history(team, as_of)
        if h.empty or h["date"].nunique() < 3:
            return None
        injured = self._injured(team)
        squad = self._squad(team)

        # 阵型: 加权众数
        fw = h.drop_duplicates(["date"])[["formation", "w"]] \
            .groupby("formation")["w"].sum()
        formation = str(fw.idxmax())

        # P(首发)
        tot_w = h.drop_duplicates(["date"])["w"].sum()
        g = h[h["started"].astype(bool)].groupby(["player", "position"])["w"] \
            .sum().reset_index()
        g["p_start"] = (g["w"] / tot_w).clip(0, 1)

        baseline = g.copy()  # 伤病排除前的基线 (用于识别核心缺阵)
        if squad is not None:
            g = g[g["player"].isin(squad)]
        g = g[~g["player"].isin(injured)]

        d, m, a = _slots(formation)
        xi, used = [], set()
        for pos, need in (("GK", 1), ("DEF", d), ("MID", m), ("ATT", a)):
            pool = g[(g["position"] == pos) & (~g["player"].isin(used))] \
                .sort_values("p_start", ascending=False).head(need)
            for _, r in pool.iterrows():
                xi.append((r["player"], pos, float(r["p_start"])))
                used.add(r["player"])
        # 槽位不满 (数据稀疏) 用任意位置最高者补齐
        if len(xi) < 11:
            rest = g[~g["player"].isin(used)] \
                .sort_values("p_start", ascending=False).head(11 - len(xi))
            xi += [(r["player"], r["position"], float(r["p_start"]))
                   for _, r in rest.iterrows()]

        missing = baseline[(baseline["p_start"] >= 0.6)
                           & (baseline["player"].isin(injured))]
        adj = float(np.clip(len(missing) * ELO_PER_MISSING_STARTER,
                            MAX_LINEUP_ADJ, 0))
        return {
            "formation": formation,
            "xi": sorted(xi, key=lambda r: ("GK DEF MID ATT".split()
                                            .index(r[1]), -r[2])),
            "certainty": float(np.mean([p for *_, p in xi])) if xi else 0.0,
            "missing_starters": list(missing["player"]),
            "elo_adjustment": adj,
        }

    def elo_adjustment(self, team: str, as_of: pd.Timestamp) -> float:
        pred = self.predict(team, as_of)
        return pred["elo_adjustment"] if pred else 0.0
