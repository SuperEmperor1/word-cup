"""首发预测模块测试 (合成数据)。"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from worldcup.lineups import LineupPredictor


def _toy():
    rows = []
    dates = pd.date_range("2025-09-01", periods=8, freq="30D")
    for i, d in enumerate(dates):
        formation = "4-3-3" if i >= 3 else "4-4-2"  # 新教练改打 4-3-3
        for k in range(11):
            pos = "GK" if k == 0 else ("DEF" if k <= 4 else
                                       ("MID" if k <= 7 else "ATT"))
            rows.append({"date": d, "team": "X", "formation": formation,
                         "player": f"P{k}", "position": pos, "started": 1})
        # 替补 B1 偶尔首发顶替 P10
        rows.append({"date": d, "team": "X", "formation": formation,
                     "player": "B1", "position": "ATT",
                     "started": 1 if i == 0 else 0})
    lineups = pd.DataFrame(rows)
    squads = pd.DataFrame([{"team": "X", "player": f"P{k}", "position": "ATT",
                            "injured": 0} for k in range(11)]
                          + [{"team": "X", "player": "B1", "position": "ATT",
                              "injured": 0}])
    coaches = pd.DataFrame([{"team": "X", "coach": "New",
                             "since": pd.Timestamp("2025-12-01")}])
    return LineupPredictor(lineups=lineups, squads=squads, coaches=coaches)


def test_formation_follows_current_coach():
    lp = _toy()
    pred = lp.predict("X", pd.Timestamp("2026-06-13"))
    assert pred["formation"] == "4-3-3"  # 教练加权压过早期 4-4-2


def test_xi_positional_structure():
    lp = _toy()
    pred = lp.predict("X", pd.Timestamp("2026-06-13"))
    pos_count = {}
    for _, pos, _ in pred["xi"]:
        pos_count[pos] = pos_count.get(pos, 0) + 1
    assert pos_count["GK"] == 1 and pos_count["DEF"] == 4
    assert len(pred["xi"]) == 11
    assert pred["certainty"] > 0.7
    assert pred["elo_adjustment"] == 0.0


def test_injury_excludes_and_adjusts():
    lp = _toy()
    lp.squads.loc[lp.squads["player"] == "P10", "injured"] = 1  # 主力前锋伤
    pred = lp.predict("X", pd.Timestamp("2026-06-13"))
    players = [p for p, *_ in pred["xi"]]
    assert "P10" not in players and "B1" in players  # 替补顶上
    assert "P10" in pred["missing_starters"]
    assert pred["elo_adjustment"] == -8.0


def test_no_data_degrades():
    lp = LineupPredictor()
    assert lp.predict("X", pd.Timestamp("2026-06-13")) is None
    assert lp.elo_adjustment("X", pd.Timestamp("2026-06-13")) == 0.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
