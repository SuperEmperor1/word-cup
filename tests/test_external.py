"""外部数据层测试: schema 校验 / as-of 防泄漏 / Shin 去水 / 阵容聚合 / 调整通路。"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from worldcup.external import (_AsOf, ExternalData, aggregate_squad,
                               devig_shin)


def _toy_squad(n_gk=2, n_out=20, injured_star=False):
    rows = []
    for k in range(n_gk):
        rows.append({"player": f"gk{k}", "position": "GK", "age": 28,
                     "current_ability": 150 - 10 * k, "condition": 0.9,
                     "injured": 0})
    for k in range(n_out):
        rows.append({"player": f"p{k}", "position": "MID", "age": 26,
                     "current_ability": 170 - 2 * k, "condition": 0.92,
                     "injured": 1 if (k == 0 and injured_star) else 0})
    return pd.DataFrame(rows)


def test_aggregate_squad_basic():
    agg = aggregate_squad(_toy_squad())
    # XI = 最佳门将(150) + outfield 前10 (170..152)
    assert abs(agg["fm_xi"] - (150 + sum(170 - 2 * k for k in range(10))) / 11) < 1e-9
    assert agg["fm_gk"] == 150 and agg["fm_inj"] == 0.0
    assert agg["fm_stardep"] > 1.0


def test_aggregate_squad_injury_burden():
    agg = aggregate_squad(_toy_squad(injured_star=True))
    assert agg["fm_inj"] > 0.04  # 头牌(CA170)伤停, 负担显著
    healthy = aggregate_squad(_toy_squad())
    assert agg["fm_cond"] <= healthy["fm_cond"] + 1e-9


def test_asof_no_future_leakage():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01", "2026-06-01"]),
        "team": ["X", "X"], "v": [1.0, 2.0]})
    asof = _AsOf(df, "team", stale_days=120)
    d = np.datetime64("2026-05-30")  # 6月快照在未来, 必须取1月的
    rec = asof.get("X", d)
    assert rec is None or rec["v"] == 1.0
    # 1月快照距 5/30 已 149 天 > 120 -> 过期
    assert asof.get("X", d) is None
    assert asof.get("X", np.datetime64("2026-03-01"))["v"] == 1.0
    assert asof.get("X", np.datetime64("2026-06-02"))["v"] == 2.0


def test_devig_shin():
    odds = np.array([[1.5, 4.2, 7.0], [2.8, 3.1, 2.9]])
    p = devig_shin(odds)
    assert np.allclose(p.sum(1), 1, atol=1e-6)
    raw = 1 / odds
    overround = raw.sum(1)
    assert (overround > 1.02).all()        # 原始含水
    assert (p < raw / 1.0).all()           # 去水后概率低于含水隐含
    assert p[0, 0] > 0.6                   # 热门仍是热门


def test_discover_missing_dir_degrades():
    ext = ExternalData.discover("/nonexistent/path")
    assert ext.sources == []
    assert np.isnan(list(ext.fm_features("Brazil", np.datetime64("2026-06-01")).values())).all()
    assert ext.elo_adjustment("Brazil", np.datetime64("2026-06-01")) == 0.0
    assert np.isnan(ext.match_odds("2026-06-13", "Brazil", "Morocco")).all()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
