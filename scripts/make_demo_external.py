"""生成「合成」外部数据演示集 -> data/external_demo/

!! 全部数据为合成, 仅用于验证外部数据管线 (schema/对齐/调整通路)。
!! 真实使用请用 FM2026 游戏内编辑器导出 / Transfermarkt 导出 / 赔率商
!! API 落成同 schema 的 CSV 放入 data/external/。

合成方式: 阵容能力值由当前 Elo 映射 + 噪声; 状态/伤停随机;
赔率由 Elo 概率加 6% 水位反推。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from worldcup.data import load_matches
from worldcup.elo import compute_elo_history, expected_score

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "external_demo")
POSITIONS = ["GK"] * 3 + ["DEF"] * 8 + ["MID"] * 8 + ["ATT"] * 7


def main():
    rng = np.random.default_rng(2026)
    raw = load_matches()
    df, ratings = compute_elo_history(raw)
    wc = df[(df["tournament"] == "FIFA World Cup") & (~df["played"])]
    teams = sorted(set(wc["home_team"]) | set(wc["away_team"]))
    snap = "2026-06-10"
    os.makedirs(OUT, exist_ok=True)

    elos = np.array([ratings[t] for t in teams])
    lo, hi = elos.min(), elos.max()

    players, values = [], []
    for t in teams:
        base_ca = 115 + 65 * (ratings[t] - lo) / (hi - lo)  # Elo -> CA 115-180
        for k, pos in enumerate(POSITIONS):
            players.append({
                "date": snap, "nation": t, "player": f"{t} Player {k+1}",
                "position": pos, "age": int(rng.normal(27, 3.5)),
                "current_ability": round(float(np.clip(
                    rng.normal(base_ca - k * 0.8, 6), 80, 195)), 1),
                "condition": round(float(np.clip(rng.normal(0.92, 0.04),
                                                 0.7, 1.0)), 3),
                "injured": int(rng.random() < 0.08),
            })
        values.append({"date": snap, "team": t,
                       "total_value_eur": int(np.exp(rng.normal(0, 0.2))
                                              * 8e6 * np.exp((base_ca - 115) / 12)),
                       "top11_value_eur": int(np.exp(rng.normal(0, 0.2))
                                              * 5e6 * np.exp((base_ca - 115) / 12))})

    odds = []
    for _, r in wc.iterrows():
        we = expected_score(ratings[r["home_team"]], ratings[r["away_team"]],
                            r["neutral"])
        pd_ = 0.24
        p = np.array([we * (1 - pd_), pd_, (1 - we) * (1 - pd_)])
        p = np.clip(p * np.exp(rng.normal(0, 0.05, 3)), 0.02, 0.96)
        p = p / p.sum() * 1.06  # 6% 水位
        odds.append({"date": r["date"].strftime("%Y-%m-%d"),
                     "home_team": r["home_team"], "away_team": r["away_team"],
                     "odds_h": round(1 / p[0], 2), "odds_d": round(1 / p[1], 2),
                     "odds_a": round(1 / p[2], 2)})

    pd.DataFrame(players).to_csv(f"{OUT}/fm_players.csv", index=False)
    pd.DataFrame(values).to_csv(f"{OUT}/market_values.csv", index=False)
    pd.DataFrame(odds).to_csv(f"{OUT}/odds.csv", index=False)
    print(f"已生成合成演示数据 -> {OUT} ({len(teams)} 队, {len(odds)} 场赔率)")


if __name__ == "__main__":
    main()
