"""预测 2026 世界杯: 全部小组赛 + 蒙特卡洛夺冠概率。

用法: python scripts/predict_worldcup.py [--sims 10000]
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from worldcup.data import load_matches, load_shootouts
from worldcup.markets import adjust_matrix, market_report
from worldcup.model import Ensemble
from worldcup.pipeline import hypothetical_rows, prepare
from worldcup.simulate import TournamentSimulator, fit_shootout_slope, infer_groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=10000)
    args = ap.parse_args()

    ens = Ensemble.load("models/ensemble.pkl")
    raw = load_matches()
    feat, ratings = prepare()

    wc = feat[(feat["tournament"] == "FIFA World Cup") & (~feat["played"])
              & (feat["date"] >= "2026-06-01")]
    print(f"待赛世界杯场次: {len(wc)}")

    # -- 逐场小组赛预测 --
    p_all = ens.predict_rows(wc)
    print("\n=== 小组赛逐场预测 (主胜/平/客胜 | 最可能比分 | 大2.5) ===")
    for k, (_, r) in enumerate(wc.iterrows()):
        lam, mu = ens.dc.rates(r["home_team"], r["away_team"], r["neutral"],
                               r["elo_home"], r["elo_away"])
        M = adjust_matrix(ens.dc.score_matrix(lam, mu), p_all[k])
        rep = market_report(M, top_scores=1)
        (si, sj), sp = rep["top_scores"][0]
        h, d, a = rep["hda"]
        print(f"{r['date'].date()}  {r['home_team']:<22} vs {r['away_team']:<22} "
              f"{h:5.1%}/{d:5.1%}/{a:5.1%}  比分 {si}-{sj} ({sp:.1%})  "
              f"大2.5 {rep['totals']['over_2.5']:.1%}")

    # -- 整赛事蒙特卡洛 --
    groups = infer_groups(wc)
    teams = sorted({t for g in groups for t in g})
    print(f"\n推断出 {len(groups)} 个小组: ")
    for i, g in enumerate(groups):
        print(f"  组{i+1}: {', '.join(g)}")

    start = pd.Timestamp("2026-06-11")

    def builder(team_list):
        pairs = [(a, b) for a in team_list for b in team_list if a != b]
        return hypothetical_rows(raw, pairs, start)

    slope = fit_shootout_slope(feat, load_shootouts())
    print(f"\n=== 蒙特卡洛模拟 ({args.sims} 次, 点球模型斜率 b={slope:.2f}) ===")
    sim = TournamentSimulator(ens, teams, ratings, builder, shootout_slope=slope)
    res = sim.simulate(groups, n_sims=args.sims)

    print(f"\n{'球队':<22}{'夺冠':>8}{'进决赛':>8}{'进四强':>8}{'出线':>8}")
    for t, p in list(res["champion"].items())[:20]:
        print(f"{t:<22}{p:>8.1%}{res['final'].get(t, 0):>8.1%}"
              f"{res['semifinal'].get(t, 0):>8.1%}{res['knockout'].get(t, 0):>8.1%}")


if __name__ == "__main__":
    main()
