"""预测 2026 世界杯: 全部小组赛 + 官方支架蒙特卡洛夺冠概率。

用法: python scripts/predict_worldcup.py [--sims 10000] [--worlds 16]
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
from worldcup.simulate import (TournamentSimulator, fit_shootout_model,
                               infer_groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=10000)
    ap.add_argument("--worlds", type=int, default=16,
                    help="DC 参数不确定性世界数 (1=关闭)")
    args = ap.parse_args()

    ens = Ensemble.load("models/ensemble.pkl")
    raw = load_matches()
    feat, ratings = prepare()

    wc = feat[(feat["tournament"] == "FIFA World Cup") & (~feat["played"])
              & (feat["date"] >= "2026-06-01")].reset_index(drop=True)
    print(f"待赛世界杯小组赛场次: {len(wc)}")

    # -- 逐场小组赛预测 --
    p_all = ens.predict_rows(wc)
    rates = ens.rates_rows(wc)
    print("\n=== 小组赛逐场预测 (主胜/平/客胜 | 最可能比分 | 大2.5) ===")
    for k, (_, r) in enumerate(wc.iterrows()):
        M = adjust_matrix(ens.dc.score_matrix(rates[k, 0], rates[k, 1]),
                          p_all[k])
        rep = market_report(M, top_scores=1)
        (si, sj), sp = rep["top_scores"][0]
        h, d, a = rep["hda"]
        print(f"{r['date'].date()}  {r['home_team']:<22} vs {r['away_team']:<22} "
              f"{h:5.1%}/{d:5.1%}/{a:5.1%}  比分 {si}-{sj} ({sp:.1%})  "
              f"大2.5 {rep['totals']['over_2.5']:.1%}")

    # -- 官方支架蒙特卡洛 --
    groups = infer_groups(wc)
    teams = sorted({t for g in groups.values() for t in g})
    print(f"\n官方分组 (锚定映射): ")
    for letter in sorted(groups):
        print(f"  {letter}: {', '.join(groups[letter])}")

    # 各组赛程 (含末轮标记, 末轮 = 该组按日期最后 2 场)
    group_fixtures = {}
    team2letter = {t: l for l, g in groups.items() for t in g}
    for letter in groups:
        gf = wc[wc["home_team"].map(team2letter) == letter] \
            .sort_values("date")
        fixtures = [(r["home_team"], r["away_team"], False)
                    for _, r in gf.iterrows()]
        group_fixtures[letter] = \
            [(h, a, i >= len(fixtures) - 2) for i, (h, a, _) in
             enumerate(fixtures)]

    so_b, so_c = fit_shootout_model(feat, load_shootouts())
    start = pd.Timestamp("2026-06-11")

    def builder(team_list):
        pairs = [(x, y) for x in team_list for y in team_list if x != y]
        return hypothetical_rows(raw, pairs, start)

    print(f"\n=== 官方支架蒙特卡洛 ({args.sims} 次, {args.worlds} 个参数世界, "
          f"点球模型 b={so_b:.2f} 先罚 c={so_c:.2f}) ===")
    sim = TournamentSimulator(ens, teams, ratings, builder,
                              shootout_params=(so_b, so_c),
                              n_worlds=args.worlds)
    res = sim.simulate(groups, group_fixtures, n_sims=args.sims)

    print(f"\n{'球队':<22}{'夺冠':>8}{'进决赛':>8}{'进四强':>8}{'出线':>8}")
    for t, p in list(res["champion"].items())[:20]:
        print(f"{t:<22}{p:>8.1%}{res['final'].get(t, 0):>8.1%}"
              f"{res['semifinal'].get(t, 0):>8.1%}{res['knockout'].get(t, 0):>8.1%}")


if __name__ == "__main__":
    main()
