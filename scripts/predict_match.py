"""预测任意对阵: 胜平负 / 比分 / 大小球 / 让球盘 / 半场 / 进球者。

用法: python scripts/predict_match.py "Brazil" "Morocco"
      [--home] [--date 2026-06-13] [--ah] [--ht] [--scorers]
      [--explain] [--uncertainty]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from worldcup.data import load_goal_events, load_matches
from worldcup.evaluate import FAMILIES
from worldcup.markets import (adjust_matrix, ah_lines, half_time_report,
                              market_report)
from worldcup.model import Ensemble
from worldcup.pipeline import hypothetical_rows


def predict_one(ens, raw, home, away, date, neutral=True):
    rows = hypothetical_rows(raw, [(home, away)], pd.Timestamp(date), neutral)
    p = ens.predict_rows(rows)[0]
    lam, mu = ens.rates_row(rows.iloc[0])
    M = adjust_matrix(ens.dc.score_matrix(lam, mu), p)
    rep = market_report(M)
    rep.update(lambda_mu=(lam, mu), matrix=M, rows=rows,
               elo=(float(rows.iloc[0]["elo_home"]),
                    float(rows.iloc[0]["elo_away"])))
    return rep


def uncertainty_band(ens, rows, n=200, seed=1):
    """DC 参数 Fisher 标准误 -> 最终概率的 90% 区间。"""
    rng = np.random.default_rng(seed)
    r = rows.iloc[0]
    # 与点估计同源: 扰动的是 DC 分量的强度 (blend 内部即用此分量)
    lam, mu = ens.dc.rates(r["home_team"], r["away_team"], r["neutral"],
                           r["elo_home"], r["elo_away"])
    se_ah, se_dh = ens.dc.strength_se(r["home_team"])
    se_aa, se_da = ens.dc.strength_se(r["away_team"])
    ps = []
    for _ in range(n):
        l = lam * np.exp(rng.normal(0, se_ah) + rng.normal(0, se_da))
        m = mu * np.exp(rng.normal(0, se_aa) + rng.normal(0, se_dh))
        ps.append(ens.blend(rows, p_dc=ens.dc.match_probs(l, m)[None, :])[0])
    ps = np.array(ps)
    return np.percentile(ps, 5, axis=0), np.percentile(ps, 95, axis=0)


def explain(ens, rows):
    """家族级局部归因: 把某特征族置为缺失 (NaN), 看 GBM 概率变化。"""
    base = ens.gbm_probs(rows)[0]
    out = []
    for name, cols in FAMILIES.items():
        rr = rows.copy()
        for c in cols:
            if c in rr.columns:
                rr[c] = np.nan
        delta = base - ens.gbm_probs(rr)[0]
        out.append((name, delta[0]))
    return base, sorted(out, key=lambda kv: -abs(kv[1]))


def scorer_probs(home, away, lam, mu, top=5):
    """P(球员进球) ≈ 1 − exp(−λ·近2年队内进球占比)。"""
    g = load_goal_events()
    g = g[(g["date"] >= "2024-06-01") & (~g["own_goal"])]
    out = {}
    for team, rate in ((home, lam), (away, mu)):
        cnt = g[g["team"] == team]["scorer"].value_counts()
        tot = cnt.sum()
        if tot < 5:
            continue
        out[team] = [(p, 1 - np.exp(-rate * c / tot))
                     for p, c in cnt.head(top).items()]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("home")
    ap.add_argument("away")
    ap.add_argument("--home", dest="is_home", action="store_true")
    ap.add_argument("--date", default=None)
    ap.add_argument("--ah", action="store_true", help="亚洲让球盘")
    ap.add_argument("--ht", action="store_true", help="半场/半全场")
    ap.add_argument("--scorers", action="store_true", help="进球者概率")
    ap.add_argument("--explain", action="store_true", help="特征族归因")
    ap.add_argument("--uncertainty", action="store_true", help="概率区间")
    args = ap.parse_args()

    ens = Ensemble.load("models/ensemble.pkl")
    raw = load_matches()
    date = args.date or (raw[raw["played"]]["date"].max()
                         + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    rep = predict_one(ens, raw, args.home, args.away, date,
                      neutral=not args.is_home)
    h, d, a = rep["hda"]
    lam, mu = rep["lambda_mu"]
    print(f"\n=== {args.home} vs {args.away} ({date}) ===")
    print(f"Elo: {rep['elo'][0]:.0f} vs {rep['elo'][1]:.0f}   "
          f"期望进球 λ={lam:.2f} : μ={mu:.2f} (合计 {rep['expected_goals_total']:.2f})")
    print(f"胜平负:  主胜 {h:.1%} | 平 {d:.1%} | 客胜 {a:.1%}")
    print("最可能比分: " + ", ".join(
        f"{i}-{j} ({p:.1%})" for (i, j), p in rep["top_scores"]))
    print("大小球:  " + "  ".join(
        f"大{ln} {p:.1%}" for ln, p in
        zip((0.5, 1.5, 2.5, 3.5, 4.5), rep["totals"].values())))
    print(f"双方进球 BTTS: {rep['btts']:.1%}")
    print("总进球分布: " + "  ".join(
        f"{g}球 {p:.1%}" for g, p in rep["goals_dist"].items()))

    if args.uncertainty:
        lo, hi = uncertainty_band(ens, rep["rows"])
        print("90% 参数不确定性区间: "
              + "  ".join(f"{n} {l:.1%}-{u:.1%}" for n, l, u in
                          zip(("主胜", "平", "客胜"), lo, hi)))
    if args.ah:
        print("\n亚洲让球盘 (主队让球 | 赢/走/输 | 公平赔率):")
        for line, r in ah_lines(rep["matrix"]).items():
            print(f"  {line}: {r['win']:.1%}/{r['push']:.1%}/{r['lose']:.1%}"
                  f"  赔率 {r['fair_odds']}")
    if args.ht:
        ht = half_time_report(ens.dc, lam, mu)
        hh, hd, ha = ht["ht_hda"]
        print(f"\n半场: 主胜 {hh:.1%} | 平 {hd:.1%} | 客胜 {ha:.1%}   "
              + "  ".join(f"大{k.split('_')[1]} {v:.1%}"
                          for k, v in ht["ht_totals"].items()))
        print("半全场: " + "  ".join(f"{k} {v:.1%}" for k, v in
                                  sorted(ht["ht_ft"].items(),
                                         key=lambda kv: -kv[1])[:5]))
    if args.scorers:
        print("\n进球者概率 (近2年队内占比模型):")
        for team, lst in scorer_probs(args.home, args.away, lam, mu).items():
            print(f"  {team}: " + ", ".join(f"{p} {q:.0%}" for p, q in lst))
    if args.explain:
        base, attr = explain(ens, rep["rows"])
        print(f"\n特征族归因 (该族信息对主胜概率的拉动, GBM 基准 {base[0]:.1%}):")
        for name, delta in attr:
            print(f"  {name:<12} {delta:+.1%}")


if __name__ == "__main__":
    main()
