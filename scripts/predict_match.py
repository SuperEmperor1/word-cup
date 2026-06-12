"""预测任意对阵: 胜平负 / 精确比分 / 进球数市场。

用法: python scripts/predict_match.py "Brazil" "Morocco" [--home] [--date 2026-06-13]
默认中立场地。--home 表示第一支球队坐镇主场。
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from worldcup.data import load_matches
from worldcup.markets import adjust_matrix, market_report
from worldcup.model import Ensemble
from worldcup.pipeline import hypothetical_rows


def predict_one(ens, raw, home, away, date, neutral=True):
    rows = hypothetical_rows(raw, [(home, away)], pd.Timestamp(date), neutral)
    r = rows.iloc[0]
    p = ens.predict_rows(rows)[0]
    lam, mu = ens.dc.rates(home, away, neutral, r["elo_home"], r["elo_away"])
    M = adjust_matrix(ens.dc.score_matrix(lam, mu), p)
    rep = market_report(M)
    rep["lambda_mu"] = (lam, mu)
    rep["elo"] = (float(r["elo_home"]), float(r["elo_away"]))
    return rep


def fmt(rep, home, away):
    h, d, a = rep["hda"]
    lam, mu = rep["lambda_mu"]
    print(f"\n=== {home} vs {away} ===")
    print(f"Elo: {rep['elo'][0]:.0f} vs {rep['elo'][1]:.0f}   "
          f"期望进球 λ={lam:.2f} : μ={mu:.2f} (合计 {rep['expected_goals_total']:.2f})")
    print(f"胜平负:  主胜 {h:.1%} | 平 {d:.1%} | 客胜 {a:.1%}")
    print("最可能比分: " + ", ".join(
        f"{i}-{j} ({p:.1%})" for (i, j), p in rep["top_scores"]))
    print("大小球:  " + "  ".join(
        f"大{line} {p:.1%}" for line, p in
        zip((0.5, 1.5, 2.5, 3.5, 4.5), rep["totals"].values())))
    print(f"双方进球 BTTS: {rep['btts']:.1%}")
    gd = rep["goals_dist"]
    print("总进球分布: " + "  ".join(f"{g}球 {p:.1%}" for g, p in gd.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("home")
    ap.add_argument("away")
    ap.add_argument("--home", dest="is_home", action="store_true",
                    help="第一支球队为主场 (默认中立)")
    ap.add_argument("--date", default=None)
    args = ap.parse_args()

    ens = Ensemble.load("models/ensemble.pkl")
    raw = load_matches()
    date = args.date or (raw[raw["played"]]["date"].max()
                         + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    rep = predict_one(ens, raw, args.home, args.away, date,
                      neutral=not args.is_home)
    fmt(rep, args.home, args.away)


if __name__ == "__main__":
    main()
