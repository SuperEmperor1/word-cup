"""预测球队赛前最可能阵型与首发 XI。

用法: python scripts/predict_lineup.py "Brazil" [--date 2026-06-13]
数据: data/external/ 下 projected_lineups.csv (媒体共识, 优先) /
      lineups.csv (历史出场推断) / squads.csv (伤病) / coaches.csv (教练任期)
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from worldcup.lineups import LineupPredictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("team")
    ap.add_argument("--date", default=None)
    args = ap.parse_args()
    date = pd.Timestamp(args.date) if args.date else pd.Timestamp.now()

    lp = LineupPredictor.discover()
    pred = lp.predict(args.team, date)
    if not pred:
        print(f"{args.team}: 无名单数据 (需 projected_lineups.csv 或 "
              f"lineups.csv, 见 data/external/README.md)")
        return
    print(f"\n=== {args.team} 预测首发 ({date.date()}) ===")
    print(f"阵型: {pred['formation']}   确定度: {pred['certainty']:.0%}"
          + (f"   来源: 媒体共识" if pred.get("source") == "projected"
             else "   来源: 出场历史推断"))
    for player, pos, p in pred["xi"]:
        print(f"  {pos:<4} {player:<24} P(首发)={p:.0%}")
    if pred["missing_starters"]:
        print(f"核心缺阵: {', '.join(pred['missing_starters'])}  "
              f"(Elo 修正 {pred['elo_adjustment']:+.0f})")


if __name__ == "__main__":
    main()
