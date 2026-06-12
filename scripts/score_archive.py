"""线上监控: 把历史预测存档与已出结果对账, 计算真实线上表现。

用法: python scripts/score_archive.py
输出: 线上 RPS / 对数损失 / 命中率 / 大小球 Brier + 与回测基线对比
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from worldcup.data import load_matches
from worldcup.model import log_loss_, rps


def main():
    files = sorted(glob.glob("predictions/preds_*.csv"))
    if not files:
        print("predictions/ 下无存档, 先运行 scripts/archive_predictions.py")
        return
    preds = pd.concat([pd.read_csv(f) for f in files])
    # 同一场取最早一次预测 (赛前最远 = 最严格的考核)
    preds = preds.sort_values("predicted_at") \
        .drop_duplicates(["match_date", "home", "away"], keep="first")

    res = load_matches()
    res = res[res["played"]]
    res["match_date"] = res["date"].dt.date.astype(str)
    m = preds.merge(res, left_on=["match_date", "home", "away"],
                    right_on=["match_date", "home_team", "away_team"])
    if m.empty:
        print(f"已存档 {len(preds)} 场, 均未完赛")
        return

    p = m[["p_home", "p_draw", "p_away"]].to_numpy()
    diff = m["home_score"] - m["away_score"]
    y = np.where(diff > 0, 0, np.where(diff == 0, 1, 2))
    total = m["home_score"] + m["away_score"]
    over = (total > 2.5).astype(float)

    print(f"线上成绩单: 已对账 {len(m)} 场 (存档共 {len(preds)} 场)")
    print(f"  RPS      = {rps(p, y):.4f}   (回测基线 ~0.158)")
    print(f"  对数损失  = {log_loss_(p, y):.4f}")
    print(f"  命中率    = {(p.argmax(1) == y).mean():.1%}")
    print(f"  大2.5 Brier = {np.mean((m['p_over25'] - over) ** 2):.4f}")
    print(f"  比分命中  = {(m['top_score'] == m['home_score'].astype(int).astype(str) + '-' + m['away_score'].astype(int).astype(str)).mean():.1%}")
    for _, r in m.iterrows():
        hit = "✓" if [r.p_home, r.p_draw, r.p_away].index(
            max(r.p_home, r.p_draw, r.p_away)) == \
            (0 if r.home_score > r.away_score else
             1 if r.home_score == r.away_score else 2) else "✗"
        print(f"  {r.match_date} {r.home} {int(r.home_score)}-"
              f"{int(r.away_score)} {r.away}  "
              f"[{r.p_home:.0%}/{r.p_draw:.0%}/{r.p_away:.0%}] {hit}")


if __name__ == "__main__":
    main()
