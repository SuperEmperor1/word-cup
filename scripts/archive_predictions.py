"""预测存档: 把当前模型对所有未赛世界杯比赛的预测落盘 (可追溯/可事后评分)。

用法: python scripts/archive_predictions.py
产出: predictions/preds_<时间戳>.csv
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from worldcup.markets import adjust_matrix, market_report
from worldcup.model import Ensemble
from worldcup.pipeline import prepare


def main():
    ens = Ensemble.load("models/ensemble.pkl")
    feat, _ = prepare()
    up = feat[(~feat["played"]) & (feat["date"] >= feat["date"].min())] \
        .reset_index(drop=True)
    up = up[up["tournament"] == "FIFA World Cup"]
    if up.empty:
        print("无待赛场次")
        return

    p = ens.predict_rows(up)
    rates = ens.rates_rows(up)
    rows = []
    for k, (_, r) in enumerate(up.iterrows()):
        M = adjust_matrix(ens.dc.score_matrix(rates[k, 0], rates[k, 1]), p[k])
        rep = market_report(M, top_scores=1)
        (si, sj), sp = rep["top_scores"][0]
        rows.append({
            "match_date": r["date"].date(), "home": r["home_team"],
            "away": r["away_team"], "p_home": round(p[k, 0], 4),
            "p_draw": round(p[k, 1], 4), "p_away": round(p[k, 2], 4),
            "lam": round(rates[k, 0], 3), "mu": round(rates[k, 1], 3),
            "top_score": f"{si}-{sj}", "p_top_score": round(sp, 4),
            "p_over25": round(rep["totals"]["over_2.5"], 4),
            "p_btts": round(rep["btts"], 4),
        })
    meta = {}
    mp = "models/metadata.json"
    if os.path.exists(mp):
        meta = json.load(open(mp))
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    df = pd.DataFrame(rows)
    df["predicted_at"] = ts
    df["model_version"] = meta.get("version", "unknown")
    os.makedirs("predictions", exist_ok=True)
    out = f"predictions/preds_{ts}.csv"
    df.to_csv(out, index=False)
    print(f"已存档 {len(df)} 场预测 -> {out}")


if __name__ == "__main__":
    main()
