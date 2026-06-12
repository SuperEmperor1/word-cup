"""生成静态预测仪表盘 -> site/index.html (零依赖, 可直接打开或部署 Pages)。

用法: python scripts/build_site.py [--days 6]
内容: 摸底考试(已锁定预测) / 近期比赛 / 夺冠概率 / 模型质量 / 线上成绩单
"""
import argparse
import glob
import html
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from worldcup.data import load_matches
from worldcup.markets import adjust_matrix, market_report
from worldcup.model import Ensemble
from worldcup.pipeline import prepare

CSS = """
body{font-family:'PingFang SC','Microsoft YaHei',system-ui,sans-serif;
 background:#0d1117;color:#e6edf3;margin:0;padding:24px;max-width:1080px;
 margin:auto}
h1{font-size:26px;margin:8px 0}h2{font-size:19px;margin:28px 0 10px;
 border-left:4px solid #58a6ff;padding-left:10px}
.meta{color:#8b949e;font-size:13px}
table{border-collapse:collapse;width:100%;font-size:14px;margin:8px 0}
th{color:#8b949e;font-weight:500;text-align:left;padding:6px 8px;
 border-bottom:1px solid #30363d;font-size:12px}
td{padding:7px 8px;border-bottom:1px solid #21262d}
.bar{display:inline-block;height:14px;border-radius:3px;vertical-align:middle}
.h{background:#3fb950}.d{background:#8b949e}.a{background:#f85149}
.prob{font-variant-numeric:tabular-nums;white-space:nowrap}
.tag{background:#21262d;border-radius:4px;padding:1px 7px;font-size:12px;
 color:#8b949e;margin-left:6px}
.exam{background:#161b22;border:1px solid #30363d;border-radius:8px;
 padding:14px 18px;margin:10px 0}
.score{font-size:15px;color:#58a6ff}
.note{color:#8b949e;font-size:12.5px;line-height:1.7;margin-top:24px;
 border-top:1px solid #30363d;padding-top:12px}
.good{color:#3fb950}.warn{color:#d29922}
"""


def prob_bar(p):
    h, d, a = (max(0.5, x * 100) for x in p)
    return (f'<span class="prob">{p[0]:.0%}/{p[1]:.0%}/{p[2]:.0%}</span> '
            f'<span class="bar h" style="width:{h*1.6}px"></span>'
            f'<span class="bar d" style="width:{d*1.6}px"></span>'
            f'<span class="bar a" style="width:{a*1.6}px"></span>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=6)
    args = ap.parse_args()

    ens = Ensemble.load("models/ensemble.pkl")
    meta = json.load(open("models/metadata.json"))
    feat, _ = prepare()
    raw = load_matches()

    today = raw[raw["played"]]["date"].max() + pd.Timedelta(days=1)
    horizon = today + pd.Timedelta(days=args.days)
    up = feat[(~feat["played"]) & (feat["tournament"] == "FIFA World Cup")
              & (feat["date"] <= horizon)].reset_index(drop=True)
    p_all = ens.predict_rows(up)
    rates = ens.rates_rows(up)

    # 摸底考试: 最早一次存档的锁定预测
    exam_rows = ""
    files = sorted(glob.glob("predictions/preds_*.csv"))
    if files:
        first = pd.concat([pd.read_csv(f) for f in files]) \
            .sort_values("predicted_at") \
            .drop_duplicates(["match_date", "home", "away"], keep="first")
        exam = first[first["match_date"] == str(today.date())]
        for _, r in exam.iterrows():
            exam_rows += (
                f"<tr><td>{r['home']} vs {r['away']}</td>"
                f"<td>{prob_bar((r['p_home'], r['p_draw'], r['p_away']))}</td>"
                f"<td class='score'>{r['top_score']} ({r['p_top_score']:.0%})</td>"
                f"<td>{r['p_over25']:.0%}</td>"
                f"<td class='meta'>{r['predicted_at']} · {r['model_version']}</td></tr>")

    match_rows = ""
    for k, (_, r) in enumerate(up.iterrows()):
        M = adjust_matrix(ens.dc.score_matrix(rates[k, 0], rates[k, 1]), p_all[k])
        rep = market_report(M, top_scores=1)
        (si, sj), sp = rep["top_scores"][0]
        mkt = ""
        if not np.isnan(r.get("mkt_ph", np.nan)):
            mkt = (f"<span class='tag'>市场 {r['mkt_ph']:.0%}/"
                   f"{r['mkt_pd']:.0%}/{r['mkt_pa']:.0%}</span>")
        match_rows += (
            f"<tr><td>{r['date'].date()}</td>"
            f"<td>{r['home_team']} vs {r['away_team']}{mkt}</td>"
            f"<td>{prob_bar(p_all[k])}</td>"
            f"<td class='score'>{si}-{sj} ({sp:.0%})</td>"
            f"<td>{rep['totals']['over_2.5']:.0%}</td></tr>")

    sim_rows = ""
    if os.path.exists("models/sim_results.json"):
        sim = json.load(open("models/sim_results.json"))
        for t, p in list(sim["champion"].items())[:15]:
            sim_rows += (
                f"<tr><td>{t}</td>"
                f"<td><span class='bar h' style='width:{p*800:.0f}px'></span> "
                f"<span class='prob'>{p:.1%}</span></td>"
                f"<td>{sim['final'].get(t, 0):.1%}</td>"
                f"<td>{sim['semifinal'].get(t, 0):.1%}</td>"
                f"<td>{sim['knockout'].get(t, 0):.1%}</td></tr>")

    bt = json.load(open("models/backtest_report.json"))
    e, m = bt["ensemble"], bt["markets"]
    cal = "".join(
        f"<tr><td>{c['bucket']}</td><td>{c['pred']:.1%}</td>"
        f"<td>{c['actual']:.1%}</td><td>{c['n']}</td></tr>"
        for c in bt["calibration"])

    # 线上成绩单
    res_played = raw[raw["played"]].copy()
    res_played["md"] = res_played["date"].dt.date.astype(str)
    live = "<p class='meta'>暂无已完赛的存档预测 —— 摸底考试进行中。</p>"
    if files:
        j = first.merge(res_played, left_on=["match_date", "home", "away"],
                        right_on=["md", "home_team", "away_team"])
        if len(j):
            from worldcup.model import rps as rps_
            p = j[["p_home", "p_draw", "p_away"]].to_numpy()
            diff = j["home_score"] - j["away_score"]
            y = np.where(diff > 0, 0, np.where(diff == 0, 1, 2)).astype(int)
            r = rps_(p, y)
            hit = (p.argmax(1) == y).mean()
            cls = "good" if r < 0.17 else "warn"
            live = (f"<p>已对账 <b>{len(j)}</b> 场 · 线上 RPS "
                    f"<b class='{cls}'>{r:.4f}</b>（回测基线 0.157）· "
                    f"方向命中 <b>{hit:.0%}</b></p><table><tr><th>比赛</th>"
                    f"<th>赛果</th><th>赛前预测</th></tr>")
            for _, rr in j.iterrows():
                live += (f"<tr><td>{rr['home']} vs {rr['away']}</td>"
                         f"<td class='score'>{int(rr['home_score'])}-"
                         f"{int(rr['away_score'])}</td>"
                         f"<td>{prob_bar((rr['p_home'], rr['p_draw'], rr['p_away']))}"
                         f"</td></tr>")
            live += "</table>"

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    page = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>世界杯预测系统 · 2026</title><style>{CSS}</style></head><body>
<h1>⚽ 世界杯预测系统 <span class="tag">v{meta['version']}</span></h1>
<p class="meta">数据截止 {meta['data_end']} · {meta['n_matches']:,} 场训练样本 ·
{meta['n_features']} 维特征 · 生成于 {now} ·
绿/灰/红 = 主胜/平/客胜概率</p>

<h2>📋 摸底考试（赛前锁定，不可修改）</h2>
<div class="exam"><table>
<tr><th>比赛</th><th>锁定预测 胜/平/负</th><th>最可能比分</th>
<th>大2.5球</th><th>锁定时间 · 模型版本</th></tr>{exam_rows}</table></div>

<h2>📊 线上成绩单</h2>{live}

<h2>🔮 近期比赛预测（含市场赔率对照）</h2>
<table><tr><th>日期</th><th>对阵</th><th>胜/平/负</th><th>最可能比分</th>
<th>大2.5</th></tr>{match_rows}</table>

<h2>🏆 夺冠概率（官方支架 · 蒙特卡洛 · 16 参数世界）</h2>
<table><tr><th>球队</th><th>夺冠</th><th>进决赛</th><th>进四强</th>
<th>小组出线</th></tr>{sim_rows}</table>

<h2>🔬 模型质量（1,313 场样本外回测）</h2>
<p>胜平负 RPS <b>{e['rps']:.4f}</b> · 对数损失 {e['log_loss']:.4f} ·
命中率 {e['accuracy']:.1%} · 比分对数损失 {m['scoreline_log_loss']:.3f}
（基准 {m['scoreline_log_loss_baseline']:.3f}）· 大2.5球 Brier
{m['totals']['over_2.5']['brier']:.4f}</p>
<table><tr><th>主胜概率分桶</th><th>预测均值</th><th>实际频率</th>
<th>场次</th></tr>{cal}</table>

<p class="note">本页面由模型自动生成。所有概率为经过校准的统计估计，
存在不可消除的随机性；精确比分单点概率上限约 30% 属正常现象。
本系统不构成任何投注建议；模型相对博彩市场的经济优势未经投注回测验证。
预测在赛前锁定存档（predictions/），事后不可修改，线上成绩单按最早存档计算。
</p></body></html>"""
    os.makedirs("site", exist_ok=True)
    with open("site/index.html", "w", encoding="utf-8") as f:
        f.write(page)
    print(f"已生成 site/index.html ({len(page)//1024}KB)")


if __name__ == "__main__":
    main()
