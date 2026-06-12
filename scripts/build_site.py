"""生成静态预测仪表盘 -> site/index.html (零依赖, 可直接打开或部署 Pages)。

用法: python scripts/build_site.py [--days 6]
内容: 摸底考试(已锁定预测) / 线上成绩单 / 近期比赛 / 夺冠概率 / 模型质量
"""
import argparse
import glob
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

FLAGS = {
    "Mexico": "🇲🇽", "South Africa": "🇿🇦", "South Korea": "🇰🇷",
    "Czech Republic": "🇨🇿", "Canada": "🇨🇦",
    "Bosnia and Herzegovina": "🇧🇦", "Qatar": "🇶🇦", "Switzerland": "🇨🇭",
    "Brazil": "🇧🇷", "Haiti": "🇭🇹", "Morocco": "🇲🇦",
    "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "Australia": "🇦🇺", "Paraguay": "🇵🇾",
    "Turkey": "🇹🇷", "United States": "🇺🇸", "Curaçao": "🇨🇼",
    "Ecuador": "🇪🇨", "Germany": "🇩🇪", "Ivory Coast": "🇨🇮",
    "Japan": "🇯🇵", "Netherlands": "🇳🇱", "Sweden": "🇸🇪",
    "Tunisia": "🇹🇳", "Belgium": "🇧🇪", "Egypt": "🇪🇬", "Iran": "🇮🇷",
    "New Zealand": "🇳🇿", "Cape Verde": "🇨🇻", "Saudi Arabia": "🇸🇦",
    "Spain": "🇪🇸", "Uruguay": "🇺🇾", "France": "🇫🇷", "Iraq": "🇮🇶",
    "Norway": "🇳🇴", "Senegal": "🇸🇳", "Algeria": "🇩🇿",
    "Argentina": "🇦🇷", "Austria": "🇦🇹", "Jordan": "🇯🇴",
    "Colombia": "🇨🇴", "DR Congo": "🇨🇩", "Portugal": "🇵🇹",
    "Uzbekistan": "🇺🇿", "Croatia": "🇭🇷", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "Ghana": "🇬🇭", "Panama": "🇵🇦",
}


def team(name):
    return f"{FLAGS.get(name, '⚽')}&nbsp;{name}"


CSS = """
:root{--bg:#0a0e1a;--card:#11182b;--card2:#0e1424;--line:#1e2942;
 --txt:#e8edf7;--mut:#7d8aa5;--green:#2dd4a7;--grey:#3d4a66;--red:#f0647c;
 --blue:#4f8ff7;--gold:#f7c948}
*{box-sizing:border-box}
body{font-family:'PingFang SC','HarmonyOS Sans','Microsoft YaHei',
 -apple-system,'Segoe UI',sans-serif;background:
 radial-gradient(1200px 500px at 70% -10%,#16305e33,transparent),
 radial-gradient(900px 400px at 10% 10%,#0f3d3433,transparent),var(--bg);
 color:var(--txt);margin:0;padding:0 20px 60px;line-height:1.55}
.wrap{max-width:1060px;margin:auto}
header{padding:44px 0 10px}
h1{font-size:30px;font-weight:700;margin:0;letter-spacing:.5px;
 background:linear-gradient(90deg,#fff,#9cc3ff 60%,#2dd4a7);
 -webkit-background-clip:text;background-clip:text;color:transparent}
.sub{color:var(--mut);font-size:13.5px;margin-top:8px}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px}
.chip{background:#13203d;border:1px solid var(--line);color:#a9bbdb;
 border-radius:999px;padding:4px 13px;font-size:12.5px}
.chip b{color:#fff;font-weight:600}
h2{font-size:17px;font-weight:600;margin:40px 0 14px;display:flex;
 align-items:center;gap:9px}
h2 .ico{font-size:19px}
h2 .ln{flex:1;height:1px;background:linear-gradient(90deg,var(--line),transparent)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));
 gap:14px}
.card{background:linear-gradient(180deg,var(--card),var(--card2));
 border:1px solid var(--line);border-radius:14px;padding:18px 20px;
 box-shadow:0 8px 24px #0006}
.card.exam{border-color:#2a4a8a}
.vs{display:flex;justify-content:space-between;align-items:center;
 font-size:16.5px;font-weight:600}
.vs .at{color:var(--mut);font-size:12px;font-weight:400}
.stack{display:flex;height:22px;border-radius:7px;overflow:hidden;
 margin:13px 0 7px;font-size:11.5px;font-weight:600;color:#06121f}
.stack div{display:flex;align-items:center;justify-content:center;
 min-width:0;white-space:nowrap;overflow:hidden}
.sH{background:linear-gradient(180deg,#3ee6b4,#1fbd8d)}
.sD{background:linear-gradient(180deg,#56678d,#46556f);color:#dbe4f5}
.sA{background:linear-gradient(180deg,#fb7b92,#e05570)}
.lg{display:flex;justify-content:space-between;color:var(--mut);
 font-size:11.5px;margin-bottom:10px}
.kv{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;color:#bccbe8}
.kv b{color:#fff}
.lock{margin-top:11px;color:var(--mut);font-size:11px;display:flex;
 align-items:center;gap:5px}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th{color:var(--mut);font-weight:500;text-align:left;padding:8px 10px;
 font-size:11.5px;text-transform:uppercase;letter-spacing:.4px;
 border-bottom:1px solid var(--line)}
td{padding:9px 10px;border-bottom:1px solid #161f36}
tr:hover td{background:#13203d55}
.bar3{display:flex;width:170px;height:9px;border-radius:5px;
 overflow:hidden;background:var(--grey)}
.bar3 i{display:block}
.pr{font-variant-numeric:tabular-nums;color:#c6d4ee;font-size:12.5px;
 white-space:nowrap}
.mk{color:var(--mut);font-size:11px}
.sc{color:var(--gold);font-weight:600;white-space:nowrap}
.tbar{height:13px;border-radius:4px;
 background:linear-gradient(90deg,#f7c948,#f08c3a);min-width:2px}
.good{color:var(--green)}.warn{color:var(--gold)}.bad{color:var(--red)}
.cal{display:flex;gap:3px;align-items:flex-end;height:90px;margin:10px 0}
.cal .col{flex:1;display:flex;flex-direction:column;justify-content:flex-end;
 gap:2px;align-items:center}
.cal .p,.cal .a{width:70%;border-radius:3px 3px 0 0}
.cal .p{background:#4f8ff7aa}.cal .a{background:#2dd4a7aa}
.cal .lab{font-size:9.5px;color:var(--mut)}
.note{color:var(--mut);font-size:12px;line-height:1.8;margin-top:46px;
 border-top:1px solid var(--line);padding-top:16px}
.empty{color:var(--mut);background:var(--card2);border:1px dashed var(--line);
 border-radius:12px;padding:22px;text-align:center;font-size:13.5px}
@media(max-width:640px){.bar3{width:110px}h1{font-size:24px}}
"""


def stack_bar(p, big=True):
    segs = ""
    for v, cls, lab in zip(p, ("sH", "sD", "sA"), ("胜", "平", "负")):
        txt = f"{v:.0%}" if (big and v >= 0.12) else ""
        segs += f'<div class="{cls}" style="width:{max(v*100,1.5):.1f}%">{txt}</div>'
    return f'<div class="stack">{segs}</div>' if big else \
        (f'<div class="bar3">'
         f'<i class="sH" style="width:{p[0]*100:.0f}%"></i>'
         f'<i class="sD" style="width:{p[1]*100:.0f}%"></i>'
         f'<i class="sA" style="width:{p[2]*100:.0f}%"></i></div>')


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

    # ---- 摸底考试卡片 ----
    files = sorted(glob.glob("predictions/preds_*.csv"))
    exam_cards = ""
    first = None
    if files:
        first = pd.concat([pd.read_csv(f) for f in files]) \
            .sort_values("predicted_at") \
            .drop_duplicates(["match_date", "home", "away"], keep="first")
        for _, r in first[first["match_date"] == str(today.date())].iterrows():
            p = (r["p_home"], r["p_draw"], r["p_away"])
            exam_cards += f"""
<div class="card exam"><div class="vs"><span>{team(r['home'])}</span>
<span class="at">VS</span><span>{team(r['away'])}</span></div>
{stack_bar(p)}<div class="lg"><span>主胜</span><span>平局</span><span>客胜</span></div>
<div class="kv"><span>最可能比分 <b class="sc">{r['top_score']}</b>
({r['p_top_score']:.0%})</span><span>大2.5球 <b>{r['p_over25']:.0%}</b></span>
<span>双方进球 <b>{r['p_btts']:.0%}</b></span></div>
<div class="lock">🔒 {r['predicted_at']} 锁定 · 模型 {r['model_version']}</div></div>"""

    # ---- 近期比赛 ----
    match_rows = ""
    for k, (_, r) in enumerate(up.iterrows()):
        M = adjust_matrix(ens.dc.score_matrix(rates[k, 0], rates[k, 1]), p_all[k])
        rep = market_report(M, top_scores=1)
        (si, sj), sp = rep["top_scores"][0]
        mkt = "—"
        if not np.isnan(r.get("mkt_ph", np.nan)):
            mkt = f"{r['mkt_ph']:.0%}/{r['mkt_pd']:.0%}/{r['mkt_pa']:.0%}"
        match_rows += f"""
<tr><td class="mk">{r['date'].strftime('%m-%d')}</td>
<td>{team(r['home_team'])} <span class="mk">vs</span> {team(r['away_team'])}</td>
<td>{stack_bar(p_all[k], big=False)}</td>
<td class="pr">{p_all[k][0]:.0%}/{p_all[k][1]:.0%}/{p_all[k][2]:.0%}</td>
<td class="mk">{mkt}</td>
<td class="sc">{si}-{sj} <span class="mk">({sp:.0%})</span></td>
<td class="pr">{rep['totals']['over_2.5']:.0%}</td></tr>"""

    # ---- 夺冠概率 ----
    sim_rows = ""
    if os.path.exists("models/sim_results.json"):
        sim = json.load(open("models/sim_results.json"))
        top = list(sim["champion"].items())[:15]
        pmax = top[0][1]
        for t, p in top:
            sim_rows += f"""
<tr><td>{team(t)}</td>
<td><div class="tbar" style="width:{p/pmax*230:.0f}px"></div></td>
<td class="pr">{p:.1%}</td><td class="pr">{sim['final'].get(t,0):.1%}</td>
<td class="pr">{sim['semifinal'].get(t,0):.1%}</td>
<td class="pr">{sim['knockout'].get(t,0):.1%}</td></tr>"""

    # ---- 线上成绩单 ----
    res_played = raw[raw["played"]].copy()
    res_played["md"] = res_played["date"].dt.date.astype(str)
    live = '<div class="empty">⏳ 暂无已完赛的存档预测 —— 摸底考试进行中，赛后自动对账。</div>'
    if first is not None:
        j = first.merge(res_played, left_on=["match_date", "home", "away"],
                        right_on=["md", "home_team", "away_team"])
        if len(j):
            from worldcup.model import rps as rps_
            p = j[["p_home", "p_draw", "p_away"]].to_numpy()
            diff = j["home_score"] - j["away_score"]
            y = np.where(diff > 0, 0, np.where(diff == 0, 1, 2)).astype(int)
            r_ = rps_(p, y)
            hit = (p.argmax(1) == y).mean()
            cls = "good" if r_ < 0.17 else ("warn" if r_ < 0.23 else "bad")
            rows = "".join(
                f"<tr><td>{team(rr['home'])} <span class='mk'>vs</span> "
                f"{team(rr['away'])}</td>"
                f"<td class='sc'>{int(rr['home_score'])}-{int(rr['away_score'])}</td>"
                f"<td>{stack_bar((rr['p_home'], rr['p_draw'], rr['p_away']), big=False)}</td>"
                f"<td class='pr'>{rr['p_home']:.0%}/{rr['p_draw']:.0%}/{rr['p_away']:.0%}</td></tr>"
                for _, rr in j.iterrows())
            live = f"""<div class="card"><div class="kv" style="margin-bottom:10px">
<span>已对账 <b>{len(j)}</b> 场</span>
<span>线上 RPS <b class="{cls}">{r_:.4f}</b> <span class="mk">(回测基线 0.157)</span></span>
<span>方向命中 <b>{hit:.0%}</b></span></div>
<table><tr><th>比赛</th><th>赛果</th><th colspan="2">赛前锁定预测</th></tr>{rows}</table></div>"""

    # ---- 校准图 ----
    bt = json.load(open("models/backtest_report.json"))
    e, m = bt["ensemble"], bt["markets"]
    cal_cols = ""
    for c in bt["calibration"]:
        cal_cols += f"""<div class="col">
<div class="p" style="height:{c['pred']*80:.0f}px" title="预测 {c['pred']:.1%}"></div>
<div class="a" style="height:{c['actual']*80:.0f}px" title="实际 {c['actual']:.1%}"></div>
<div class="lab">{c['pred']:.0%}</div></div>"""

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    page = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>⚽ 世界杯预测系统 2026</title><style>{CSS}</style></head><body>
<div class="wrap">
<header><h1>世界杯预测系统 2026</h1>
<p class="sub">Dixon-Coles 双泊松 · 101 维特征 Stacking · 官方支架蒙特卡洛 ·
市场赔率锚定 —— 所有预测赛前锁定，事后可验</p>
<div class="chips">
<span class="chip">模型 <b>v{meta['version']}</b></span>
<span class="chip">训练样本 <b>{meta['n_matches']:,}</b> 场</span>
<span class="chip">样本外 RPS <b>{e['rps']:.4f}</b></span>
<span class="chip">命中率 <b>{e['accuracy']:.0%}</b></span>
<span class="chip">更新 <b>{now}</b></span></div></header>

<h2><span class="ico">📋</span>摸底考试 · {today.date()}<span class="ln"></span></h2>
<div class="cards">{exam_cards or '<div class="empty">今日无锁定考卷</div>'}</div>

<h2><span class="ico">📊</span>线上成绩单<span class="ln"></span></h2>
{live}

<h2><span class="ico">🔮</span>近期比赛预测<span class="ln"></span></h2>
<div class="card"><table>
<tr><th>日期</th><th>对阵</th><th colspan="2">胜 / 平 / 负</th>
<th>市场参考</th><th>最可能比分</th><th>大2.5</th></tr>{match_rows}</table></div>

<h2><span class="ico">🏆</span>夺冠概率 <span class="mk">官方支架 ·
蒙特卡洛 · 16 参数世界</span><span class="ln"></span></h2>
<div class="card"><table>
<tr><th>球队</th><th></th><th>夺冠</th><th>决赛</th><th>四强</th>
<th>出线</th></tr>{sim_rows}</table></div>

<h2><span class="ico">🔬</span>模型质量 <span class="mk">1,313 场样本外
滚动回测</span><span class="ln"></span></h2>
<div class="cards">
<div class="card"><div class="kv">
<span>胜平负 RPS <b>{e['rps']:.4f}</b></span>
<span>对数损失 <b>{e['log_loss']:.3f}</b></span>
<span>比分对数损失 <b>{m['scoreline_log_loss']:.3f}</b>
<span class="mk">(基准 {m['scoreline_log_loss_baseline']:.3f})</span></span>
<span>大2.5 Brier <b>{m['totals']['over_2.5']['brier']:.4f}</b></span></div></div>
<div class="card"><div class="mk" style="margin-bottom:4px">
校准检验：<span style="color:#4f8ff7">■</span> 预测概率 vs
<span style="color:#2dd4a7">■</span> 实际频率（按主胜概率分桶）</div>
<div class="cal">{cal_cols}</div></div></div>

<p class="note">⚠️ 本页面由模型自动生成。所有概率为经过严格校准的统计估计，
足球比赛存在不可消除的随机性，精确比分单点概率上限约 30% 属正常现象。
本系统不构成任何投注建议，模型相对博彩市场的经济优势未经投注回测验证。
预测在赛前锁定存档，事后不可修改；线上成绩单按每场最早一次存档计算（最严格口径）。
数据源：martj42/international_results (CC0) · 市场赔率为公开报价。</p>
</div></body></html>"""
    os.makedirs("site", exist_ok=True)
    with open("site/index.html", "w", encoding="utf-8") as f:
        f.write(page)
    print(f"已生成 site/index.html ({len(page)//1024}KB)")


if __name__ == "__main__":
    main()
