"""抓取中国体彩竞彩足球公开赔率 -> data/external/odds.csv

数据源: 体彩官方公开接口 (webapi.sporttery.cn, 即竞彩官网页面所用接口)。
本仓库的执行环境网络受限时, 请在本地运行本脚本:

    python scripts/fetch_odds_sporttery.py            # 拉取并合并
    python scripts/fetch_odds_sporttery.py --input resp.json  # 离线解析

产出 schema 与 worldcup/external.py 约定一致, 预测时自动:
  1. Shin 方法去水 (体彩固定奖金返还率约 71%, 水位很高, 必须去水)
  2. 与模型概率线性池锚定 (Ensemble.predict_rows market_anchor)

注意: 体彩"胜平负"以 90 分钟计, 与模型口径一致。
"""
import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

API = ("https://webapi.sporttery.cn/gateway/jc/football/"
       "getMatchCalculatorV1.qry?poolCode=had&channel=c")

# 体彩中文队名 -> 数据集英文队名 (2026 世界杯 48 队)
CN2EN = {
    "墨西哥": "Mexico", "南非": "South Africa", "韩国": "South Korea",
    "捷克": "Czech Republic", "加拿大": "Canada",
    "波黑": "Bosnia and Herzegovina", "波斯尼亚和黑塞哥维那": "Bosnia and Herzegovina",
    "卡塔尔": "Qatar", "瑞士": "Switzerland", "巴西": "Brazil",
    "海地": "Haiti", "摩洛哥": "Morocco", "苏格兰": "Scotland",
    "澳大利亚": "Australia", "巴拉圭": "Paraguay", "土耳其": "Turkey",
    "美国": "United States", "库拉索": "Curaçao", "厄瓜多尔": "Ecuador",
    "德国": "Germany", "科特迪瓦": "Ivory Coast", "日本": "Japan",
    "荷兰": "Netherlands", "瑞典": "Sweden", "突尼斯": "Tunisia",
    "比利时": "Belgium", "埃及": "Egypt", "伊朗": "Iran",
    "新西兰": "New Zealand", "佛得角": "Cape Verde",
    "沙特阿拉伯": "Saudi Arabia", "沙特": "Saudi Arabia",
    "西班牙": "Spain", "乌拉圭": "Uruguay", "法国": "France",
    "伊拉克": "Iraq", "挪威": "Norway", "塞内加尔": "Senegal",
    "阿尔及利亚": "Algeria", "阿根廷": "Argentina", "奥地利": "Austria",
    "约旦": "Jordan", "哥伦比亚": "Colombia",
    "刚果民主共和国": "DR Congo", "刚果(金)": "DR Congo", "刚果金": "DR Congo",
    "葡萄牙": "Portugal", "乌兹别克斯坦": "Uzbekistan",
    "克罗地亚": "Croatia", "英格兰": "England", "加纳": "Ghana",
    "巴拿马": "Panama",
}

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "external",
                   "odds.csv")


def _to_en(name: str) -> str | None:
    if not name:
        return None
    name = name.strip()
    if name in CN2EN:
        return CN2EN[name]
    for cn, en in CN2EN.items():  # 容错: 体彩队名偶有前后缀
        if cn in name:
            return en
    return None


def _walk(obj, found: list) -> None:
    """递归扫描 JSON, 提取含 主客队名+胜平负赔率 的节点 (对接口结构变化鲁棒)。"""
    if isinstance(obj, dict):
        keys = set(obj)
        name_h = obj.get("homeTeamAllName") or obj.get("homeTeamAbbName")
        name_a = obj.get("awayTeamAllName") or obj.get("awayTeamAbbName")
        if name_h and name_a:
            had = obj.get("had") or obj
            h, d, a = had.get("h"), had.get("d"), had.get("a")
            date = (obj.get("matchDate") or obj.get("businessDate")
                    or obj.get("matchTime", ""))[:10]
            if h and d and a and date:
                found.append((date, name_h, name_a, float(h), float(d),
                              float(a)))
        for v in obj.values():
            _walk(v, found)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, found)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="离线解析已保存的接口 JSON")
    args = ap.parse_args()

    if args.input:
        data = json.load(open(args.input, encoding="utf-8"))
    else:
        req = urllib.request.Request(API, headers={
            "User-Agent": "Mozilla/5.0", "Referer": "https://www.sporttery.cn/"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))

    found: list = []
    _walk(data, found)
    rows, skipped = [], set()
    for date, cn_h, cn_a, h, d, a in found:
        en_h, en_a = _to_en(cn_h), _to_en(cn_a)
        if not en_h or not en_a:
            skipped.add(f"{cn_h} vs {cn_a}")
            continue
        rows.append({"date": date, "home_team": en_h, "away_team": en_a,
                     "odds_h": h, "odds_d": d, "odds_a": a,
                     "fetched_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"),
                     "source": "sporttery"})
    if skipped:
        print(f"未识别队名 (非世界杯场次或需补充 CN2EN): {sorted(skipped)[:10]}")
    if not rows:
        print("未解析到任何赔率 — 检查接口返回或用 --input 离线调试")
        return

    new = pd.DataFrame(rows).drop_duplicates(
        ["date", "home_team", "away_team"], keep="last")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT):
        old = pd.read_csv(OUT)
        # 历史赔率保留 (供特征通路积累), 同场新快照覆盖旧值
        merged = pd.concat([old, new]).drop_duplicates(
            ["date", "home_team", "away_team"], keep="last")
    else:
        merged = new
    merged.to_csv(OUT, index=False)
    print(f"已写入 {len(new)} 场新赔率, odds.csv 现共 {len(merged)} 场 -> {OUT}")
    print("下一步: python scripts/archive_predictions.py (预测将自动市场锚定)")


if __name__ == "__main__":
    main()
