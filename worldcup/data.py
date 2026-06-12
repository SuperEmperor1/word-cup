"""数据加载与清洗。

数据源: martj42/international_results —— 1872 年至今所有国际 A 级赛事,
字段: date, home_team, away_team, home_score, away_score, tournament,
      city, country, neutral
未来赛程 (如 2026 世界杯) 的比分为 NA, 用于预测。
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "results.csv")

# 赛事重要性 -> Elo K 因子 (World Football Elo Ratings 标准)
K_WORLD_CUP = 60.0
K_CONTINENTAL = 50.0
K_QUALIFIER = 40.0
K_NATIONS = 30.0
K_FRIENDLY = 20.0

_CONTINENTAL = (
    "UEFA Euro", "Copa América", "African Cup of Nations",
    "Africa Cup of Nations", "AFC Asian Cup", "Gold Cup",
    "CONCACAF Championship", "Oceania Nations Cup", "Confederations Cup",
)


def match_importance(tournament: str) -> float:
    """按赛事名称返回 Elo K 因子, 同时作为模型的重要性权重基准。"""
    t = tournament or ""
    if "qualification" in t.lower():
        return K_QUALIFIER
    if t == "FIFA World Cup":
        return K_WORLD_CUP
    if any(t.startswith(c) for c in _CONTINENTAL):
        return K_CONTINENTAL
    if "Nations League" in t:
        return K_NATIONS
    if t == "Friendly":
        return K_FRIENDLY
    return K_NATIONS  # 其余小型锦标赛


def load_matches(path: str = DATA_PATH) -> pd.DataFrame:
    """加载全部比赛, 返回按日期排序的 DataFrame。

    返回两类行: 已完赛 (played=True) 与未来赛程 (played=False)。
    摄入时做数据质量校验 (重复行 / 异常比分 / 队名空白)。
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df = validate_matches(df)
    df["neutral"] = df["neutral"].astype(bool)
    df["played"] = df["home_score"].notna() & df["away_score"].notna()
    df["importance"] = df["tournament"].map(match_importance)
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    # 极端比分 (历史早期 31-0 之类) 截断, 避免扭曲泊松强度
    for c in ("home_score", "away_score"):
        df[c] = np.minimum(df[c].astype(float), 12)
    return df


def validate_matches(df: pd.DataFrame) -> pd.DataFrame:
    """防御性摄入校验: 队名修剪、自我对阵/负比分剔除、重复行去重。"""
    import warnings
    for c in ("home_team", "away_team"):
        df[c] = df[c].str.strip()
    bad = (df["home_team"] == df["away_team"]) \
        | (df["home_score"] < 0) | (df["away_score"] < 0)
    if bad.any():
        warnings.warn(f"剔除 {int(bad.sum())} 行异常数据 (自我对阵/负比分)")
        df = df[~bad]
    dup = df.duplicated(["date", "home_team", "away_team"], keep="first")
    if dup.any():
        warnings.warn(f"剔除 {int(dup.sum())} 行重复比赛")
        df = df[~dup]
    return df


GOALS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "goalscorers.csv")
SHOOTOUTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "shootouts.csv")
CENTROIDS_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                              "country_centroids.csv")

# 数据集球队/举办国名 -> 坐标表国名 的差异修正
_COUNTRY_ALIASES = {
    "DR Congo": "Congo DRC", "Republic of Ireland": "Ireland",
    "Cape Verde": "Cabo Verde", "Ivory Coast": "Côte d'Ivoire",
    "Curaçao": "Curacao", "São Tomé and Príncipe": "Sao Tome and Principe",
    "Czech Republic": "Czechia", "Turkey": "Turkiye", "Swaziland": "Eswatini",
    "Macedonia": "North Macedonia", "Micronesia": "Micronesia (Federated States of)",
    "United States Virgin Islands": "US Virgin Islands",
    "Saint Vincent and the Grenadines": "St Vincent and the Grenadines",
    "England": "United Kingdom", "Scotland": "United Kingdom",
    "Wales": "United Kingdom", "Northern Ireland": "United Kingdom",
}


# 高海拔比赛城市 (米)。海拔效应在 >800m 才显著, 其余城市近似为 0;
# GeoNames 全量数据被网络策略限制, 静态表覆盖全部文献中显著的场地。
CITY_ALTITUDES: dict[str, float] = {
    "El Alto": 4090, "La Paz": 3640, "Cusco": 3400, "Potosí": 4060,
    "Sucre": 2810, "Oruro": 3710, "Cochabamba": 2560, "Quito": 2850,
    "Bogotá": 2640, "Arequipa": 2335, "Toluca": 2660, "Mexico City": 2240,
    "Pachuca": 2432, "Puebla": 2135, "Aguascalientes": 1880,
    "San Luis Potosí": 1860, "Querétaro": 1820, "León": 1815,
    "Guadalajara": 1566, "Zapopan": 1560, "Monterrey": 540,
    "Addis Ababa": 2355, "Asmara": 2325, "Sana'a": 2250,
    "Nairobi": 1795, "Johannesburg": 1753, "Pretoria": 1339,
    "Bloemfontein": 1395, "Harare": 1490, "Kigali": 1567, "Kampala": 1190,
    "Kathmandu": 1400, "Thimphu": 2334, "Tehran": 1190, "Yerevan": 990,
    "Ankara": 938, "Denver": 1609, "Guatemala City": 1500,
    "Quetzaltenango": 2330, "Tegucigalpa": 990, "San José": 1170,
    "Madrid": 667,
}


def load_goal_events(path: str = GOALS_PATH) -> pd.DataFrame:
    """球员级进球明细: date/home_team/away_team/team/scorer/minute/own_goal/penalty"""
    g = pd.read_csv(path)
    g["own_goal"] = g["own_goal"].astype(bool)
    g["penalty"] = g["penalty"].astype(bool)
    return g


def load_shootouts(path: str = SHOOTOUTS_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def load_centroids(path: str = CENTROIDS_PATH) -> dict[str, tuple[float, float]]:
    """国家 -> (纬度, 经度), 含足球队名别名修正。"""
    c = pd.read_csv(path)
    table = {r["COUNTRY"]: (float(r["latitude"]), float(r["longitude"]))
             for _, r in c.iterrows()}
    out = dict(table)
    for alias, canon in _COUNTRY_ALIASES.items():
        if canon in table:
            out[alias] = table[canon]
    return out


def played(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["played"]].reset_index(drop=True)


def outcome_labels(df: pd.DataFrame) -> np.ndarray:
    """胜平负标签: 0=主胜 H, 1=平 D, 2=客胜 A。"""
    diff = df["home_score"].to_numpy() - df["away_score"].to_numpy()
    return np.where(diff > 0, 0, np.where(diff == 0, 1, 2))
