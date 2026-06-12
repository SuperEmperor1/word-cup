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
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df["neutral"] = df["neutral"].astype(bool)
    df["played"] = df["home_score"].notna() & df["away_score"].notna()
    df["importance"] = df["tournament"].map(match_importance)
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    # 极端比分 (历史早期 31-0 之类) 截断, 避免扭曲泊松强度
    for c in ("home_score", "away_score"):
        df[c] = np.minimum(df[c].astype(float), 12)
    return df


def played(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["played"]].reset_index(drop=True)


def outcome_labels(df: pd.DataFrame) -> np.ndarray:
    """胜平负标签: 0=主胜 H, 1=平 D, 2=客胜 A。"""
    diff = df["home_score"].to_numpy() - df["away_score"].to_numpy()
    return np.where(diff > 0, 0, np.where(diff == 0, 1, 2))
