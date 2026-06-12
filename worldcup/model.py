"""胜平负集成模型: 梯度提升 + Dixon-Coles 对数池融合 + 温度校准。

融合公式 (log-pool):
    p ∝ exp{ [ w·log p_DC + (1-w)·log p_GBM ] / T }
w (融合权重) 与 T (温度) 在验证集上以 RPS / 对数损失最优化。
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier

from .dixon_coles import DixonColes
from .features import FEATURE_COLS

_EPS = 1e-9


def rps(probs: np.ndarray, outcome: np.ndarray) -> float:
    """Ranked Probability Score (有序三分类), 越小越好。"""
    cum = np.cumsum(probs, axis=1)
    o = np.zeros_like(probs)
    o[np.arange(len(outcome)), outcome] = 1.0
    ocum = np.cumsum(o, axis=1)
    return float(np.mean(np.sum((cum - ocum) ** 2, axis=1) / (probs.shape[1] - 1)))


def log_loss_(probs: np.ndarray, outcome: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(outcome)), outcome], _EPS, 1)
    return float(-np.mean(np.log(p)))


def brier(probs: np.ndarray, outcome: np.ndarray) -> float:
    o = np.zeros_like(probs)
    o[np.arange(len(outcome)), outcome] = 1.0
    return float(np.mean(np.sum((probs - o) ** 2, axis=1)))


def _log_pool(p_dc: np.ndarray, p_ml: np.ndarray, w: float, temp: float) -> np.ndarray:
    z = (w * np.log(np.clip(p_dc, _EPS, 1))
         + (1 - w) * np.log(np.clip(p_ml, _EPS, 1))) / temp
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class Ensemble:
    gbm: HistGradientBoostingClassifier | None = None
    dc: DixonColes | None = None
    weight: float = 0.5
    temp: float = 1.0
    feature_cols: list[str] = field(default_factory=lambda: list(FEATURE_COLS))

    def fit_gbm(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self.gbm = HistGradientBoostingClassifier(
            loss="log_loss", max_iter=400, learning_rate=0.05,
            max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.12, random_state=42)
        self.gbm.fit(X[self.feature_cols], y)

    def gbm_probs(self, X: pd.DataFrame) -> np.ndarray:
        return self.gbm.predict_proba(X[self.feature_cols])

    def dc_probs_rows(self, rows: pd.DataFrame) -> np.ndarray:
        out = np.empty((len(rows), 3))
        for k, (_, r) in enumerate(rows.iterrows()):
            lam, mu = self.dc.rates(r["home_team"], r["away_team"], r["neutral"],
                                    r["elo_home"], r["elo_away"])
            out[k] = self.dc.match_probs(lam, mu)
        return out

    def tune(self, p_dc: np.ndarray, p_ml: np.ndarray, y: np.ndarray) -> None:
        """在验证集上网格搜权重 w, 再一维优化温度 T。"""
        best = (0.5, np.inf)
        for w in np.linspace(0, 1, 21):
            s = rps(_log_pool(p_dc, p_ml, w, 1.0), y)
            if s < best[1]:
                best = (w, s)
        self.weight = float(best[0])
        res = minimize_scalar(
            lambda t: log_loss_(_log_pool(p_dc, p_ml, self.weight, t), y),
            bounds=(0.5, 3.0), method="bounded")
        self.temp = float(res.x)

    def blend(self, p_dc: np.ndarray, p_ml: np.ndarray) -> np.ndarray:
        return _log_pool(p_dc, p_ml, self.weight, self.temp)

    def predict_rows(self, rows: pd.DataFrame) -> np.ndarray:
        """rows 须含特征列 + home_team/away_team/neutral/elo_*。"""
        return self.blend(self.dc_probs_rows(rows), self.gbm_probs(rows))

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "Ensemble":
        with open(path, "rb") as f:
            return pickle.load(f)
