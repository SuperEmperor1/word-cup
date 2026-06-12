"""胜平负集成模型 (v2): 三基模型 + Stacking 元学习器 + 温度校准。

基模型 (结构互补):
  1. Dixon-Coles 双变量泊松     —— 生成式比分模型
  2. HistGradientBoosting (52 维特征) —— 非线性判别模型
  3. 有序 Logit (ordered logit) —— 利用 负<平<胜 的有序结构的稳健参数模型

元学习器: 多项 Logistic 回归, 输入三个基模型的对数概率 + 情境特征,
在验证期用 5 折交叉预测训练以避免过拟合; 最终输出经温度缩放校准。
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from .dixon_coles import DixonColes
from .features import FEATURE_COLS

_EPS = 1e-9


# ------------------------------------------------------------------ 指标
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


# ------------------------------------------------------- 有序 Logit 基模型
OL_COLS = ["elo_diff", "form_pts_h", "form_pts_a", "sd_xg_h", "sd_xg_a",
           "h2h_gd", "neutral_f"]


@dataclass
class OrderedLogit:
    """潜变量 s = Xβ, 切点 c1<c2:  P(客胜)=σ(c1−s), P(平)=σ(c2−s)−σ(c1−s)。"""
    beta: np.ndarray | None = None
    cuts: tuple[float, float] = (-0.5, 0.5)
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def _design(self, X: pd.DataFrame) -> np.ndarray:
        Z = X[OL_COLS].to_numpy(float)
        Z = np.nan_to_num(Z, nan=0.0)
        return (Z - self.mean_) / self.std_

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "OrderedLogit":
        Z0 = np.nan_to_num(X[OL_COLS].to_numpy(float), nan=0.0)
        self.mean_, self.std_ = Z0.mean(0), np.maximum(Z0.std(0), 1e-6)
        Z = self._design(X)
        k = Z.shape[1]

        def nll(p):
            beta, c1, gap = p[:k], p[k], np.exp(p[k + 1])
            s = Z @ beta
            pa = expit(c1 - s)
            pad = expit(c1 + gap - s)
            probs = np.stack([1 - pad, pad - pa, pa], axis=1)
            return -np.log(np.clip(probs[np.arange(len(y)), y], _EPS, 1)).sum()

        p0 = np.zeros(k + 2)
        p0[k], p0[k + 1] = -0.5, np.log(1.0)
        res = minimize(nll, p0, method="L-BFGS-B")
        self.beta = res.x[:k]
        self.cuts = (res.x[k], res.x[k] + np.exp(res.x[k + 1]))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        s = self._design(X) @ self.beta
        pa = expit(self.cuts[0] - s)
        pad = expit(self.cuts[1] - s)
        return np.stack([1 - pad, pad - pa, pa], axis=1)


# ------------------------------------------------------------- 集成模型
META_CTX = ["importance", "neutral_f"]


@dataclass
class Ensemble:
    gbm: HistGradientBoostingClassifier | None = None
    dc: DixonColes | None = None
    ol: OrderedLogit | None = None
    meta: LogisticRegression | None = None
    temp: float = 1.0
    feature_cols: list[str] = field(default_factory=lambda: list(FEATURE_COLS))
    active_cols: list[str] = field(default_factory=list)

    # ------------------------------------------------ 基模型
    def fit_base(self, X: pd.DataFrame, y: np.ndarray) -> None:
        # 训练期内无任何观测的特征列 (如尚未积累的外部数据快照) 剔除:
        # 树模型从全缺失列学不到分裂, 且会让分箱崩溃; 赔率走市场锚定通路。
        self.active_cols = [c for c in self.feature_cols if X[c].notna().any()]
        self.gbm = HistGradientBoostingClassifier(
            loss="log_loss", max_iter=500, learning_rate=0.05,
            max_leaf_nodes=31, l2_regularization=1.0, min_samples_leaf=40,
            early_stopping=True, validation_fraction=0.12, random_state=42)
        self.gbm.fit(X[self.active_cols], y)
        self.ol = OrderedLogit().fit(X, y)

    def gbm_probs(self, X: pd.DataFrame) -> np.ndarray:
        return self.gbm.predict_proba(X[self.active_cols])

    def dc_probs_rows(self, rows: pd.DataFrame) -> np.ndarray:
        out = np.empty((len(rows), 3))
        for k, (_, r) in enumerate(rows.iterrows()):
            lam, mu = self.dc.rates(r["home_team"], r["away_team"], r["neutral"],
                                    r["elo_home"], r["elo_away"])
            out[k] = self.dc.match_probs(lam, mu)
        return out

    def base_probs(self, rows: pd.DataFrame, p_dc: np.ndarray | None = None
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if p_dc is None:
            p_dc = self.dc_probs_rows(rows)
        return p_dc, self.gbm_probs(rows), self.ol.predict_proba(rows)

    # ------------------------------------------------ 元学习器
    @staticmethod
    def _meta_design(p_dc, p_ml, p_ol, rows: pd.DataFrame) -> np.ndarray:
        logs = [np.log(np.clip(p, _EPS, 1)) for p in (p_dc, p_ml, p_ol)]
        ctx = np.column_stack([rows["importance"].to_numpy(float) / 60.0,
                               rows["neutral_f"].to_numpy(float)])
        return np.column_stack(logs + [ctx])

    def fit_meta(self, rows: pd.DataFrame, y: np.ndarray,
                 p_dc: np.ndarray | None = None) -> None:
        p_dc, p_ml, p_ol = self.base_probs(rows, p_dc)
        Z = self._meta_design(p_dc, p_ml, p_ol, rows)
        self.meta = LogisticRegression(C=1.0, max_iter=2000)
        # 交叉预测上拟合温度, 避免在元训练数据上自我评估
        cv_pred = cross_val_predict(LogisticRegression(C=1.0, max_iter=2000),
                                    Z, y, cv=5, method="predict_proba")
        self.meta.fit(Z, y)
        res = minimize_scalar(
            lambda t: log_loss_(self._temper(cv_pred, t), y),
            bounds=(0.6, 2.5), method="bounded")
        self.temp = float(res.x)

    @staticmethod
    def _temper(p: np.ndarray, t: float) -> np.ndarray:
        z = np.log(np.clip(p, _EPS, 1)) / t
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def blend(self, rows: pd.DataFrame, p_dc: np.ndarray | None = None) -> np.ndarray:
        p_dc, p_ml, p_ol = self.base_probs(rows, p_dc)
        Z = self._meta_design(p_dc, p_ml, p_ol, rows)
        return self._temper(self.meta.predict_proba(Z), self.temp)

    def predict_rows(self, rows: pd.DataFrame,
                     market_anchor: float = 0.30) -> np.ndarray:
        """rows 须含特征列 + home_team/away_team/neutral/elo_*。

        market_anchor: 若行内含去水市场概率 (mkt_*), 以该权重与模型线性池。
        博彩收盘价是文献公认最强单一预测源; 在历史赔率快照不足以让 GBM
        从特征通路学习之前, 固定权重锚定是无需训练的稳健利用方式。
        """
        p = self.blend(rows)
        if market_anchor > 0 and "mkt_ph" in rows.columns:
            mkt = rows[["mkt_ph", "mkt_pd", "mkt_pa"]].to_numpy(float)
            has = ~np.isnan(mkt).any(axis=1)
            if has.any():
                p[has] = (1 - market_anchor) * p[has] + market_anchor * mkt[has]
        return p

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "Ensemble":
        with open(path, "rb") as f:
            return pickle.load(f)
