"""时间衰减 Dixon-Coles 双变量泊松模型 (Dixon & Coles, 1997)。

进球强度:
    log λ = att_h + def_a + γ·home     (主队期望进球)
    log μ  = att_a + def_h              (客队期望进球)
att 为进攻强度, def 为防守弱度 (越大失球越多), γ 为主场效应。

低比分相关性修正 τ_ρ(x, y):
    (0,0): 1-λμρ   (1,0): 1+μρ   (0,1): 1+λρ   (1,1): 1-ρ   其余: 1

加权极大似然:
    w_m = exp(-ξ·Δt_m) · imp_m     (时间衰减 × 赛事重要性)
使用解析梯度 + L-BFGS-B, 防守强度均值约束保证可识别性。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

# 时间衰减: 半衰期约 2 年, 适配国际比赛的稀疏节奏
DEFAULT_XI = np.log(2) / 730.0
MAX_GOALS = 10  # 比分矩阵截断

_EPS = 1e-9


def importance_weight(k: np.ndarray) -> np.ndarray:
    """K 因子 -> 似然权重: 友谊赛信息量低, 降权; 大赛升权。"""
    return np.clip(k / 40.0, 0.5, 1.5)


@dataclass
class DixonColes:
    xi: float = DEFAULT_XI
    window_days: int = 365 * 8     # 仅用最近 8 年比赛拟合
    min_matches: int = 10          # 场次过少的球队不参与拟合
    l2: float = 1.0                # 轻度 L2 正则, 稳定弱样本球队
    teams: list[str] = field(default_factory=list)
    att: np.ndarray | None = None
    deff: np.ndarray | None = None
    home_adv: float = 0.25
    rho: float = -0.05
    _idx: dict[str, int] = field(default_factory=dict)
    # Elo 后备模型 (覆盖不在拟合集中的球队): log λ = a + b·elo_diff
    fallback_coef: tuple[float, float] = (0.05, 0.0017)
    fallback_base: float = 1.25

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame, as_of: pd.Timestamp | None = None) -> "DixonColes":
        """df 需含 date/home_team/away_team/home_score/away_score/neutral/importance。"""
        as_of = as_of or df["date"].max()
        d = df[(df["date"] <= as_of)
               & (df["date"] > as_of - pd.Timedelta(days=self.window_days))]

        counts = pd.concat([d["home_team"], d["away_team"]]).value_counts()
        keep = set(counts[counts >= self.min_matches].index)
        d = d[d["home_team"].isin(keep) & d["away_team"].isin(keep)]

        self.teams = sorted(keep)
        self._idx = {t: i for i, t in enumerate(self.teams)}
        nT = len(self.teams)

        hi = d["home_team"].map(self._idx).to_numpy()
        ai = d["away_team"].map(self._idx).to_numpy()
        x = d["home_score"].to_numpy(float)
        y = d["away_score"].to_numpy(float)
        home = (~d["neutral"].to_numpy()).astype(float)
        dt = (as_of - d["date"]).dt.days.to_numpy(float)
        w = np.exp(-self.xi * dt) * importance_weight(d["importance"].to_numpy(float))

        def unpack(p):
            return p[:nT], p[nT:2 * nT], p[2 * nT], p[2 * nT + 1]

        def nll_grad(p):
            att, deff, gamma, rho = unpack(p)
            log_lam = att[hi] + deff[ai] + gamma * home
            log_mu = att[ai] + deff[hi]
            lam, mu = np.exp(log_lam), np.exp(log_mu)

            # τ 修正及其梯度
            tau = np.ones_like(lam)
            g_ll = x - lam          # ∂LL/∂logλ (泊松部分)
            g_lm = y - mu
            g_rho = np.zeros_like(lam)

            m00 = (x == 0) & (y == 0)
            t00 = np.maximum(1.0 - lam[m00] * mu[m00] * rho, _EPS)
            tau[m00] = t00
            g_ll[m00] += -lam[m00] * mu[m00] * rho / t00
            g_lm[m00] += -lam[m00] * mu[m00] * rho / t00
            g_rho[m00] = -lam[m00] * mu[m00] / t00

            m10 = (x == 1) & (y == 0)
            t10 = 1.0 + mu[m10] * rho
            tau[m10] = t10
            g_lm[m10] += mu[m10] * rho / t10
            g_rho[m10] = mu[m10] / t10

            m01 = (x == 0) & (y == 1)
            t01 = 1.0 + lam[m01] * rho
            tau[m01] = t01
            g_ll[m01] += lam[m01] * rho / t01
            g_rho[m01] = lam[m01] / t01

            m11 = (x == 1) & (y == 1)
            tau[m11] = 1.0 - rho
            g_rho[m11] = -1.0 / (1.0 - rho)

            ll = w * (x * log_lam - lam + y * log_mu - mu + np.log(np.maximum(tau, _EPS)))
            nll = -ll.sum()

            wg_ll, wg_lm = w * g_ll, w * g_lm
            g_att = -(np.bincount(hi, wg_ll, nT) + np.bincount(ai, wg_lm, nT))
            g_def = -(np.bincount(ai, wg_ll, nT) + np.bincount(hi, wg_lm, nT))
            g_gamma = -(wg_ll * home).sum()
            g_rho_tot = -(w * g_rho).sum()

            # 可识别性: 防守均值约束 + 轻度 L2
            nll += 1e4 * deff.mean() ** 2 + self.l2 * (att @ att + deff @ deff)
            g_att += 2 * self.l2 * att
            g_def += 2e4 * deff.mean() / nT + 2 * self.l2 * deff

            return nll, np.concatenate([g_att, g_def, [g_gamma, g_rho_tot]])

        p0 = np.concatenate([np.full(nT, 0.1), np.zeros(nT), [0.25, -0.05]])
        bounds = [(-3, 3)] * (2 * nT) + [(0, 1), (-0.15, 0.10)]
        res = minimize(nll_grad, p0, jac=True, method="L-BFGS-B",
                       bounds=bounds, options={"maxiter": 500})
        self.att, self.deff, self.home_adv, self.rho = unpack(res.x)
        self._fit_fallback(d)
        return self

    def _fit_fallback(self, d: pd.DataFrame) -> None:
        """用 Elo 差 -> 进球数 的对数线性回归覆盖拟合集外的球队。"""
        if "elo_home" not in d.columns:
            return
        eh = d["elo_home"].to_numpy() + np.where(d["neutral"], 0, 80.0)
        ea = d["elo_away"].to_numpy()
        diff = np.concatenate([eh - ea, ea - eh])
        goals = np.concatenate([d["home_score"].to_numpy(float),
                                d["away_score"].to_numpy(float)])
        X = np.column_stack([np.ones_like(diff), diff / 400.0])
        # 泊松回归 (IRLS, 几步即可收敛)
        beta = np.array([np.log(max(goals.mean(), 0.2)), 0.6])
        for _ in range(25):
            lam = np.exp(np.clip(X @ beta, -3, 3))
            grad = X.T @ (goals - lam)
            hess = (X * lam[:, None]).T @ X
            beta = beta + np.linalg.solve(hess + 1e-6 * np.eye(2), grad)
        self.fallback_base = float(np.exp(beta[0]))
        self.fallback_coef = (float(beta[0]), float(beta[1]))

    # -------------------------------------------------------------- predict
    def rates(self, home_team: str, away_team: str, neutral: bool = True,
              elo_home: float | None = None, elo_away: float | None = None
              ) -> tuple[float, float]:
        """返回 (λ, μ) 期望进球。拟合集外的球队回退到 Elo 泊松回归。"""
        i, j = self._idx.get(home_team), self._idx.get(away_team)
        gamma = 0.0 if neutral else self.home_adv
        if i is not None and j is not None:
            lam = float(np.exp(self.att[i] + self.deff[j] + gamma))
            mu = float(np.exp(self.att[j] + self.deff[i]))
            return lam, mu
        if elo_home is None or elo_away is None:
            raise ValueError(f"球队不在模型中且未提供 Elo: {home_team} vs {away_team}")
        a, b = self.fallback_coef
        eh = elo_home + (0.0 if neutral else 80.0)
        lam = float(np.exp(np.clip(a + b * (eh - elo_away) / 400.0, -3, 3)))
        mu = float(np.exp(np.clip(a + b * (elo_away - eh) / 400.0, -3, 3)))
        return lam, mu

    def score_matrix(self, lam: float, mu: float, max_goals: int = MAX_GOALS
                     ) -> np.ndarray:
        """P(主队 x 球, 客队 y 球) 联合概率矩阵, 含 τ_ρ 低比分修正。"""
        gx = poisson.pmf(np.arange(max_goals + 1), lam)
        gy = poisson.pmf(np.arange(max_goals + 1), mu)
        M = np.outer(gx, gy)
        rho = self.rho
        M[0, 0] *= max(1 - lam * mu * rho, _EPS)
        M[1, 0] *= 1 + mu * rho
        M[0, 1] *= 1 + lam * rho
        M[1, 1] *= 1 - rho
        return M / M.sum()

    def match_probs(self, lam: float, mu: float) -> np.ndarray:
        """[P(主胜), P(平), P(客胜)]"""
        M = self.score_matrix(lam, mu)
        return np.array([np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()])
