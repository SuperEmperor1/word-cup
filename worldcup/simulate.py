"""2026 世界杯蒙特卡洛整赛事模拟 (官方支架版)。

精确实现:
  - 官方 12 组 (A-L) 与 32 强对阵表 (FIFA 2026 赛程, 比赛 73-88)
  - 第三名 8 个晋级槽位: 按官方各槽位允许组别集合做二分图完美匹配
  - 小组排名: 积分 -> 同分子集 head-to-head 积分/净胜球 -> 全组净胜球
    -> 进球数 -> 抽签
  - 末轮死局: 双方均已 >=6 分锁定出线时平局概率上调 (默契球效应)
  - 点球大战: 677 场真实数据拟合 Elo 斜率 + 先罚优势, 先罚方掷币决定
  - 参数不确定性: DC 攻防参数按 Fisher 标准误抽样 K 个"世界",
    每次模拟取一个世界 (球队强度在整届内保持一致 -> 正确的尾部相关)
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from .markets import adjust_matrix

# ---------------------------------------------------------------- 官方结构
# 锚定球队 -> 组别字母 (与 2025-12-05 官方抽签一致)
GROUP_ANCHORS = {
    "A": "Mexico", "B": "Canada", "C": "Brazil", "D": "United States",
    "E": "Germany", "F": "Netherlands", "G": "Belgium", "H": "Spain",
    "I": "France", "J": "Argentina", "K": "Portugal", "L": "England",
}

# 32 强对阵 (比赛号: (槽位1, 槽位2)); W=组头名, R=次名, T=第三名(允许组别集合)
R32 = {
    73: ("R:A", "R:B"), 74: ("W:E", "T:ABCDF"), 75: ("W:F", "R:C"),
    76: ("W:C", "R:F"), 77: ("W:I", "T:CDFGH"), 78: ("R:E", "R:I"),
    79: ("W:A", "T:CEFHI"), 80: ("W:L", "T:EHIJK"), 81: ("W:D", "T:BEFIJ"),
    82: ("W:G", "T:AEHIJ"), 83: ("R:K", "R:L"), 84: ("W:H", "R:J"),
    85: ("W:B", "T:EFGIJ"), 86: ("W:J", "R:H"), 87: ("W:K", "T:DEIJL"),
    88: ("R:D", "R:G"),
}
R16 = {89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
       93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87)}
QF = {97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96)}
SF = {101: (97, 98), 102: (99, 100)}

THIRD_SLOTS = {m: set(spec.split(":")[1]) for m, (_, s2) in R32.items()
               for spec in [s2] if spec.startswith("T:")}


def assign_thirds(qualified: list[str]) -> dict[int, str] | None:
    """8 个晋级第三名组别 -> 槽位 的二分图完美匹配 (回溯, 确定性顺序)。"""
    slots = sorted(THIRD_SLOTS, key=lambda m: len(THIRD_SLOTS[m] & set(qualified)))
    assign: dict[int, str] = {}
    used: set[str] = set()

    def bt(k):
        if k == len(slots):
            return True
        m = slots[k]
        for g in sorted(THIRD_SLOTS[m] & set(qualified) - used):
            assign[m] = g
            used.add(g)
            if bt(k + 1):
                return True
            used.discard(g)
            del assign[m]
        return False

    return assign if bt(0) else None


def infer_groups(fixtures: pd.DataFrame) -> dict[str, list[str]]:
    """由小组赛对阵推断分组 (连通分量), 并按锚定球队映射官方字母。"""
    adj = defaultdict(set)
    for _, r in fixtures.iterrows():
        adj[r["home_team"]].add(r["away_team"])
        adj[r["away_team"]].add(r["home_team"])
    seen, comps = set(), []
    for t in adj:
        if t in seen:
            continue
        comp, stack = [], [t]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp.append(u)
            stack.extend(adj[u] - seen)
        comps.append(sorted(comp))
    out = {}
    for letter, anchor in GROUP_ANCHORS.items():
        for comp in comps:
            if anchor in comp:
                out[letter] = comp
                break
    if len(out) != 12:
        raise ValueError(f"分组映射失败: 只匹配到 {len(out)} 组")
    return out


# ---------------------------------------------------------------- 点球模型
def fit_shootout_model(feat: pd.DataFrame, shootouts: pd.DataFrame
                       ) -> tuple[float, float]:
    """拟合 P(主列球队赢) = σ(b·Δelo/400 + c·先罚指示)。返回 (b, c)。"""
    m = feat.merge(
        shootouts[["date", "home_team", "away_team", "winner", "first_shooter"]],
        on=["date", "home_team", "away_team"])
    d = (m["elo_home"] - m["elo_away"]).to_numpy() / 400.0
    w = (m["winner"] == m["home_team"]).to_numpy(float)
    first = np.where(m["first_shooter"] == m["home_team"], 1.0,
                     np.where(m["first_shooter"] == m["away_team"], -1.0, 0.0))

    def nll(p):
        z = np.clip(expit(p[0] * d + p[1] * first), 1e-9, 1 - 1e-9)
        return -(w * np.log(z) + (1 - w) * np.log(1 - z)).sum()

    res = minimize(nll, [0.6, 0.1], method="Nelder-Mead")
    return float(res.x[0]), float(res.x[1])


# 兼容旧接口
def fit_shootout_slope(feat, shootouts):
    return fit_shootout_model(feat, shootouts)[0]


# ---------------------------------------------------------------- 模拟器
DEAD_RUBBER_DRAW_BOOST = 1.30   # 双方均锁定出线时的平局倍增 (再归一)


class TournamentSimulator:
    """预计算 K 个参数世界下所有两两对阵的比分分布, 再批量模拟。"""

    def __init__(self, ensemble, teams: list[str], elo: dict[str, float],
                 feature_builder, max_goals: int = 8, seed: int = 7,
                 shootout_params: tuple[float, float] = (0.6, 0.1),
                 n_worlds: int = 16):
        self.teams = teams
        self.idx = {t: i for i, t in enumerate(teams)}
        self.elo = elo
        self.rng = np.random.default_rng(seed)
        self.mg = max_goals
        self.so_b, self.so_c = shootout_params
        self.n_worlds = n_worlds
        nT = len(teams)
        dc = ensemble.dc

        rows = feature_builder(teams)
        p_ens = ensemble.predict_rows(rows)
        lam = np.empty((nT, nT))
        mu = np.empty((nT, nT))
        targ = np.empty((nT, nT, 3))
        for k, (_, r) in enumerate(rows.iterrows()):
            i, j = self.idx[r["home_team"]], self.idx[r["away_team"]]
            lam[i, j], mu[i, j] = ensemble.rates_row(r)
            targ[i, j] = p_ens[k]

        # 参数世界: 每个世界为每队抽强度扰动 (Fisher 标准误)
        att_se = np.array([dc.strength_se(t)[0] for t in teams])
        def_se = np.array([dc.strength_se(t)[1] for t in teams])
        self.cdf = np.zeros((n_worlds, nT, nT, (max_goals + 1) ** 2))
        self.draw_cdf = np.zeros_like(self.cdf)  # 平局上调版 (死局用)
        for w in range(n_worlds):
            if w == 0:
                da = np.zeros(nT)
                dd = np.zeros(nT)
            else:
                da = self.rng.normal(0, att_se)
                dd = self.rng.normal(0, def_se)
            for i in range(nT):
                for j in range(nT):
                    if i == j:
                        continue
                    lam_w = lam[i, j] * np.exp(da[i] + dd[j])
                    mu_w = mu[i, j] * np.exp(da[j] + dd[i])
                    M0 = dc.score_matrix(lam[i, j], mu[i, j], max_goals)
                    Mw = dc.score_matrix(lam_w, mu_w, max_goals)
                    # 把参数扰动转成 HDA 目标的平移, 保持集成水准
                    p0 = np.array([np.tril(M0, -1).sum(), np.trace(M0),
                                   np.triu(M0, 1).sum()])
                    pw = np.array([np.tril(Mw, -1).sum(), np.trace(Mw),
                                   np.triu(Mw, 1).sum()])
                    t = np.clip(targ[i, j] + (pw - p0), 1e-4, None)
                    M = adjust_matrix(Mw, t / t.sum())
                    self.cdf[w, i, j] = np.cumsum(M.ravel())
                    td = t.copy()
                    td[1] *= DEAD_RUBBER_DRAW_BOOST
                    Md = adjust_matrix(Mw, td / td.sum())
                    self.draw_cdf[w, i, j] = np.cumsum(Md.ravel())

    def sample_score(self, w, ti, tj, dead=False):
        cdf = self.draw_cdf if dead else self.cdf
        k = int(np.searchsorted(cdf[w, ti, tj], self.rng.random()))
        return divmod(min(k, (self.mg + 1) ** 2 - 1), self.mg + 1)

    def _penalty_win(self, ti, tj):
        d = (self.elo[self.teams[ti]] - self.elo[self.teams[tj]]) / 400.0
        first = 1.0 if self.rng.random() < 0.5 else -1.0  # 掷币决定先罚
        return self.rng.random() < expit(self.so_b * d + self.so_c * first)

    def _ko_winner(self, w, ti, tj):
        x, y = self.sample_score(w, ti, tj)
        if x != y:
            return ti if x > y else tj
        if self.rng.random() < 1 / 3:  # 加时 ~ 1/3 强度
            x2, y2 = self.sample_score(w, ti, tj)
            if x2 != y2:
                return ti if x2 > y2 else tj
        return ti if self._penalty_win(ti, tj) else tj

    # ------------------------------------------------------------ 小组排序
    def _rank_group(self, g, pts, gd, gf, results):
        """FIFA 细则: 积分 -> 同分子集 H2H 积分/净胜球 -> 全组净胜球
        -> 进球 -> 抽签。"""
        order = sorted(g, key=lambda t: -pts[t])
        final = []
        i = 0
        while i < len(order):
            tied = [t for t in order if pts[t] == pts[order[i]]
                    and t not in final]
            if len(tied) > 1:
                h2h_p = {t: 0 for t in tied}
                h2h_gd = {t: 0 for t in tied}
                for (a, b), (x, y) in results.items():
                    if a in tied and b in tied:
                        h2h_p[a] += 3 if x > y else (1 if x == y else 0)
                        h2h_p[b] += 3 if y > x else (1 if x == y else 0)
                        h2h_gd[a] += x - y
                        h2h_gd[b] += y - x
                tied.sort(key=lambda t: (h2h_p[t], h2h_gd[t], gd[t], gf[t],
                                         self.rng.random()), reverse=True)
            final.extend(tied)
            i = len(final)
        return final

    # ------------------------------------------------------------ 主循环
    def simulate(self, groups: dict[str, list[str]],
                 group_fixtures: dict[str, list[tuple[str, str, bool]]],
                 n_sims: int = 10000) -> dict:
        """group_fixtures: 组别 -> [(home, away, is_last_round), ...]"""
        champion, finalist, semis, ko32 = Counter(), Counter(), Counter(), Counter()
        for s in range(n_sims):
            w = s % self.n_worlds
            winners, runners, thirds = {}, {}, []
            for letter, g in groups.items():
                pts = {t: 0 for t in g}
                gd = {t: 0 for t in g}
                gf = {t: 0 for t in g}
                results = {}
                for home, away, last in group_fixtures[letter]:
                    dead = last and pts[home] >= 6 and pts[away] >= 6
                    ti, tj = self.idx[home], self.idx[away]
                    x, y = self.sample_score(w, ti, tj, dead=dead)
                    results[(home, away)] = (x, y)
                    pts[home] += 3 if x > y else (1 if x == y else 0)
                    pts[away] += 3 if y > x else (1 if x == y else 0)
                    gd[home] += x - y
                    gd[away] += y - x
                    gf[home] += x
                    gf[away] += y
                order = self._rank_group(g, pts, gd, gf, results)
                winners[letter] = order[0]
                runners[letter] = order[1]
                thirds.append((letter, order[2], pts[order[2]],
                               gd[order[2]], gf[order[2]]))
            thirds.sort(key=lambda r: (r[2], r[3], r[4], self.rng.random()),
                        reverse=True)
            qual3 = {letter: team for letter, team, *_ in thirds[:8]}
            slot_assign = assign_thirds(list(qual3))
            if slot_assign is None:  # 官方集合设计下不应发生
                slot_assign = {m: g for m, g in
                               zip(sorted(THIRD_SLOTS), sorted(qual3))}

            mw = {}
            for m, (s1, s2) in R32.items():
                t1 = winners[s1[2:]] if s1.startswith("W:") else \
                    runners[s1[2:]] if s1.startswith("R:") else None
                if s2.startswith("T:"):
                    t2 = qual3[slot_assign[m]]
                else:
                    t2 = winners[s2[2:]] if s2.startswith("W:") else runners[s2[2:]]
                for t in (t1, t2):
                    ko32[t] += 1
                mw[m] = self._ko_winner(w, self.idx[t1], self.idx[t2])
            for m, (a, b) in R16.items():
                mw[m] = self._ko_winner(w, mw[a], mw[b])
            for m, (a, b) in QF.items():
                mw[m] = self._ko_winner(w, mw[a], mw[b])
            for m, (a, b) in SF.items():
                for t in (mw[a], mw[b]):
                    semis[self.teams[t]] += 1
                mw[m] = self._ko_winner(w, mw[a], mw[b])
            f1, f2 = mw[101], mw[102]
            finalist[self.teams[f1]] += 1
            finalist[self.teams[f2]] += 1
            champion[self.teams[self._ko_winner(w, f1, f2)]] += 1

        def norm(c):
            return {t: v / n_sims for t, v in c.most_common() if v > 0}
        return {"champion": norm(champion), "final": norm(finalist),
                "semifinal": norm(semis), "knockout": norm(ko32)}
