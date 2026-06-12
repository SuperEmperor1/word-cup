"""2026 世界杯蒙特卡洛整赛事模拟。

- 小组从赛程自动推断 (小组赛对阵图的连通分量, 12 组 × 4 队)
- 小组排名: 积分 -> 净胜球 -> 进球数 -> 随机 (FIFA 抽签近似)
- 48 队赛制: 每组前 2 + 8 个最佳第三 共 32 队进入淘汰赛
- 淘汰赛对阵: 按小组排名蛇形种子排列 (1v32, 2v31, ...), 同组规避,
  这是对官方支架的近似 (官方第三名分配规则依赖具体组别组合)
- 90 分钟用集成校准后的比分分布抽样; 淘汰赛平局进入
  加时(按 90 分钟强度的 1/3 重新抽样) -> 点球 (按 Elo 微调的胜率)
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .markets import adjust_matrix


def infer_groups(fixtures: pd.DataFrame) -> list[list[str]]:
    """由小组赛对阵推断分组 (连通分量)。"""
    adj = defaultdict(set)
    for _, r in fixtures.iterrows():
        adj[r["home_team"]].add(r["away_team"])
        adj[r["away_team"]].add(r["home_team"])
    seen, groups = set(), []
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
        groups.append(sorted(comp))
    return sorted(groups)


class TournamentSimulator:
    """预先计算所有球队两两对阵的 (经胜平负校准的) 比分分布, 再批量模拟。"""

    def __init__(self, ensemble, teams: list[str], elo: dict[str, float],
                 feature_builder, max_goals: int = 8, seed: int = 7):
        self.teams = teams
        self.idx = {t: i for i, t in enumerate(teams)}
        self.elo = elo
        self.rng = np.random.default_rng(seed)
        self.mg = max_goals
        n = len(teams)
        self.cdf = np.zeros((n, n, (max_goals + 1) ** 2))

        rows = feature_builder(teams)  # 所有有序对的特征行
        p_ens = ensemble.predict_rows(rows)
        for k, (_, r) in enumerate(rows.iterrows()):
            i, j = self.idx[r["home_team"]], self.idx[r["away_team"]]
            lam, mu = ensemble.dc.rates(r["home_team"], r["away_team"], True,
                                        r["elo_home"], r["elo_away"])
            M = ensemble.dc.score_matrix(lam, mu, max_goals)
            M = adjust_matrix(M, p_ens[k])
            self.cdf[i, j] = np.cumsum(M.ravel())

    def sample_score(self, ti: int, tj: int) -> tuple[int, int]:
        u = self.rng.random()
        k = int(np.searchsorted(self.cdf[ti, tj], u))
        return divmod(min(k, (self.mg + 1) ** 2 - 1), self.mg + 1)

    def _penalty_win(self, ti: int, tj: int) -> bool:
        d = self.elo[self.teams[ti]] - self.elo[self.teams[tj]]
        return self.rng.random() < 1 / (1 + 10 ** (-d / 1200))

    def _ko_winner(self, ti: int, tj: int) -> int:
        x, y = self.sample_score(ti, tj)
        if x != y:
            return ti if x > y else tj
        # 加时: 以 1/3 强度近似 (再抽一场, 1/3 概率采纳其净胜关系)
        if self.rng.random() < 1 / 3:
            x2, y2 = self.sample_score(ti, tj)
            if x2 != y2:
                return ti if x2 > y2 else tj
        return ti if self._penalty_win(ti, tj) else tj

    def simulate(self, groups: list[list[str]], n_sims: int = 10000) -> dict:
        champion, finalist, semis, ko32 = Counter(), Counter(), Counter(), Counter()
        for _ in range(n_sims):
            winners, runners, thirds = [], [], []
            for g in groups:
                pts = {t: 0 for t in g}
                gd = {t: 0 for t in g}
                gf = {t: 0 for t in g}
                for a in range(4):
                    for b in range(a + 1, 4):
                        ti, tj = self.idx[g[a]], self.idx[g[b]]
                        x, y = self.sample_score(ti, tj)
                        pts[g[a]] += 3 if x > y else (1 if x == y else 0)
                        pts[g[b]] += 3 if y > x else (1 if x == y else 0)
                        gd[g[a]] += x - y
                        gd[g[b]] += y - x
                        gf[g[a]] += x
                        gf[g[b]] += y
                order = sorted(g, key=lambda t: (pts[t], gd[t], gf[t],
                                                 self.rng.random()), reverse=True)
                winners.append(order[0])
                runners.append(order[1])
                thirds.append((order[2], pts[order[2]], gd[order[2]], gf[order[2]]))
            thirds.sort(key=lambda r: (r[1], r[2], r[3], self.rng.random()),
                        reverse=True)
            best3 = [t for t, *_ in thirds[:8]]

            # 蛇形种子: 组头名按 Elo 排前 12, 其后次名+第三
            seeds = sorted(winners, key=lambda t: -self.elo[t]) \
                + sorted(runners + best3, key=lambda t: -self.elo[t])
            field32 = seeds
            for t in field32:
                ko32[t] += 1
            rnd = [self.idx[t] for t in field32]
            # 1v32, 2v31 ... 折叠配对逐轮淘汰
            while len(rnd) > 1:
                nxt = []
                for k in range(len(rnd) // 2):
                    nxt.append(self._ko_winner(rnd[k], rnd[len(rnd) - 1 - k]))
                if len(rnd) == 4:
                    for t in rnd:
                        semis[self.teams[t]] += 1
                if len(rnd) == 2:
                    for t in rnd:
                        finalist[self.teams[t]] += 1
                rnd = nxt
            champion[self.teams[rnd[0]]] += 1

        def norm(c):
            return {t: v / n_sims for t, v in c.most_common()}
        return {"champion": norm(champion), "final": norm(finalist),
                "semifinal": norm(semis), "knockout": norm(ko32)}
