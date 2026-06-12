"""核心数学组件的正确性测试 (合成数据, 不依赖真实数据集)。"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from worldcup.dixon_coles import DixonColes
from worldcup.elo import compute_elo_history, expected_score
from worldcup.markets import adjust_matrix, market_report
from worldcup.model import rps


def synthetic_matches(n=3000, seed=0):
    """已知攻防强度的合成联赛, 用于验证 DC 参数还原。"""
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(12)]
    att = rng.normal(0.1, 0.3, 12)
    deff = rng.normal(0.0, 0.3, 12)
    rows = []
    d0 = pd.Timestamp("2023-01-01")
    for k in range(n):
        i, j = rng.choice(12, 2, replace=False)
        lam = np.exp(att[i] + deff[j] + 0.3)
        mu = np.exp(att[j] + deff[i])
        rows.append({
            "date": d0 + pd.Timedelta(days=k % 700),
            "home_team": teams[i], "away_team": teams[j],
            "home_score": rng.poisson(lam), "away_score": rng.poisson(mu),
            "neutral": False, "tournament": "Friendly", "importance": 40.0,
            "played": True})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True), att, deff


def test_dixon_coles_recovers_parameters():
    df, att_true, deff_true = synthetic_matches()
    dc = DixonColes(min_matches=5, window_days=10000, xi=0.0).fit(df)
    order = [dc._idx[f"T{i}"] for i in range(12)]
    att_hat = dc.att[order] - dc.att[order].mean()
    corr = np.corrcoef(att_hat, att_true - att_true.mean())[0, 1]
    assert corr > 0.9, f"进攻参数还原相关性过低: {corr:.3f}"
    assert 0.15 < dc.home_adv < 0.45, f"主场效应估计异常: {dc.home_adv:.3f}"


def test_score_matrix_consistency():
    df, *_ = synthetic_matches(1500)
    dc = DixonColes(min_matches=5, window_days=10000).fit(df)
    M = dc.score_matrix(1.6, 1.1)
    assert abs(M.sum() - 1) < 1e-9
    p = dc.match_probs(1.6, 1.1)
    assert abs(p.sum() - 1) < 1e-9 and p[0] > p[2]  # λ>μ 则主胜概率更大


def test_adjust_matrix_hits_target():
    dc = DixonColes()
    dc.rho = -0.05
    M = dc.score_matrix(1.4, 1.2)
    target = np.array([0.5, 0.3, 0.2])
    M2 = adjust_matrix(M, target)
    rep = market_report(M2)
    assert np.allclose(rep["hda"], target, atol=1e-9)
    assert abs(sum(rep["goals_dist"].values()) - 1) < 1e-6


def test_elo_zero_sum_and_expectation():
    df, *_ = synthetic_matches(500)
    out, ratings = compute_elo_history(df)
    total = sum(ratings.values())
    assert abs(total - 1500 * len(ratings)) < 1e-6  # Elo 零和
    assert expected_score(1600, 1400, True) > 0.7


def test_rps_perfect_and_uniform():
    y = np.array([0, 1, 2])
    perfect = np.eye(3)[y]
    assert rps(perfect, y) == 0.0
    uniform = np.full((3, 3), 1 / 3)
    assert 0 < rps(uniform, y) < 0.5


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
