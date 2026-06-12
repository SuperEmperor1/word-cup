"""比分矩阵 -> 各市场概率, 并用 IPF 使比分分布与集成胜平负自洽。

IPF (迭代比例拟合): 对比分矩阵的 主胜/平/客胜 三个区域分别缩放,
使区域概率和精确等于集成模型输出的胜平负概率, 保证
"精确比分 / 进球数 / 胜平负" 三类预测内部一致。
"""
from __future__ import annotations

import numpy as np

GOAL_LINES = (0.5, 1.5, 2.5, 3.5, 4.5)


def adjust_matrix(M: np.ndarray, target_hda: np.ndarray) -> np.ndarray:
    """缩放比分矩阵使 W/D/L 区域和等于 target_hda。"""
    lo = np.tril(np.ones_like(M), -1)          # 主胜区域
    di = np.eye(M.shape[0])                    # 平局区域
    up = np.triu(np.ones_like(M), 1)           # 客胜区域
    cur = np.array([(M * lo).sum(), (M * di).sum(), (M * up).sum()])
    scale = np.divide(target_hda, np.maximum(cur, 1e-12))
    out = M * (lo * scale[0] + di * scale[1] + up * scale[2])
    return out / out.sum()


def market_report(M: np.ndarray, top_scores: int = 6) -> dict:
    """从 (已调整的) 比分矩阵导出全部市场。"""
    n = M.shape[0]
    hda = [float(np.tril(M, -1).sum()), float(np.trace(M)),
           float(np.triu(M, 1).sum())]

    totals = {}
    idx = np.add.outer(np.arange(n), np.arange(n))
    for line in GOAL_LINES:
        totals[f"over_{line}"] = float(M[idx > line].sum())

    goals_dist = {g: float(M[idx == g].sum()) for g in range(8)}
    goals_dist[8] = float(M[idx >= 8].sum())  # 8+ 球归并
    exp_total = float((M * idx).sum())

    btts = float(M[1:, 1:].sum())

    flat = [((i, j), float(M[i, j])) for i in range(n) for j in range(n)]
    flat.sort(key=lambda t: -t[1])

    return {
        "hda": hda,
        "totals": totals,
        "goals_dist": goals_dist,
        "expected_goals_total": exp_total,
        "btts": btts,
        "top_scores": flat[:top_scores],
    }


# ------------------------------------------------------------- 亚洲让球盘
def asian_handicap(M: np.ndarray, line: float) -> dict:
    """主队让 line 球 (负数=让球) 的 赢/走水/输 概率, 四分之一盘自动拆注。"""
    q = round(line * 4)
    if abs(line * 4 - q) > 1e-9:
        raise ValueError("盘口须为 0.25 的倍数")
    if q % 2 != 0:  # 四分之一盘 = 相邻两个半盘各半注
        a = asian_handicap(M, line - 0.25)
        b = asian_handicap(M, line + 0.25)
        return {k: (a[k] + b[k]) / 2 for k in ("win", "push", "lose")}
    n = M.shape[0]
    diff = np.subtract.outer(np.arange(n), np.arange(n)) + line
    return {"win": float(M[diff > 1e-9].sum()),
            "push": float(M[np.abs(diff) <= 1e-9].sum()),
            "lose": float(M[diff < -1e-9].sum())}


def ah_lines(M: np.ndarray) -> dict[str, dict]:
    """围绕期望净胜球的常用盘口表 (含公平赔率)。"""
    n = M.shape[0]
    exp_margin = float((M * np.subtract.outer(np.arange(n), np.arange(n))).sum())
    center = round(-exp_margin * 4) / 4
    out = {}
    for line in (center - 0.5, center - 0.25, center, center + 0.25, center + 0.5):
        r = asian_handicap(M, line)
        eff = r["win"] + r["lose"]
        out[f"{line:+.2f}"] = {**{k: round(v, 4) for k, v in r.items()},
                               "fair_odds": round(eff / r["win"], 3)
                               if r["win"] > 1e-9 else None}
    return out


# ------------------------------------------------------------- 半场市场
FIRST_HALF_SHARE = 0.4363  # 上半场进球占比 (2000 年后 28,399 球实测)


def half_time_report(dc, lam: float, mu: float, max_goals: int = 6) -> dict:
    """半场 1X2 / 大小球 / 半全场 (HT/FT) 九宫格。

    近似: 上下半场进球独立泊松, 强度按 FIRST_HALF_SHARE 分配;
    半场矩阵沿用 DC 低比分修正。
    """
    l1, m1 = lam * FIRST_HALF_SHARE, mu * FIRST_HALF_SHARE
    l2, m2 = lam - l1, mu - m1
    M1 = dc.score_matrix(l1, m1, max_goals)
    M2 = dc.score_matrix(l2, m2, max_goals)
    ht = [float(np.tril(M1, -1).sum()), float(np.trace(M1)),
          float(np.triu(M1, 1).sum())]
    idx = np.add.outer(np.arange(max_goals + 1), np.arange(max_goals + 1))
    ht_over = {f"over_{ln}": float(M1[idx > ln].sum()) for ln in (0.5, 1.5)}

    # HT/FT 联合: 枚举 (半场比分) x (下半场比分)
    htft = np.zeros((3, 3))
    sgn = np.sign(np.subtract.outer(np.arange(max_goals + 1),
                                    np.arange(max_goals + 1)))
    for a in range(max_goals + 1):
        for b in range(max_goals + 1):
            p1 = M1[a, b]
            if p1 < 1e-7:
                continue
            ht_o = 0 if a > b else (1 if a == b else 2)
            for c in range(max_goals + 1):
                for d in range(max_goals + 1):
                    ft = a + c - b - d
                    ft_o = 0 if ft > 0 else (1 if ft == 0 else 2)
                    htft[ht_o, ft_o] += p1 * M2[c, d]
    htft /= htft.sum()
    return {"ht_hda": ht, "ht_totals": ht_over,
            "ht_ft": {f"{x}/{y}": round(float(htft[i, j]), 4)
                      for i, x in enumerate("HDA") for j, y in enumerate("HDA")}}
