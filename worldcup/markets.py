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
