"""滚动前向回测 (walk-forward backtest)。

协议 (严格无泄漏):
  基模型训练 (GBM/有序Logit):  date < train_end
  元学习器 + 温度校准:          train_end <= date < valid_end
  测试:                         date >= valid_end
DC 模型在测试期内按季度重拟合, 每次仅用当时已发生的比赛。
指标: RPS / Brier / 对数损失 / 命中率, 与 频率先验 和 纯Elo 基准对比。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import outcome_labels
from .dixon_coles import DixonColes
from .model import Ensemble, brier, log_loss_, rps


def _elo_baseline(rows: pd.DataFrame, draw_rate: float) -> np.ndarray:
    """纯 Elo 基准: 期望胜率按固定平局率拆分。"""
    we = rows["elo_exp_home"].to_numpy()
    p = np.empty((len(rows), 3))
    p[:, 1] = draw_rate
    p[:, 0] = we * (1 - draw_rate)
    p[:, 2] = (1 - we) * (1 - draw_rate)
    return p


def walk_forward(feat: pd.DataFrame,
                 train_end: str = "2023-01-01",
                 valid_end: str = "2025-01-01",
                 ml_start: str = "2002-01-01") -> dict:
    """feat: 已完赛 + 已建特征的 DataFrame。返回各模型测试期指标。"""
    feat = feat[feat["played"]].reset_index(drop=True)
    y = outcome_labels(feat)

    tr = (feat["date"] >= ml_start) & (feat["date"] < train_end)
    va = (feat["date"] >= train_end) & (feat["date"] < valid_end)
    te = feat["date"] >= valid_end

    ens = Ensemble()
    ens.fit_base(feat[tr], y[tr])

    # 验证集: DC 拟合至 train_end, 元学习器在验证期训练
    ens.dc = DixonColes().fit(feat[feat["date"] < train_end])
    ens.fit_meta(feat[va].reset_index(drop=True), y[va])

    # 测试集: DC 按季度 walk-forward 重拟合
    test = feat[te].reset_index(drop=True)
    p_dc_te = np.empty((len(test), 3))
    quarters = test["date"].dt.to_period("Q")
    for q in quarters.unique():
        mask = (quarters == q).to_numpy()
        dc_q = DixonColes().fit(feat[feat["date"] < q.start_time])
        p_dc_te[mask] = Ensemble(dc=dc_q).dc_probs_rows(test[mask])
    p_ml_te = ens.gbm_probs(test)
    p_ol_te = ens.ol.predict_proba(test)
    p_ens = ens.blend(test, p_dc=p_dc_te)

    y_te = y[te.to_numpy()]
    draw_rate = float((y[tr] == 1).mean())
    base = np.tile([(y[tr] == 0).mean(), draw_rate, (y[tr] == 2).mean()],
                   (len(test), 1))

    def metrics(p):
        return {"rps": rps(p, y_te), "brier": brier(p, y_te),
                "log_loss": log_loss_(p, y_te),
                "accuracy": float((p.argmax(1) == y_te).mean())}

    return {
        "n_test": int(len(test)),
        "ensemble": metrics(p_ens),
        "dixon_coles": metrics(p_dc_te),
        "gbm": metrics(p_ml_te),
        "ordered_logit": metrics(p_ol_te),
        "elo_only": metrics(_elo_baseline(test, draw_rate)),
        "baseline_prior": metrics(base),
        "temperature": ens.temp,
        "calibration": calibration_table(p_ens, y_te),
    }


def calibration_table(probs: np.ndarray, y: np.ndarray, bins: int = 8) -> list:
    """主胜概率分桶 vs 实际主胜频率。"""
    p = probs[:, 0]
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    rows = []
    for k in range(bins):
        m = (p >= edges[k]) & (p <= edges[k + 1] if k == bins - 1 else p < edges[k + 1])
        if m.sum() > 0:
            rows.append({"bucket": f"{edges[k]:.2f}-{edges[k+1]:.2f}",
                         "n": int(m.sum()),
                         "pred": float(p[m].mean()),
                         "actual": float((y[m] == 0).mean())})
    return rows
