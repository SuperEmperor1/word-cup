"""滚动前向回测 (walk-forward backtest) 与消融框架。

协议 (严格无泄漏):
  基模型训练 (GBM/有序Logit):  date < train_end
  元学习器 + 温度校准:          train_end <= date < valid_end
  测试:                         date >= valid_end
DC 模型在测试期内按季度重拟合, 每次仅用当时已发生的比赛。

输出三层验证:
  1. 胜平负: RPS / Brier / 对数损失 / 命中率 + 分段 (赛事级别/悬殊度)
  2. 比分与进球市场: 比分多项对数损失 / 大小球 Brier 与校准 / BTTS / 总进球 MAE
  3. 消融: 按特征族移除后重跑 (共享 DC 计算)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import outcome_labels
from .dixon_coles import DixonColes
from .markets import adjust_matrix
from .model import Ensemble, brier, log_loss_, rps

# 消融用特征族定义
FAMILIES = {
    "geo": ["travel_h", "travel_a", "travel_diff", "tz_h", "tz_a",
            "clim_mismatch_h", "clim_mismatch_a", "cumtravel_h", "cumtravel_a",
            "venue_alt", "alt_mismatch_h", "alt_mismatch_a"],
    "player": ["late_goal_h", "late_goal_a", "late_concede_h", "late_concede_a",
               "pen_share_h", "pen_share_a", "scorer_hhi_h", "scorer_hhi_a",
               "star_form_h", "star_form_a", "pen_concede_h", "pen_concede_a"],
    "progression": ["comeback_h", "comeback_a", "hold_h", "hold_a",
                    "early_goal_h", "early_goal_a",
                    "early_concede_h", "early_concede_a"],
    "context": ["vs_strong_h", "vs_strong_a", "major_gap_h", "major_gap_a",
                "elo_vol_h", "elo_vol_a", "major_exp_h", "major_exp_a"],
    "sd_rating": ["sd_att_h", "sd_def_h", "sd_att_a", "sd_def_a",
                  "sd_xg_h", "sd_xg_a"],
    "schedule": ["rest_h", "rest_a", "rest_diff", "density_h", "density_a"],
}


def _elo_baseline(rows: pd.DataFrame, draw_rate: float) -> np.ndarray:
    we = rows["elo_exp_home"].to_numpy()
    p = np.empty((len(rows), 3))
    p[:, 1] = draw_rate
    p[:, 0] = we * (1 - draw_rate)
    p[:, 2] = (1 - we) * (1 - draw_rate)
    return p


def _metrics(p, y):
    return {"rps": rps(p, y), "brier": brier(p, y),
            "log_loss": log_loss_(p, y),
            "accuracy": float((p.argmax(1) == y).mean())}


def _dc_test_walkforward(feat: pd.DataFrame, test: pd.DataFrame
                         ) -> tuple[np.ndarray, np.ndarray]:
    """季度重拟合的 DC 测试期概率与 (λ,μ)。"""
    p_dc = np.empty((len(test), 3))
    rates = np.empty((len(test), 2))
    quarters = test["date"].dt.to_period("Q")
    for q in quarters.unique():
        mask = (quarters == q).to_numpy()
        dc_q = DixonColes().fit(feat[feat["date"] < q.start_time])
        sub = test[mask]
        p_dc[mask] = Ensemble(dc=dc_q).dc_probs_rows(sub)
        for k, (_, r) in zip(np.where(mask)[0], sub.iterrows()):
            rates[k] = dc_q.rates(r["home_team"], r["away_team"], r["neutral"],
                                  r["elo_home"], r["elo_away"])
    return p_dc, rates


def walk_forward(feat: pd.DataFrame,
                 train_end: str = "2023-01-01",
                 valid_end: str = "2025-01-01",
                 ml_start: str = "2002-01-01",
                 with_markets: bool = True) -> dict:
    feat = feat[feat["played"]].reset_index(drop=True)
    y = outcome_labels(feat)

    tr = (feat["date"] >= ml_start) & (feat["date"] < train_end)
    va = (feat["date"] >= train_end) & (feat["date"] < valid_end)
    te = feat["date"] >= valid_end

    ens = Ensemble()
    ens.fit_base(feat[tr], y[tr])
    ens.dc = DixonColes().fit(feat[feat["date"] < train_end])
    ens.fit_meta(feat[va].reset_index(drop=True), y[va])

    test = feat[te].reset_index(drop=True)
    p_dc_te, rates_te = _dc_test_walkforward(feat, test)
    p_ens = ens.blend(test, p_dc=p_dc_te)
    y_te = y[te.to_numpy()]

    # 把验证期调好的 xG 混合与进球尺度应用到测试期强度
    if ens.xg_weight > 0 or ens.goal_scale != 1.0:
        w = ens.xg_weight
        xh = np.maximum(ens.xg_h.predict(test[ens.active_cols]), 0.05)
        xa = np.maximum(ens.xg_a.predict(test[ens.active_cols]), 0.05)
        rates_te[:, 0] = rates_te[:, 0] ** (1 - w) * xh ** w * ens.goal_scale
        rates_te[:, 1] = rates_te[:, 1] ** (1 - w) * xa ** w * ens.goal_scale

    draw_rate = float((y[tr] == 1).mean())
    base = np.tile([(y[tr] == 0).mean(), draw_rate, (y[tr] == 2).mean()],
                   (len(test), 1))

    report = {
        "n_test": int(len(test)),
        "ensemble": _metrics(p_ens, y_te),
        "dixon_coles": _metrics(p_dc_te, y_te),
        "gbm": _metrics(ens.gbm_probs(test), y_te),
        "ordered_logit": _metrics(ens.ol.predict_proba(test), y_te),
        "elo_only": _metrics(_elo_baseline(test, draw_rate), y_te),
        "baseline_prior": _metrics(base, y_te),
        "temperature": ens.temp,
        "xg_weight": ens.xg_weight,
        "goal_scale": ens.goal_scale,
        "calibration": calibration_table(p_ens, y_te),
        "segments": segment_metrics(test, p_ens, y_te),
    }
    if with_markets:
        report["markets"] = market_metrics(ens, test, rates_te, p_ens, feat[tr])
    return report


def segment_metrics(test: pd.DataFrame, p: np.ndarray, y: np.ndarray) -> dict:
    """分段校准检验: 赛事级别 × 实力悬殊度。"""
    out = {}
    imp = test["importance"].to_numpy()
    for name, mask in (("friendly", imp < 30),
                       ("qualifier_nations", (imp >= 30) & (imp < 50)),
                       ("major", imp >= 50)):
        if mask.sum() > 30:
            out[name] = {"n": int(mask.sum()), "rps": rps(p[mask], y[mask]),
                         "log_loss": log_loss_(p[mask], y[mask])}
    ed = np.abs(test["elo_diff"].to_numpy())
    terc = np.quantile(ed, [1 / 3, 2 / 3])
    for name, mask in (("balanced", ed <= terc[0]),
                       ("medium", (ed > terc[0]) & (ed <= terc[1])),
                       ("mismatch", ed > terc[1])):
        out[name] = {"n": int(mask.sum()), "rps": rps(p[mask], y[mask]),
                     "log_loss": log_loss_(p[mask], y[mask])}
    return out


def market_metrics(ens: Ensemble, test: pd.DataFrame, rates: np.ndarray,
                   p_hda: np.ndarray, train: pd.DataFrame) -> dict:
    """比分 / 大小球 / BTTS / 总进球 的样本外验证。"""
    hs = np.minimum(test["home_score"].to_numpy(int), 10)
    as_ = np.minimum(test["away_score"].to_numpy(int), 10)
    total = hs + as_
    n = len(test)

    sc_ll = np.empty(n)
    p_over = {1.5: np.empty(n), 2.5: np.empty(n), 3.5: np.empty(n)}
    p_btts = np.empty(n)
    exp_total = np.empty(n)
    idx = np.add.outer(np.arange(11), np.arange(11))
    for k in range(n):
        M = adjust_matrix(ens.dc.score_matrix(rates[k, 0], rates[k, 1]),
                          p_hda[k])
        sc_ll[k] = -np.log(max(M[hs[k], as_[k]], 1e-9))
        for line in p_over:
            p_over[line][k] = M[idx > line].sum()
        p_btts[k] = M[1:, 1:].sum()
        exp_total[k] = (M * idx).sum()

    # 基准: 联赛平均常数泊松
    lam0 = float(train["home_score"].mean())
    mu0 = float(train["away_score"].mean())
    from scipy.stats import poisson
    M0 = np.outer(poisson.pmf(np.arange(11), lam0),
                  poisson.pmf(np.arange(11), mu0))
    M0 /= M0.sum()
    sc_ll0 = -np.log(np.maximum(M0[hs, as_], 1e-9))

    def brier1(p, o):
        return float(np.mean((p - o) ** 2))

    out = {
        "scoreline_log_loss": float(sc_ll.mean()),
        "scoreline_log_loss_baseline": float(sc_ll0.mean()),
        "btts_brier": brier1(p_btts, ((hs > 0) & (as_ > 0)).astype(float)),
        "total_goals_mae": float(np.abs(exp_total - total).mean()),
        "totals": {},
    }
    for line, p in p_over.items():
        o = (total > line).astype(float)
        out["totals"][f"over_{line}"] = {
            "brier": brier1(p, o),
            "brier_baseline": brier1(np.full(n, float((M0[idx > line]).sum())), o),
            "calibration": _binary_calibration(p, o),
        }
    return out


def _binary_calibration(p: np.ndarray, o: np.ndarray, bins: int = 5) -> list:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    rows = []
    for k in range(bins):
        m = (p >= edges[k]) & ((p < edges[k + 1]) | (k == bins - 1))
        if m.sum() > 0:
            rows.append({"pred": round(float(p[m].mean()), 3),
                         "actual": round(float(o[m].mean()), 3),
                         "n": int(m.sum())})
    return rows


def ablation(feat: pd.DataFrame, families: dict | None = None,
             train_end: str = "2023-01-01", valid_end: str = "2025-01-01"
             ) -> dict:
    """按特征族移除后重跑 GBM+元学习器 (DC 概率只计算一次, 共享)。"""
    from .features import FEATURE_COLS
    families = families or FAMILIES
    feat = feat[feat["played"]].reset_index(drop=True)
    y = outcome_labels(feat)
    tr = (feat["date"] >= "2002-01-01") & (feat["date"] < train_end)
    va = (feat["date"] >= train_end) & (feat["date"] < valid_end)
    te = feat["date"] >= valid_end
    test = feat[te].reset_index(drop=True)
    y_te = y[te.to_numpy()]

    dc_full = DixonColes().fit(feat[feat["date"] < train_end])
    p_dc_va = Ensemble(dc=dc_full).dc_probs_rows(feat[va])
    p_dc_te, _ = _dc_test_walkforward(feat, test)

    def run(cols):
        ens = Ensemble(feature_cols=cols, dc=dc_full)
        ens.fit_base(feat[tr], y[tr])
        ens.fit_meta(feat[va].reset_index(drop=True), y[va], p_dc=p_dc_va)
        return rps(ens.blend(test, p_dc=p_dc_te), y_te)

    base_rps = run(list(FEATURE_COLS))
    out = {"baseline_rps": base_rps, "families": {}}
    for name, cols in families.items():
        kept = [c for c in FEATURE_COLS if c not in cols]
        r = run(kept)
        out["families"][name] = {
            "rps_without": r,
            "contribution": r - base_rps,  # >0 = 该族有正贡献
        }
    return out


def calibration_table(probs: np.ndarray, y: np.ndarray, bins: int = 8) -> list:
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
