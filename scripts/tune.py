"""系统超参调优 (时间序列协议: 在 2023-24 验证期上评估, 不碰测试期)。

调参对象:
  - DC: 时间衰减半衰期 xi, 拟合窗口
  - GBM: 学习率 x 叶子数
用法: python scripts/tune.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.ensemble import HistGradientBoostingClassifier

from worldcup.data import outcome_labels
from worldcup.dixon_coles import DixonColes
from worldcup.model import Ensemble, log_loss_, rps
from worldcup.pipeline import prepare

TRAIN_END, VALID_END = "2023-01-01", "2025-01-01"


def main():
    feat, _ = prepare()
    feat = feat[feat["played"]].reset_index(drop=True)
    y = outcome_labels(feat)
    tr = (feat["date"] >= "2002-01-01") & (feat["date"] < TRAIN_END)
    va = (feat["date"] >= TRAIN_END) & (feat["date"] < VALID_END)
    valid = feat[va].reset_index(drop=True)
    y_va = y[va.to_numpy()]

    print("== DC: 半衰期 x 窗口 (验证集 RPS) ==")
    best_dc = (None, np.inf)
    for half_life in (365, 548, 730, 1095):
        for years in (6, 8, 10):
            dc = DixonColes(xi=np.log(2) / half_life,
                            window_days=365 * years)
            dc.fit(feat[feat["date"] < TRAIN_END])
            p = Ensemble(dc=dc).dc_probs_rows(valid)
            r = rps(p, y_va)
            tag = f"half_life={half_life}d window={years}y"
            print(f"  {tag:<32} RPS={r:.5f}")
            if r < best_dc[1]:
                best_dc = (tag, r)
    print(f"  最优: {best_dc[0]} ({best_dc[1]:.5f})")

    print("== GBM: 学习率 x 叶子数 (验证集对数损失) ==")
    ens = Ensemble()
    cols_probe = Ensemble()
    cols_probe.active_cols = [c for c in cols_probe.feature_cols
                              if feat.loc[tr, c].notna().any()]
    best_gbm = (None, np.inf)
    for lr in (0.03, 0.05, 0.08):
        for leaves in (15, 31, 63):
            gbm = HistGradientBoostingClassifier(
                loss="log_loss", max_iter=800, learning_rate=lr,
                max_leaf_nodes=leaves, l2_regularization=1.0,
                min_samples_leaf=40, early_stopping=True,
                validation_fraction=0.12, random_state=42)
            gbm.fit(feat.loc[tr, cols_probe.active_cols], y[tr.to_numpy()])
            p = gbm.predict_proba(valid[cols_probe.active_cols])
            ll = log_loss_(p, y_va)
            tag = f"lr={lr} leaves={leaves}"
            print(f"  {tag:<24} LogLoss={ll:.5f} (iters={gbm.n_iter_})")
            if ll < best_gbm[1]:
                best_gbm = (tag, ll)
    print(f"  最优: {best_gbm[0]} ({best_gbm[1]:.5f})")
    print("\n如最优配置不同于当前默认值, 请更新 dixon_coles.py / model.py 默认参数")


if __name__ == "__main__":
    main()
