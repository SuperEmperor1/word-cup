"""训练线上模型并运行滚动前向回测。

用法: python scripts/train.py [--no-backtest]
产出: models/ensemble.pkl + 回测指标报告
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from worldcup.evaluate import walk_forward
from worldcup.pipeline import prepare, train_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-backtest", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print("== 加载数据 / 计算 Elo / 构建特征 ==")
    feat, ratings = prepare()
    print(f"   比赛总数 {len(feat)}, 已完赛 {feat['played'].sum()}, "
          f"耗时 {time.time()-t0:.1f}s")

    if not args.no_backtest:
        print("== 滚动前向回测 (训练<2023, 验证 2023-24, 测试 2025+) ==")
        report = walk_forward(feat)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        os.makedirs("models", exist_ok=True)
        with open("models/backtest_report.json", "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    print("== 训练线上模型 (全量数据) ==")
    ens = train_full(feat)
    os.makedirs("models", exist_ok=True)
    ens.save("models/ensemble.pkl")
    # 版本元数据: 预测可追溯到 数据截止/代码版本/特征数
    import subprocess
    done = feat[feat["played"]]
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        rev = "unknown"
    meta = {
        "version": f"{done['date'].max().date()}_{rev}",
        "git_rev": rev,
        "data_end": str(done["date"].max().date()),
        "n_matches": int(len(done)),
        "n_features": len(ens.active_cols),
        "xg_weight": ens.xg_weight, "goal_scale": ens.goal_scale,
        "temperature": ens.temp,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open("models/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"   xG权重={ens.xg_weight:.2f} 尺度={ens.goal_scale:.2f} "
          f"温度={ens.temp:.2f}")
    print(f"   已保存 models/ensemble.pkl (版本 {meta['version']}), "
          f"总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
