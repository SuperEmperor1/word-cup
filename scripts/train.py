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
    print(f"   融合权重 w(DC)={ens.weight:.2f}, 温度 T={ens.temp:.2f}")
    print(f"   已保存 models/ensemble.pkl, 总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
