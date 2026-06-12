#!/usr/bin/env bash
# 每日更新: 拉数据 -> 重训 -> 存档预测 -> 线上对账
# 可由 cron 或 GitHub Actions 定时触发
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/update_data.sh
python3 scripts/train.py --no-backtest
python3 scripts/archive_predictions.py
python3 scripts/score_archive.py
python3 scripts/build_site.py
