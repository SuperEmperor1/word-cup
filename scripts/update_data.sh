#!/usr/bin/env bash
# 更新国际比赛结果数据集 (martj42/international_results, CC0)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
curl -fsSL -o data/results.csv \
  https://raw.githubusercontent.com/martj42/international_results/master/results.csv
echo "已更新: $(wc -l < data/results.csv) 行"
