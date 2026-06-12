#!/usr/bin/env bash
# 更新全部数据源 (CC0/开放许可)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
base="https://raw.githubusercontent.com/martj42/international_results/master"
curl -fsSL -o data/results.csv "$base/results.csv"
curl -fsSL -o data/goalscorers.csv "$base/goalscorers.csv"
curl -fsSL -o data/shootouts.csv "$base/shootouts.csv"
curl -fsSL -o data/country_centroids.csv \
  "https://raw.githubusercontent.com/gavinr/world-countries-centroids/master/dist/countries.csv"
echo "results: $(wc -l < data/results.csv) 行, goalscorers: $(wc -l < data/goalscorers.csv) 行"
