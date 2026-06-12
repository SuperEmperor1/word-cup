# 外部商业数据接入目录

将符合以下 schema 的 CSV 放入本目录，系统自动发现并接入
（`worldcup/external.py::ExternalData.discover`）。缺任何文件都会优雅降级。

**法律提示**：FM2026 数据库为 Sports Interactive 专有资产，请用游戏内
编辑器自行导出（个人使用）；Transfermarkt 数据请遵守其使用条款；
赔率请使用己方有授权的数据商 API。本仓库不分发任何商业数据。

## fm_players.csv —— FM2026 球员导出（球员状态/伤停/能力）

| 列 | 说明 |
|---|---|
| date | 快照日期 YYYY-MM-DD（可多期快照，系统按比赛日 as-of 取最近一期，超 120 天视为过期）|
| nation | 国家队名（与 results.csv 的队名一致）|
| player | 球员名 |
| position | GK / DEF / MID / ATT |
| age | 年龄 |
| current_ability | FM 当前能力 CA（0–200）|
| condition | 体能状态 0–1（FM 中的 Condition%）|
| injured | 是否伤停 0/1 |

## market_values.csv —— 转会市场身价

`date,team,total_value_eur,top11_value_eur`（建议每季度一期快照，超 400 天过期）

## odds.csv —— 博彩赔率（欧赔小数）

`date,home_team,away_team,odds_h,odds_d,odds_a`
（按比赛精确匹配；系统用 Shin 方法去水后使用，额外列会被忽略）

**中国体彩一键抓取**：`python scripts/fetch_odds_sporttery.py`
（官方公开接口，含 48 队中文名映射、防重合并、历史快照保留；
执行环境网络受限时请在本地运行）。体彩固定奖金水位约 12-30%，
Shin 去水已处理。其他来源（含外围盘口）只要落成同 schema 即可接入——
请确保数据获取方式符合你所在地区的法律与数据源条款。

## 演示数据

`python scripts/make_demo_external.py` 会在 `data/external_demo/` 生成
**完全合成**的演示数据用于验证管线；验证时将其复制到本目录，
**切勿**把合成数据当真实数据用于实际预测。

## 生效机制

1. **特征通路**：快照 as-of 对齐为 21 个特征列（fm_*/mv_*/mkt_*），
   历史快照积累 ≥ 数百场后 GBM 自动学习其分裂价值；
2. **即时调整通路**：伤停负担与平均状态映射为 Elo 当量修正
   （每 10% 阵容 CA 伤停 ≈ −40 Elo，状态每 ±0.01 ≈ ±4 Elo，截断于
   [−120, +60]），只作用于未来比赛的预测行，无需重训即时生效；
3. **市场锚定**：预测行带赔率时，最终概率 = 0.7·模型 + 0.3·去水市场
   （`Ensemble.predict_rows(market_anchor=...)` 可调）。
