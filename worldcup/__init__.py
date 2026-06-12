"""世界杯比赛预测系统

核心组件:
- elo:         加权 Elo 评分体系 (赛事重要性 K 因子 / 净胜球倍增 / 主场优势)
- dixon_coles: 时间衰减 Dixon-Coles 双变量泊松模型 (比分概率矩阵)
- features:    防泄漏的滚动特征工程
- model:       梯度提升 + 对数池融合 + 温度校准
- markets:     比分矩阵 -> 胜平负 / 精确比分 / 大小球市场 (IPF 自洽调整)
- evaluate:    RPS / Brier / 对数损失 滚动前向回测
- simulate:    2026 世界杯蒙特卡洛整赛事模拟
"""

__version__ = "1.0.0"
