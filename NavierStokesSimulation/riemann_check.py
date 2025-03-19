import numpy as np

# リーマンゼータ関数の非自明なゼロ点の虚部
riemann_zeros = np.array([14.134725, 21.022040, 25.010858, 30.424876, 32.935062])

# エントロピーと流体速度の計算
entropy = 2.3235
c_fluid = entropy * 1.5

# 不等式の左辺（分子と分母の計算）
numerator = np.sum(1.0 / (riemann_zeros**2 + 0.25))
denominator = np.sum(np.log(riemann_zeros) / (riemann_zeros**2 + 0.25))
ratio = numerator / denominator

# 不等式の右辺（閾値）
threshold = 6.0 * np.pi / c_fluid

# 結果表示
print(f"比率: {ratio}")
print(f"閾値: {threshold}")
print(f"条件を満たすか: {ratio > threshold}")
