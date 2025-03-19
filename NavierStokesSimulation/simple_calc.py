import numpy as np

# リーマンゼータ関数の非自明なゼロ点の虚部
riemann_zeros = np.array([
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062
])

# シミュレーション結果から得られたエントロピー
entropy = 2.3235  
c_fluid = entropy * 1.5

# 分子と分母の計算
numerator = np.sum(1.0 / (riemann_zeros**2 + 0.25))
denominator = np.sum(np.log(riemann_zeros) / (riemann_zeros**2 + 0.25))

# 比率と閾値
ratio = numerator / denominator
threshold = 6.0 * np.pi / c_fluid

# 結果を表示
print("===== リーマン予想に基づく大域解の存在性条件の評価 =====")
print(f"エントロピー: {entropy}")
print(f"c_fluid: {c_fluid}")
print(f"分子 (合計 1/(gamma^2+1/4)): {numerator}")
print(f"分母 (合計 log(gamma)/(gamma^2+1/4)): {denominator}")
print(f"比率: {ratio}")
print(f"閾値 (6π/c_fluid): {threshold}")
print(f"条件を満たす: {ratio > threshold}")
print("=====================================================")

# 各ゼロ点の寄与を表示
print("\n各ゼロ点の寄与:")
for i, gamma in enumerate(riemann_zeros):
    contrib_num = 1.0 / (gamma**2 + 0.25)
    contrib_den = np.log(gamma) / (gamma**2 + 0.25)
    print(f"ゼロ点 γ_{i+1} = {gamma:.6f}: 分子への寄与 = {contrib_num:.8f}, 分母への寄与 = {contrib_den:.8f}") 