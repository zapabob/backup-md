import numpy as np

# リーマンゼータ関数の非自明なゼロ点の虚部
riemann_zeros = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832
]

def evaluate_global_existence_condition(c_fluid=3.0):
    """
    リーマン予想に基づく大域解の存在性条件を評価する
    
    $$\frac{\sum_{n=1}^{\infty}\frac{1}{\gamma_n^2+1/4}}{\sum_{n=1}^{\infty}\frac{\log\gamma_n}{\gamma_n^2+1/4}} > \frac{6\pi}{c_{\text{fluid}}}$$
    """
    # リーマンゼータ関数の非自明ゼロ点の虚部を使用
    gamma_n = np.array(riemann_zeros)
    
    # 部分和の計算（有限個のゼロ点で近似）
    numerator = np.sum(1.0 / (gamma_n**2 + 0.25))
    denominator = np.sum(np.log(gamma_n) / (gamma_n**2 + 0.25))
    
    # 比率の計算
    ratio = numerator / denominator
    
    # 閾値の計算
    threshold = 6.0 * np.pi / c_fluid
    
    print(f"計算された比率: {ratio:.6f}")
    print(f"閾値 (c_fluid={c_fluid:.2f}): {threshold:.6f}")
    print(f"条件を満たす: {ratio > threshold}")
    
    # 個々の項の寄与を表示
    print("\n各ゼロ点の寄与:")
    for i, gamma in enumerate(gamma_n):
        contrib_num = 1.0 / (gamma**2 + 0.25)
        contrib_den = np.log(gamma) / (gamma**2 + 0.25)
        print(f"ゼロ点 γ_{i+1} = {gamma:.6f}: 分子への寄与 = {contrib_num:.6f}, 分母への寄与 = {contrib_den:.6f}")
    
    return ratio, threshold, ratio > threshold

# 論文におけるシミュレーション結果から得られた値
entropy_from_simulation = 2.3235
c_fluid_estimated = entropy_from_simulation * 1.5

print("===== リーマン予想に基づく大域解の存在性条件の評価 =====")
print("1. 標準的な流体力学パラメータを使用 (c_fluid = 3.0):")
evaluate_global_existence_condition(3.0)

print("\n2. シミュレーション結果から推定したパラメータを使用:")
evaluate_global_existence_condition(c_fluid_estimated)

print("\n3. 異なるパラメータ値での閾値の変化:")
for c in [2.0, 2.5, 3.0, 3.5, 4.0]:
    ratio, threshold, satisfied = evaluate_global_existence_condition(c)
    print(f"c_fluid = {c:.1f}: 閾値 = {threshold:.6f}, 条件を満たす: {satisfied}") 