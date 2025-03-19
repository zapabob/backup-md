import numpy as np

# リーマンゼータ関数の非自明なゼロ点の虚部
riemann_zeros = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832
]

# 結果を書き込むファイル
output_file = open("riemann_analysis_results.txt", "w", encoding="utf-8")

def print_to_file(text):
    """ファイルと標準出力の両方に出力"""
    print(text)
    output_file.write(text + "\n")

def evaluate_condition(c_fluid=3.0):
    """
    リーマン予想に基づく大域解の存在性条件を評価
    
    $$\frac{\sum_{n=1}^{\infty}\frac{1}{\gamma_n^2+1/4}}{\sum_{n=1}^{\infty}\frac{\log\gamma_n}{\gamma_n^2+1/4}} > \frac{6\pi}{c_{\text{fluid}}}$$
    """
    gamma_n = np.array(riemann_zeros)
    
    # 分子と分母の計算
    numerator = np.sum(1.0 / (gamma_n**2 + 0.25))
    denominator = np.sum(np.log(gamma_n) / (gamma_n**2 + 0.25))
    
    # 比率と閾値
    ratio = numerator / denominator
    threshold = 6.0 * np.pi / c_fluid
    
    print_to_file(f"計算された比率: {ratio:.6f}")
    print_to_file(f"閾値 (c_fluid={c_fluid:.2f}): {threshold:.6f}")
    print_to_file(f"条件を満たす: {ratio > threshold}")
    
    return ratio, threshold, ratio > threshold

# シミュレーション結果と理論の比較
print_to_file("======= リーマン予想に基づく大域解の存在性条件の評価 =======")

# エントロピー値から推定したパラメータ
entropy = 2.3235
c_fluid = entropy * 1.5

print_to_file(f"\nシミュレーションから得られたエントロピー: {entropy:.4f}")
print_to_file(f"推定されたc_fluid: {c_fluid:.4f}")

# 条件の評価
ratio, threshold, satisfied = evaluate_condition(c_fluid)

print_to_file("\n不等式の詳細評価:")
print_to_file(f"左辺: {ratio:.6f}")
print_to_file(f"右辺: {threshold:.6f}")
print_to_file(f"結論: {'条件を満たし、大域的滑らかな解の存在が予測される' if satisfied else '条件を満たさない'}")

# 異なるパラメータでの評価
print_to_file("\n異なるc_fluidでの評価:")
for c in [2.0, 2.5, 3.0, 3.5, 4.0]:
    ratio, threshold, satisfied = evaluate_condition(c)
    print_to_file(f"c_fluid = {c:.1f}: 条件を満たす: {satisfied}, 閾値 = {threshold:.6f}")

# 数値計算による有限和の収束性
print_to_file("\n有限個のゼロ点を使用した総和の収束性:")
for n in [1, 2, 5, 10]:
    gamma_partial = np.array(riemann_zeros[:n])
    num_partial = np.sum(1.0 / (gamma_partial**2 + 0.25))
    den_partial = np.sum(np.log(gamma_partial) / (gamma_partial**2 + 0.25))
    ratio_partial = num_partial / den_partial
    print_to_file(f"最初の{n}個のゼロ点: 比率 = {ratio_partial:.6f}")

output_file.close()
print("解析が完了しました。結果は riemann_analysis_results.txt に保存されています。") 