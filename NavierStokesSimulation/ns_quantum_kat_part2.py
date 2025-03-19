import numpy as np
import matplotlib.pyplot as plt

# リーマンゼータ関数の非自明なゼロ点の虚部
riemann_zeros = np.array([
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832
])

def evaluate_global_existence_condition(c_fluid=3.0):
    """
    リーマン予想に基づく大域解の存在性条件を評価する
    
    $$\frac{\sum_{n=1}^{\infty}\frac{1}{\gamma_n^2+1/4}}{\sum_{n=1}^{\infty}\frac{\log\gamma_n}{\gamma_n^2+1/4}} > \frac{6\pi}{c_{\text{fluid}}}$$
    
    引数:
        c_fluid: 流体力学的パラメータ (デフォルト: 3.0)
    
    戻り値:
        condition_satisfied: 条件を満たす場合はTrue
        ratio: 左辺の計算値
        threshold: 右辺の閾値
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
    
    # 条件の評価
    condition_satisfied = ratio > threshold
    
    return condition_satisfied, ratio, threshold

def analyze_riemann_zeros():
    """リーマンゼータ関数のゼロ点を分析し、結果を表示する"""
    print("\n===== リーマン予想に基づく大域解の存在性条件の評価 =====")
    
    # シミュレーション結果から得られたエントロピー
    entropy = 2.3235  
    c_fluid = entropy * 1.5
    
    print(f"エントロピー: {entropy}")
    print(f"c_fluid: {c_fluid}")
    
    # 条件の評価
    condition_satisfied, ratio, threshold = evaluate_global_existence_condition(c_fluid)
    
    print(f"比率: {ratio}")
    print(f"閾値 (6π/c_fluid): {threshold}")
    print(f"条件を満たす: {condition_satisfied}")
    
    if condition_satisfied:
        print("大域的な滑らかな解の存在が予測されます")
    else:
        print("大域的な滑らかな解の存在は保証されません")
    
    print("=====================================================")
    
    # 各ゼロ点の寄与を表示
    print("\n各ゼロ点の寄与:")
    for i, gamma in enumerate(riemann_zeros):
        contrib_num = 1.0 / (gamma**2 + 0.25)
        contrib_den = np.log(gamma) / (gamma**2 + 0.25)
        print(f"ゼロ点 γ_{i+1} = {gamma:.6f}: 分子への寄与 = {contrib_num:.8f}, 分母への寄与 = {contrib_den:.8f}")

def plot_riemann_condition():
    """リーマン予想の条件をc_fluidの関数としてプロット"""
    c_fluid_values = np.linspace(1.0, 5.0, 100)
    thresholds = 6.0 * np.pi / c_fluid_values
    
    # リーマンゼータ関数のゼロ点による比率（c_fluidに依存しない定数）
    gamma_n = np.array(riemann_zeros)
    numerator = np.sum(1.0 / (gamma_n**2 + 0.25))
    denominator = np.sum(np.log(gamma_n) / (gamma_n**2 + 0.25))
    ratio = numerator / denominator
    
    # プロット作成
    plt.figure(figsize=(10, 6))
    plt.plot(c_fluid_values, thresholds, 'r-', label='閾値 (6π/c_fluid)')
    plt.axhline(y=ratio, color='b', linestyle='-', label=f'計算された比率: {ratio:.4f}')
    
    # シミュレーションから得られたc_fluid値を強調表示
    entropy = 2.3235
    c_fluid_sim = entropy * 1.5
    threshold_sim = 6.0 * np.pi / c_fluid_sim
    plt.plot(c_fluid_sim, threshold_sim, 'go', markersize=10, label=f'シミュレーションの値 (c_fluid={c_fluid_sim:.4f})')
    
    # グラフの装飾
    plt.xlabel('c_fluid')
    plt.ylabel('値')
    plt.title('リーマン予想に基づく大域解の存在性条件')
    plt.grid(True)
    plt.legend()
    plt.axvline(x=c_fluid_sim, color='g', linestyle='--')
    
    # 条件を満たす領域を塗りつぶし
    idx = np.where(thresholds < ratio)[0]
    if len(idx) > 0:
        plt.fill_between(c_fluid_values[idx], 0, thresholds[idx], alpha=0.3, color='g', label='条件を満たす領域')
    
    plt.savefig("riemann_condition_plot.png")
    plt.show()

if __name__ == "__main__":
    # リーマン予想の検証を実行
    analyze_riemann_zeros()
    
    # 条件のグラフをプロット
    plot_riemann_condition()
