import numpy as np
import matplotlib.pyplot as plt
from scipy.special import zeta
from mpmath import mp

# 高精度計算のための設定
mp.dps = 50  # 桁数設定

# リーマンゼータ関数の非自明なゼロ点の虚部 (最初の20個)
riemann_zeros = np.array([
    14.134725141734693790,
    21.022039638771554993,
    25.010857580145688763,
    30.424876125859513210,
    32.935061587739189691,
    37.586178158825671257,
    40.918719012147495187,
    43.327073280914999519,
    48.005150881167159727,
    49.773832477672302182,
    52.970321477714460644,
    56.446247697063394307,
    59.347044002602353079,
    60.831778524609976569,
    65.112544048081730433,
    67.079810529494173714,
    69.546401711173979646,
    72.067157674481907582,
    75.704690699083933372,
    77.144840132021822813
])

def evaluate_condition(n_zeros, c_fluid):
    """
    条件式 Σ(1/(γ_n²+1/4)) / Σ(log(γ_n)/(γ_n²+1/4)) > 6π/c_fluid を評価
    
    引数:
        n_zeros: 計算に使用するリーマンゼロ点の数
        c_fluid: 流体定数
    
    戻り値:
        条件が満たされるかどうか、および関連する値
    """
    # 使用するゼロ点の数を制限
    zeros = riemann_zeros[:n_zeros]
    
    # 分子: Σ 1/(γ_n² + 1/4)
    numerator = np.sum(1.0 / (zeros**2 + 0.25))
    
    # 分母: Σ log(γ_n)/(γ_n² + 1/4)
    denominator = np.sum(np.log(zeros) / (zeros**2 + 0.25))
    
    # 比率
    ratio = numerator / denominator
    
    # 閾値
    threshold = 6 * np.pi / c_fluid
    
    # 条件が満たされるか
    is_satisfied = ratio > threshold
    
    return {
        'zeros_used': n_zeros,
        'c_fluid': c_fluid,
        'numerator': numerator,
        'denominator': denominator,
        'ratio': ratio,
        'threshold': threshold,
        'is_satisfied': is_satisfied
    }

def analyze_c_fluid_range(max_zeros=20):
    """
    様々な流体定数c_fluidに対して条件が満たされるか分析
    
    引数:
        max_zeros: 使用するリーマンゼロ点の最大数
    """
    c_fluid_values = np.linspace(1.0, 20.0, 100)
    results = []
    
    for c in c_fluid_values:
        result = evaluate_condition(max_zeros, c)
        results.append(result['is_satisfied'])
    
    # c_fluidのどの値から条件が満たされるかを見つける
    critical_index = np.argmax(results)
    critical_c_fluid = c_fluid_values[critical_index] if critical_index < len(c_fluid_values) else None
    
    return {
        'c_fluid_values': c_fluid_values,
        'satisfied': results,
        'critical_c_fluid': critical_c_fluid
    }

def analyze_zero_contributions(c_fluid=3.0):
    """
    各リーマンゼロ点の寄与を分析
    
    引数:
        c_fluid: 流体定数
    """
    contributions = []
    
    for i, zero in enumerate(riemann_zeros):
        # この1つのゼロ点による寄与
        single_contribution = 1.0 / (zero**2 + 0.25)
        log_contribution = np.log(zero) / (zero**2 + 0.25)
        
        contributions.append({
            'zero_index': i + 1,
            'zero_value': zero,
            'numerator_contribution': single_contribution,
            'denominator_contribution': log_contribution,
            'relative_importance': single_contribution / log_contribution
        })
    
    return contributions

def plot_results(results, c_fluid_analysis):
    """結果をプロットする"""
    # プロット1: c_fluidに対する比率と閾値
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    c_values = np.linspace(1.0, 20.0, 100)
    ratios = [evaluate_condition(results['zeros_used'], c)['ratio'] for c in c_values]
    thresholds = [6 * np.pi / c for c in c_values]
    
    plt.plot(c_values, ratios, 'b-', label='比率')
    plt.plot(c_values, thresholds, 'r--', label='閾値 6π/c_fluid')
    plt.axvline(x=results['c_fluid'], color='g', linestyle='-', label=f'c_fluid = {results["c_fluid"]}')
    plt.grid(True)
    plt.xlabel('流体定数 c_fluid')
    plt.ylabel('値')
    plt.title('流体定数に対する比率と閾値')
    plt.legend()
    
    # プロット2: 条件が満たされる領域
    plt.subplot(2, 2, 2)
    satisfied = [ratios[i] > thresholds[i] for i in range(len(c_values))]
    plt.plot(c_values, satisfied, 'g-')
    plt.axvline(x=c_fluid_analysis['critical_c_fluid'], color='r', linestyle='--', 
                label=f'臨界値 c ≈ {c_fluid_analysis["critical_c_fluid"]:.2f}')
    plt.grid(True)
    plt.xlabel('流体定数 c_fluid')
    plt.ylabel('条件満足 (1=Yes, 0=No)')
    plt.title('どの流体定数で条件が満たされるか')
    plt.legend()
    
    # プロット3: ゼロ点の数の影響
    plt.subplot(2, 2, 3)
    n_zeros_range = range(1, len(riemann_zeros) + 1)
    ratio_by_zeros = [evaluate_condition(n, results['c_fluid'])['ratio'] for n in n_zeros_range]
    threshold = results['threshold']
    
    plt.plot(n_zeros_range, ratio_by_zeros, 'b-', label='比率')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'閾値 = {threshold:.4f}')
    plt.grid(True)
    plt.xlabel('使用したリーマンゼロ点の数')
    plt.ylabel('比率')
    plt.title('ゼロ点の数に対する比率の変化')
    plt.legend()
    
    # プロット4: 各ゼロ点の寄与
    plt.subplot(2, 2, 4)
    contributions = analyze_zero_contributions(results['c_fluid'])
    zero_indices = [c['zero_index'] for c in contributions]
    relative_importance = [c['relative_importance'] for c in contributions]
    
    plt.bar(zero_indices, relative_importance)
    plt.grid(True)
    plt.xlabel('ゼロ点のインデックス')
    plt.ylabel('相対的寄与')
    plt.title('各リーマンゼロ点の相対的寄与')
    
    plt.tight_layout()
    plt.savefig('wormhole_condition_analysis.png')
    plt.show()

def print_detailed_report(results, contributions):
    """詳細な結果レポートを表示"""
    print("\n===== ワームホール存在条件の評価 =====")
    print(f"使用したリーマンゼロ点の数: {results['zeros_used']}")
    print(f"流体定数 c_fluid: {results['c_fluid']}")
    print("\n----- 計算結果 -----")
    print(f"分子: Σ 1/(γ_n² + 1/4) ≈ {results['numerator']:.8f}")
    print(f"分母: Σ log(γ_n)/(γ_n² + 1/4) ≈ {results['denominator']:.8f}")
    print(f"比率: {results['ratio']:.8f}")
    print(f"閾値 6π/c_fluid: {results['threshold']:.8f}")
    print(f"条件 {results['ratio']:.4f} > {results['threshold']:.4f} は {'満たされます' if results['is_satisfied'] else '満たされません'}")
    
    print("\n----- 重要なゼロ点の寄与 -----")
    # 寄与の大きい順にソート
    sorted_contributions = sorted(contributions, key=lambda x: x['relative_importance'], reverse=True)
    for i, contrib in enumerate(sorted_contributions[:5]):
        print(f"ゼロ点 γ_{contrib['zero_index']} ≈ {contrib['zero_value']:.6f}:")
        print(f"  相対的寄与: {contrib['relative_importance']:.6f}")
        print(f"  分子への寄与: {contrib['numerator_contribution']:.6f}")
        print(f"  分母への寄与: {contrib['denominator_contribution']:.6f}")
    
    print("\n----- 理論的解釈 -----")
    if results['is_satisfied']:
        print("条件が満たされるため、理論上は大域的滑らかな解が存在する可能性が高いです。")
        print("量子効果（リーマン予想に基づく補正）が古典的な特異点形成を抑制していると解釈できます。")
    else:
        print("条件が満たされないため、理論上は特異点が形成される可能性があります。")
        print("流体定数c_fluidを大きくするか、より多くのリーマンゼロ点を考慮することで条件が満たされる可能性があります。")

# メイン実行部分
if __name__ == "__main__":
    # デフォルトのパラメータ
    n_zeros = 20  # 使用するリーマンゼロ点の数
    c_fluid = 3.0  # 流体定数
    
    # コマンドライン引数があれば使用
    import sys
    if len(sys.argv) > 1:
        n_zeros = int(sys.argv[1])
    if len(sys.argv) > 2:
        c_fluid = float(sys.argv[2])
    
    # 条件の評価
    results = evaluate_condition(n_zeros, c_fluid)
    
    # 各ゼロ点の寄与分析
    contributions = analyze_zero_contributions(c_fluid)
    
    # c_fluidの範囲分析
    c_fluid_analysis = analyze_c_fluid_range(n_zeros)
    
    # 詳細レポート表示
    print_detailed_report(results, contributions)
    
    # 結果プロット
    plot_results(results, c_fluid_analysis)
    
    # c_fluidをどれだけ大きくすれば条件が満たされるか
    if not results['is_satisfied'] and c_fluid_analysis['critical_c_fluid'] is not None:
        print(f"\n流体定数を c_fluid ≈ {c_fluid_analysis['critical_c_fluid']:.2f} 以上にすると条件が満たされます。") 