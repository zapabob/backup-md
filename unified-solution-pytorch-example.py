import numpy as np
import matplotlib.pyplot as plt
import torch
from unified_solution_pytorch import AdvancedPyTorchUnifiedSolutionCalculator
import time

def plot_complex_function(x, y, z, title):
    """複素関数のプロット"""
    plt.figure(figsize=(10, 8))
    plt.subplot(121)
    plt.scatter(x, y, c=np.abs(z), cmap='viridis')
    plt.colorbar(label='|Ψ|')
    plt.title(f'{title} - 振幅')
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.subplot(122)
    plt.scatter(x, y, c=np.angle(z), cmap='hsv')
    plt.colorbar(label='arg(Ψ)')
    plt.title(f'{title} - 位相')
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.tight_layout()

def main():
    print("統合特解の数理的精緻化に基づくPyTorch+CUDA計算を開始します...")
    
    # CUDAが利用可能かチェック
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print(f"CUDA対応GPUが見つかりました: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("GPUが見つかりません。CPUで実行します。")
    
    # 高次元統合特解の計算
    n_dims = 3  # 3次元
    q_max = 6   # qの最大値
    max_k = 50  # 展開の最大項数
    L_max = 15  # チェビシェフ多項式の最大次数
    
    print(f"パラメータ: n_dims={n_dims}, q_max={q_max}, max_k={max_k}, L_max={L_max}, use_cuda={use_cuda}")
    
    # 高度な統合特解計算器のインスタンス化
    calculator = AdvancedPyTorchUnifiedSolutionCalculator(
        n_dims=n_dims, q_max=q_max, max_k=max_k, L_max=L_max, use_cuda=use_cuda
    )
    
    # テスト計算用の点を生成
    n_samples = 1000
    print(f"テスト計算用に{n_samples}点をランダムサンプリング...")
    points = np.random.rand(n_samples, n_dims)
    
    # 統合特解の計算
    print("統合特解を計算中...")
    start_time = time.time()
    psi_values, metrics = calculator.compute_unified_solution_advanced(points)
    end_time = time.time()
    print(f"計算時間: {end_time - start_time:.3f}秒")
    
    # 計量情報の表示
    print("\n統合特解の計量情報:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # トポロジカル不変量の計算
    print("\nトポロジカル不変量を計算中...")
    invariants = calculator.compute_topological_invariants()
    print("チャーン・サイモンズ不変量:", invariants['chern_simons'])
    print("ジョーンズ多項式の係数（最初の5項）:", invariants['jones_polynomial'][:5])
    
    # 情報幾何学的計量の計算
    print("\n情報幾何学的計量を計算中...")
    info_metrics = calculator.compute_information_geometry_metrics(points[:100])  # 100点で計算
    print("スカラー曲率:", info_metrics['scalar_curvature'])
    print("フィッシャー情報行列の最大固有値:", np.max(np.linalg.eigvals(info_metrics['fisher_matrix'])))
    
    # 量子計算複雑性の評価
    print("\n量子計算複雑性を評価中...")
    complexity = calculator.compute_quantum_complexity()
    print("量子回路の深さ:", complexity['circuit_depth'])
    print("量子ゲート数:", complexity['gate_count'])
    print("テンソルネットワークのボンド次元:", complexity['bond_dimension'])
    
    # 量子動力学のシミュレーション
    print("\n量子動力学をシミュレーション中...")
    time_steps = 5
    dt = 0.1
    test_points = np.random.rand(10, n_dims)  # 10点で計算
    psi_history, metric_history = calculator.simulate_quantum_dynamics(
        time_steps=time_steps, dt=dt, initial_points=test_points
    )
    
    print(f"{time_steps}ステップの時間発展を計算完了")
    for t in range(time_steps):
        energy = metric_history[t]['energy']
        print(f"t={t*dt:.1f}: エネルギー = {energy:.6f}")
    
    # 2次元断面でのプロット（3次元以上の場合は最初の2次元をプロットする）
    if n_dims >= 2 and len(points) >= 100:
        print("\n統合特解の2次元断面をプロット中...")
        plot_points = points[:100]  # 100点のみプロット
        plot_values = psi_values[:100]
        
        plot_complex_function(
            plot_points[:, 0], plot_points[:, 1], plot_values, 
            "統合特解 Ψ(x) の2次元断面 (PyTorch+CUDA計算)"
        )
        plt.savefig('unified_solution_pytorch_plot.png')
        print("プロットをunified_solution_pytorch_plot.pngに保存しました")
    
    # CUDA性能ベンチマーク（CUDAが利用可能な場合）
    if use_cuda:
        print("\nCUDA性能ベンチマークを実行中...")
        batch_sizes = [100, 500, 1000, 5000]
        times = []
        
        for batch_size in batch_sizes:
            test_points = np.random.rand(batch_size, n_dims)
            start_time = time.time()
            calculator.compute_unified_solution_advanced(test_points)
            end_time = time.time()
            times.append(end_time - start_time)
            print(f"バッチサイズ {batch_size}: {times[-1]:.3f}秒")
        
        # 結果のプロット
        plt.figure(figsize=(10, 6))
        plt.plot(batch_sizes, times, 'o-', linewidth=2)
        plt.xlabel('バッチサイズ')
        plt.ylabel('計算時間 (秒)')
        plt.title('PyTorch + CUDA 性能ベンチマーク')
        plt.grid(True)
        plt.savefig('pytorch_cuda_benchmark.png')
        print("ベンチマーク結果をpytorch_cuda_benchmark.pngに保存しました")
    
    print("\n計算完了!")

if __name__ == "__main__":
    main() 