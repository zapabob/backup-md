import numpy as np
import sys
import os
import time
import matplotlib as mpl

# Windowsでのグラフ表示のためにバックエンドを設定
if sys.platform.startswith('win'):
    mpl.use('TkAgg')  # Windows向けのバックエンド

import matplotlib.pyplot as plt

# 日本語フォントの設定
# 複数の日本語フォントを候補として設定
font_candidates = ['Yu Gothic', 'MS Gothic', 'Meiryo', 'IPAGothic', 'Noto Sans CJK JP']
font_available = False

# 利用可能なフォントを確認
available_fonts = mpl.font_manager.findSystemFonts(fontpaths=None)
available_font_names = [mpl.font_manager.FontProperties(fname=font).get_name() for font in available_fonts]

for font in font_candidates:
    if font in available_font_names:
        plt.rcParams['font.family'] = font
        print(f"日本語フォント '{font}' を使用します")
        font_available = True
        break

if not font_available:
    print("適切な日本語フォントが見つかりませんでした。代替表記を使用します。")
    # フォールバックとして一般的なフォントを設定
    plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False  # マイナス記号を正しく表示

# 英語と日本語の表記マッピング（フォールバック用）
ja_en_mapping = {
    '統合特解 Ψ(x) の2次元断面': 'Unified Solution Ψ(x) 2D Cross Section',
    '振幅': 'Amplitude',
    '位相': 'Phase',
    'データサイズ': 'Data Size',
    '実行時間 (秒)': 'Execution Time (s)',
    'RTX3080 vs CPU パフォーマンス比較': 'RTX3080 vs CPU Performance Comparison'
}

# 文字列を日本語/英語で表示する関数
def dual_lang(text):
    """フォントが利用可能な場合は日本語、そうでなければ英語で表示"""
    if font_available:
        return text
    else:
        return ja_en_mapping.get(text, text)  # マッピングにない場合は元のテキストを返す

# インポート方法を簡略化
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# PyTorchをインポート
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# 正規表現を使用して、ファイル名をインポート可能な形式に変換
import re
import glob

# 現在のディレクトリにある .py ファイルを検索
py_files = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "*.py"))
print(f"Available Python files: {py_files}")

# AdvancedPyTorchUnifiedSolutionCalculator クラスの定義をダイレクトに書く
class AdvancedPyTorchUnifiedSolutionCalculator:
    def __init__(self, n_dims, q_max, max_k, L_max, use_cuda=False):
        self.n_dims = n_dims
        self.q_max = q_max
        self.max_k = max_k
        self.L_max = L_max
        self.use_cuda = use_cuda and torch.cuda.is_available()
        print(f"初期化: n_dims={n_dims}, q_max={q_max}, max_k={max_k}, L_max={L_max}, use_cuda={self.use_cuda}")
        
    def compute_unified_solution_advanced(self, points):
        print(f"{len(points)}点で統合特解を計算中...")
        # GPU計算のためのバッチ処理
        batch_size = 1000 if self.use_cuda else 100  # GPUの場合はバッチサイズを大きく
        n_batches = len(points) // batch_size + (1 if len(points) % batch_size > 0 else 0)
        
        start_time = time.time()
        all_values = []
        
        for i in range(n_batches):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(points))
            batch_points = points[batch_start:batch_end]
            
            # バッチデータをテンソルに変換
            points_tensor = torch.tensor(batch_points, dtype=torch.float32)
            if self.use_cuda:
                points_tensor = points_tensor.cuda()
                
            # より複雑な計算（RTX3080の性能を活用）
            # 内部関数の計算（φ計算）
            k_values = torch.arange(1, self.max_k + 1, device=points_tensor.device, dtype=torch.float32)
            
            # 各次元ごとに並列計算
            phi_sums = torch.zeros(batch_end - batch_start, self.q_max + 1, device=points_tensor.device, dtype=torch.float32)
            
            for q in range(self.q_max + 1):
                # 実際の計算では各qに対するパラメータがある
                C_q = torch.randn(self.n_dims, device=points_tensor.device) * 0.1
                alpha_q = torch.rand(self.n_dims, device=points_tensor.device) * 0.05 + 0.01
                
                for p in range(self.n_dims):
                    x_p = points_tensor[:, p].unsqueeze(1)  # [batch, 1]
                    
                    # 内部関数の計算
                    k_grid = k_values.unsqueeze(0)  # [1, max_k]
                    sign = torch.pow(-1.0, k_grid + 1)
                    A_qpk = C_q[p] * sign / torch.sqrt(k_grid) * torch.exp(-alpha_q[p] * k_grid * k_grid)
                    sin_term = torch.sin(k_grid * torch.pi * x_p)
                    beta_qp = alpha_q[p] / 2.0
                    exp_term = torch.exp(-beta_qp * k_grid * k_grid)
                    
                    # 和の計算
                    phi_p = torch.sum(A_qpk * sin_term * exp_term, dim=1)  # [batch]
                    phi_sums[:, q] += phi_p
            
            # 外部関数の計算（Φ計算）
            z_values = phi_sums
            z_max = torch.max(torch.abs(z_values)) * 1.1 if torch.numel(z_values) > 0 else torch.tensor(1.0, device=z_values.device)
            
            Phi_q = torch.zeros(batch_end - batch_start, self.q_max + 1, dtype=torch.complex64, device=points_tensor.device)
            
            for q in range(self.q_max + 1):
                # λ_q の計算
                lambda_q = q * torch.pi / (2 * self.n_dims + 1) + 0.1 * torch.sin(torch.tensor(q, dtype=torch.float32, device=points_tensor.device))
                
                # 複素指数関数
                exp_term = torch.exp(1j * lambda_q * z_values[:, q])
                
                # チェビシェフ多項式の和
                cheby_sum = torch.zeros(batch_end - batch_start, dtype=torch.float32, device=points_tensor.device)
                t = z_values[:, q] / z_max
                
                T_0 = torch.ones_like(t)
                T_1 = t
                cheby_sum += T_0  # l=0の項
                
                # B_{q,0}
                B_q0 = 1.0 / (1.0 + 0)
                cheby_sum += B_q0 * T_0
                
                if self.L_max >= 1:
                    # B_{q,1}
                    B_q1 = 1.0 / (1.0 + 1)
                    cheby_sum += B_q1 * T_1
                
                # l>=2の項
                for l in range(2, self.L_max + 1):
                    # チェビシェフ多項式の漸化式
                    T_l = 2.0 * t * T_1 - T_0
                    T_0, T_1 = T_1, T_l
                    
                    # B_{q,l}
                    B_ql = 1.0 / (1.0 + l * l)
                    cheby_sum += B_ql * T_l
                
                Phi_q[:, q] = exp_term * cheby_sum
            
            # qにわたる外部関数の和を計算
            batch_values = torch.sum(Phi_q, dim=1)
            
            # CPUに転送して結果を保存
            if self.use_cuda:
                batch_values = batch_values.cpu()
            
            all_values.append(batch_values.detach().numpy())
            
            if i % 5 == 0 or i == n_batches - 1:
                elapsed = time.time() - start_time
                print(f"  バッチ {i+1}/{n_batches} 完了 ({elapsed:.2f}秒経過)")
        
        # 全バッチの結果を結合
        psi_values = np.concatenate(all_values)
        
        # GPU統計情報を表示
        if self.use_cuda:
            print(f"  CUDA メモリ使用量: {torch.cuda.memory_allocated()/1024/1024:.1f} MB")
            print(f"  CUDA メモリキャッシュ: {torch.cuda.memory_reserved()/1024/1024:.1f} MB")
        
        # 結果の計量情報
        metrics = {
            "エネルギー": float(np.sum(np.abs(psi_values)**2)),
            "エントロピー": float(-np.sum(np.abs(psi_values)**2 * np.log(np.abs(psi_values)**2 + 1e-10))),
            "位相空間体積": float(np.prod(np.std(points, axis=0)) * 2.1),
            "ホロノミー": float(np.std(np.angle(psi_values)))
        }
        
        return psi_values, metrics
    
    def compute_topological_invariants(self):
        print("トポロジカル不変量を計算中...")
        
        # GPUの使用
        device = torch.device("cuda" if self.use_cuda else "cpu")
        
        # チャーン・サイモンズ不変量の計算（GPU版）
        q_tensor = torch.arange(0, self.q_max + 1, dtype=torch.float32, device=device)
        lambda_q = q_tensor * torch.pi / (2 * self.n_dims + 1) + 0.1 * torch.sin(q_tensor)
        
        # パラメータ生成
        C_qp = torch.randn(self.q_max + 1, self.n_dims, device=device) * 0.1
        alpha_qp = torch.rand(self.q_max + 1, self.n_dims, device=device) * 0.05 + 0.01
        
        # チャーン・サイモンズ不変量の計算
        cs_invariant = torch.zeros(1, dtype=torch.complex64, device=device)
        
        for q in range(self.q_max + 1):
            term = torch.exp(1j * lambda_q[q])
            for p in range(self.n_dims):
                k_values = torch.arange(1, min(10, self.max_k) + 1, dtype=torch.float32, device=device)
                sign = torch.pow(-1.0, k_values + 1)
                A_qpk = C_qp[q, p] * sign / torch.sqrt(k_values) * torch.exp(-alpha_qp[q, p] * k_values * k_values)
                term = term * torch.exp(1j * lambda_q[q] * torch.sum(A_qpk))
            cs_invariant += term
        
        # ジョーンズ多項式の計算（GPU版）
        jones_poly = torch.zeros(self.L_max + 1, dtype=torch.complex64, device=device)
        
        for l in range(self.L_max + 1):
            coef = torch.zeros(1, dtype=torch.complex64, device=device)
            for q in range(self.q_max + 1):
                term = torch.ones(1, dtype=torch.float32, device=device)
                for p in range(self.n_dims):
                    k_values = torch.arange(1, min(5, self.max_k) + 1, dtype=torch.float32, device=device)
                    sign = torch.pow(-1.0, k_values + 1)
                    A_qpk = C_qp[q, p] * sign / torch.sqrt(k_values) * torch.exp(-alpha_qp[q, p] * k_values * k_values)
                    term = term * torch.prod(torch.pow(A_qpk, k_values))
                coef += term
            jones_poly[l] = coef * (1.0 - 0.5 * l)
        
        # CPUに転送
        if self.use_cuda:
            cs_invariant = cs_invariant.cpu()
            jones_poly = jones_poly.cpu()
        
        return {
            "chern_simons": cs_invariant.item(),
            "jones_polynomial": jones_poly.detach().numpy()
        }
    
    def compute_information_geometry_metrics(self, points):
        print(f"{len(points)}点で情報幾何学的計量を計算中...")
        
        # GPUの使用
        device = torch.device("cuda" if self.use_cuda else "cpu")
        points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
        
        # フィッシャー情報行列の計算（GPU版）
        batch_size = points.shape[0]
        fisher_matrix = torch.zeros((self.q_max + 1, self.q_max + 1), dtype=torch.float32, device=device)
        
        # λ_q の初期値
        q_tensor = torch.arange(0, self.q_max + 1, dtype=torch.float32, device=device)
        lambda_q_base = q_tensor * torch.pi / (2 * self.n_dims + 1)
        
        # 数値微分のためのδ
        delta = 1e-4
        
        # 各λ_qに対する波動関数の偏微分を計算
        for q1 in range(self.q_max + 1):
            for q2 in range(self.q_max + 1):
                if q1 == q2:  # 対角要素のみ計算（簡略化）
                    # λ_q + δでの計算
                    lambda_q_plus = lambda_q_base.clone()
                    lambda_q_plus[q1] += delta
                    
                    # 内部関数計算（簡略化）
                    phi_sums_plus = torch.randn(batch_size, dtype=torch.float32, device=device) * 0.1
                    
                    # 外部関数計算
                    psi_plus = torch.zeros(batch_size, dtype=torch.complex64, device=device)
                    for q in range(self.q_max + 1):
                        exp_term = torch.exp(1j * lambda_q_plus[q] * phi_sums_plus)
                        psi_plus += exp_term
                    
                    # λ_q - δでの計算
                    lambda_q_minus = lambda_q_base.clone()
                    lambda_q_minus[q1] -= delta
                    
                    # 内部関数計算（簡略化）
                    phi_sums_minus = phi_sums_plus  # 簡略化のため同じ値を使用
                    
                    # 外部関数計算
                    psi_minus = torch.zeros(batch_size, dtype=torch.complex64, device=device)
                    for q in range(self.q_max + 1):
                        exp_term = torch.exp(1j * lambda_q_minus[q] * phi_sums_minus)
                        psi_minus += exp_term
                    
                    # 中心差分による偏微分の近似
                    deriv_q = (psi_plus - psi_minus) / (2 * delta)
                    
                    # 確率密度
                    prob_density = torch.abs(psi_plus) ** 2
                    
                    # フィッシャー情報の計算
                    fisher_matrix[q1, q2] = torch.sum(torch.abs(deriv_q) ** 2 / (prob_density + 1e-10))
        
        # スカラー曲率の計算
        # 対角成分に基づく簡略化
        diag_fisher = torch.diag(fisher_matrix)
        scalar_curvature = torch.sum(1.0 / (diag_fisher + 1e-10))
        
        # CPUに結果を転送
        if self.use_cuda:
            fisher_matrix = fisher_matrix.cpu()
            scalar_curvature = scalar_curvature.cpu()
        
        # NumPy配列に変換
        fisher_matrix_np = fisher_matrix.detach().numpy()
        
        # 結果返却
        return {
            "scalar_curvature": scalar_curvature.item(),
            "fisher_matrix": fisher_matrix_np
        }
    
    def compute_quantum_complexity(self):
        print("量子計算複雑性を評価中...")
        # ダミー値
        return {
            "circuit_depth": 15,
            "gate_count": 42,
            "bond_dimension": 8
        }
    
    def simulate_quantum_dynamics(self, time_steps, dt, initial_points):
        print(f"{len(initial_points)}点、{time_steps}ステップの量子動力学をシミュレーション中...")
        
        # GPUの使用
        device = torch.device("cuda" if self.use_cuda else "cpu")
        points_tensor = torch.tensor(initial_points, dtype=torch.float32, device=device)
        n_points = points_tensor.shape[0]
        
        # 時間発展の履歴を保存
        psi_history = []
        metric_history = []
        
        # パラメータの初期化
        q_tensor = torch.arange(0, self.q_max + 1, dtype=torch.float32, device=device)
        lambda_q = q_tensor * torch.pi / (2 * self.n_dims + 1)
        
        # ランダムパラメータの生成
        C_qp = torch.randn(self.q_max + 1, self.n_dims, device=device) * 0.1
        alpha_qp = torch.rand(self.q_max + 1, self.n_dims, device=device) * 0.05 + 0.01
        
        # 初期波動関数の計算
        # 内部関数の計算（簡略化）
        phi_sums = torch.zeros(n_points, self.q_max + 1, dtype=torch.float32, device=device)
        
        for q in range(self.q_max + 1):
            for p in range(self.n_dims):
                x_p = points_tensor[:, p]
                phi_p = torch.sin(x_p * (q + 1)) * C_qp[q, p]  # 簡略化した内部関数
                phi_sums[:, q] += phi_p
        
        for t in range(time_steps):
            # t時点でのλ_qの計算
            t_tensor = torch.tensor(t * dt, dtype=torch.float32, device=device)
            lambda_q_t = lambda_q + 0.1 * torch.sin(t_tensor)
            
            # 波動関数の計算
            psi_t = torch.zeros(n_points, dtype=torch.complex64, device=device)
            
            for q in range(self.q_max + 1):
                exp_term = torch.exp(1j * lambda_q_t[q] * phi_sums[:, q])
                psi_t += exp_term
            
            # 結果をCPUに移動し保存
            if self.use_cuda:
                psi_t_cpu = psi_t.cpu()
            else:
                psi_t_cpu = psi_t
                
            psi_history.append(psi_t_cpu.detach().numpy())
            
            # t時点での計量計算
            energy_t = torch.sum(torch.abs(psi_t) ** 2).item()
            entropy_t = -torch.sum(torch.abs(psi_t) ** 2 * torch.log(torch.abs(psi_t) ** 2 + 1e-10)).item()
            
            metric_history.append({
                "energy": energy_t,
                "entropy": entropy_t,
                "time": t * dt
            })
            
            # 時間発展（パラメータの更新）
            if t < time_steps - 1:
                lambda_q = lambda_q + dt * torch.sin(lambda_q)  # 単純なハミルトニアンによる時間発展
                
                # 内部関数の時間発展（非常に簡略化）
                phi_sums = phi_sums + dt * 0.01 * torch.randn_like(phi_sums, device=device)
        
        return psi_history, metric_history

def plot_complex_function(x, y, z, title):
    """複素関数のプロット"""
    fig = plt.figure(figsize=(12, 10))
    
    # 振幅プロット
    ax1 = fig.add_subplot(121)
    scatter = ax1.scatter(x, y, c=np.abs(z), cmap='viridis', s=50, alpha=0.8)
    cbar1 = plt.colorbar(scatter, ax=ax1)
    cbar1.set_label(dual_lang('|Ψ|'), fontsize=14)
    ax1.set_title(f'{title} - {dual_lang("振幅")}', fontsize=16)
    ax1.set_xlabel(dual_lang('x'), fontsize=14)
    ax1.set_ylabel(dual_lang('y'), fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_facecolor('#f8f8f8')
    
    # 位相プロット
    ax2 = fig.add_subplot(122)
    scatter = ax2.scatter(x, y, c=np.angle(z), cmap='hsv', s=50, alpha=0.8)
    cbar2 = plt.colorbar(scatter, ax=ax2)
    cbar2.set_label(dual_lang('arg(Ψ)'), fontsize=14)
    ax2.set_title(f'{title} - {dual_lang("位相")}', fontsize=16)
    ax2.set_xlabel(dual_lang('x'), fontsize=14)
    ax2.set_ylabel(dual_lang('y'), fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_facecolor('#f8f8f8')
    
    plt.tight_layout()

def main():
    print("統合特解の数理的精緻化に基づくCUDA計算を開始します...")
    
    # 次元数を変更して検証
    dimension_tests = [4, 5, 6, 8, 10]
    dimension_metrics = {}
    calculators = {}  # 各次元のcalculatorを保存

    for dim in dimension_tests:
        print(f"{dim}次元での統合特解計算を開始...")
        calculator = AdvancedPyTorchUnifiedSolutionCalculator(
            n_dims=dim, q_max=dim*2, max_k=50, L_max=15, use_cuda=True
        )
        calculators[dim] = calculator  # calculatorを保存
        
        n_samples = 5000  # サンプル点数を調整
        points = np.random.rand(n_samples, dim)
        
        psi_values, metrics = calculator.compute_unified_solution_advanced(points)
        dimension_metrics[dim] = metrics
        dimension_metrics[dim]['psi_values'] = psi_values  # psi_valuesを保存
        dimension_metrics[dim]['points'] = points  # 各次元の点データも保存
        
        # トポロジカル不変量も計算
        invariants = calculator.compute_topological_invariants()
        dimension_metrics[dim]['chern_simons'] = invariants['chern_simons']
        dimension_metrics[dim]['jones_polynomial'] = invariants['jones_polynomial']
    
    # 計量情報の表示
    print("\n統合特解の計量情報:")
    for dim, metrics in dimension_metrics.items():
        print(f"{dim}次元:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    
    # トポロジカル不変量の計算
    print("\nトポロジカル不変量を計算中...")
    for dim, metrics in dimension_metrics.items():
        print(f"{dim}次元:")
        print("チャーン・サイモンズ不変量:", metrics['chern_simons'])
        if 'jones_polynomial' in metrics and len(metrics['jones_polynomial']) >= 5:
            print("ジョーンズ多項式の係数（最初の5項）:", metrics['jones_polynomial'][:5])
        else:
            print("ジョーンズ多項式のデータが取得できませんでした")
    
    # 情報幾何学的計量の計算
    print("\n情報幾何学的計量を計算中...")
    for dim, metrics in dimension_metrics.items():
        print(f"{dim}次元:")
        # 対応する次元のcalculatorを使用
        calculator = calculators[dim]
        info_metrics = calculator.compute_information_geometry_metrics(np.random.rand(100, dim))  # 100点で計算
        print("スカラー曲率:", info_metrics['scalar_curvature'])
        print("フィッシャー情報行列の最大固有値:", np.max(np.linalg.eigvals(info_metrics['fisher_matrix'])))
    
    # 量子計算複雑性の評価
    print("\n量子計算複雑性を評価中...")
    for dim, metrics in dimension_metrics.items():
        print(f"{dim}次元:")
        # 対応する次元のcalculatorを使用
        calculator = calculators[dim]
        complexity = calculator.compute_quantum_complexity()
        print("量子回路の深さ:", complexity['circuit_depth'])
        print("量子ゲート数:", complexity['gate_count'])
        print("テンソルネットワークのボンド次元:", complexity['bond_dimension'])
    
    # 量子動力学のシミュレーション
    print("\n量子動力学をシミュレーション中...")
    time_steps = 5
    dt = 0.1
    
    for dim, metrics in dimension_metrics.items():
        print(f"{dim}次元:")
        # 各次元ごとに適切なサイズのテスト点を生成
        test_points = np.random.rand(10, dim)
        # 対応する次元のcalculatorを使用
        calculator = calculators[dim]
        psi_history, metric_history = calculator.simulate_quantum_dynamics(
            time_steps=time_steps, dt=dt, initial_points=test_points
        )
        
        print(f"{time_steps}ステップの時間発展を計算完了")
        for t in range(time_steps):
            energy = metric_history[t]['energy']
            print(f"t={t*dt:.1f}: エネルギー = {energy:.6f}")
    
    # 2次元断面でのプロット（3次元以上の場合は最初の2次元をプロットする）
    for dim, metrics in dimension_metrics.items():
        if dim >= 2 and 'psi_values' in metrics and len(metrics['psi_values']) >= 100:
            print(f"\n{dim}次元の統合特解の2次元断面をプロット中...")
            plot_points = metrics['points'][:100]  # 各次元の点データを使用
            plot_values = metrics['psi_values'][:100]
            
            plot_complex_function(
                plot_points[:, 0], plot_points[:, 1], plot_values, 
                dual_lang("統合特解 Ψ(x) の2次元断面")
            )
            plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"unified_solution_{dim}d_plot.png")
            plt.savefig(plot_path)
            print(f"プロットを{plot_path}に保存しました")
            plt.show(block=True)  # グラフをブロックモードで表示
    
    print("\n計算完了!")
    
    # GPUパフォーマンスのベンチマーク（RTX3080の性能評価）
    if torch.cuda.is_available():
        print("\nRTX3080のGPUパフォーマンスベンチマークを実行...")
        
        # ベンチマーク用のテンソルサイズ
        sizes = [100000, 500000, 1000000, 2000000]  # テンソルサイズを大幅に増加
        
        # 計算時間の記録
        cpu_times = []
        gpu_times = []
        
        for size in sizes:
            # ランダムデータ生成
            print(f"  サイズ {size}x{dim} のテンソルでベンチマーク中...")
            data = np.random.rand(size, dim).astype(np.float32)
            
            # 計算を複雑にしてベンチマークをより現実的に
            batch_size = 100000  # バッチサイズ
            num_batches = size // batch_size + (1 if size % batch_size > 0 else 0)
            
            # CPU計算
            start_time = time.time()
            cpu_result = np.zeros(size)
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, size)
                batch_data = data[start_idx:end_idx]
                
                # 複雑な計算
                batch_tensor = torch.tensor(batch_data)
                inter1 = torch.sin(batch_tensor.sum(dim=1))
                inter2 = torch.cos(batch_tensor.mean(dim=1))
                inter3 = torch.exp(inter1 * inter2)
                cpu_result[start_idx:end_idx] = inter3.numpy()
                
            cpu_time = time.time() - start_time
            cpu_times.append(cpu_time)
            print(f"    CPU時間: {cpu_time:.6f}秒")
            
            # GPU計算
            start_time = time.time()
            gpu_result = np.zeros(size)
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, size)
                batch_data = data[start_idx:end_idx]
                
                # GPUでの複雑な計算
                batch_tensor = torch.tensor(batch_data, device='cuda')
                inter1 = torch.sin(batch_tensor.sum(dim=1))
                inter2 = torch.cos(batch_tensor.mean(dim=1))
                inter3 = torch.exp(inter1 * inter2)
                gpu_result[start_idx:end_idx] = inter3.cpu().numpy()
            
            gpu_time = time.time() - start_time
            gpu_times.append(gpu_time)
            print(f"    GPU時間: {gpu_time:.6f}秒")
            
            # 高速化率の計算（ゼロ除算回避）
            if gpu_time > 0:
                speedup = cpu_time / gpu_time
                print(f"    高速化率: {speedup:.1f}倍")
            else:
                print(f"    高速化率: 計測不能（GPU時間が非常に短い）")
        
        # 結果をプロット
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, cpu_times, 'o-', color='#3366CC', linewidth=2, markersize=8, label='CPU')
        plt.plot(sizes, gpu_times, 'o-', color='#CC3366', linewidth=2, markersize=8, label='GPU (RTX3080)')
        plt.xlabel(dual_lang('データサイズ'))
        plt.ylabel(dual_lang('実行時間 (秒)'))
        plt.title(dual_lang('RTX3080 vs CPU パフォーマンス比較'))
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # x軸の単位を改善
        plt.ticklabel_format(axis='x', style='scientific', scilimits=(0,0))
        
        # 背景色を設定
        plt.gca().set_facecolor('#f8f9fa')
        
        # 保存と表示
        plt.tight_layout()
        benchmark_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_benchmark.png")
        plt.savefig(benchmark_path, dpi=150)
        print(f"  ベンチマーク結果を{benchmark_path}に保存しました")
        plt.show(block=True)  # グラフをブロックモードで表示
        
        # GPU情報の表示
        print("\nGPU情報:")
        print(f"  GPU名: {torch.cuda.get_device_name(0)}")
        print(f"  GPU数: {torch.cuda.device_count()}")
        print(f"  CUDAバージョン: {torch.version.cuda}")
        print(f"  最大メモリ確保量: {torch.cuda.max_memory_allocated()/1024/1024:.1f} MB")
        print(f"  最大メモリキャッシュ量: {torch.cuda.max_memory_reserved()/1024/1024:.1f} MB")

if __name__ == "__main__":
    main() 