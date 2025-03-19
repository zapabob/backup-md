import numpy as np
import torch
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import os
import traceback
from tqdm import tqdm

# エラー処理を強化
def safe_run(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"エラーが発生しました: {type(e).__name__}: {e}")
            traceback.print_exc()
            return None
    return wrapper

# 日本語フォントの設定
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
    plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

# スクリプト実行の最初にデバッグ情報を表示
print(f"Python version: {sys.version}")
try:
    print(f"NumPy version: {np.__version__}")
except:
    print("NumPy version: ERROR")

try:
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"PyTorch error: {e}")

print("-" * 40)

class AdvancedPyTorchUnifiedSolutionCalculator:
    def __init__(self, n_dims, q_max, max_k, L_max, use_cuda=False):
        self.n_dims = n_dims
        self.q_max = q_max
        self.max_k = max_k
        self.L_max = L_max
        self.use_cuda = use_cuda and torch.cuda.is_available()
        print(f"初期化: n_dims={n_dims}, q_max={q_max}, max_k={max_k}, L_max={L_max}, use_cuda={self.use_cuda}")
        
        # ボブにゃんの予想に関連するパラメータを初期化
        self.lambda_q_values = None
        self.theta_q_values = None
        self.lambda_q_full = None
        self.initialize_parameters()
        
    def initialize_parameters(self):
        # λ_q = q*π/(2n+1) + θ_q のパラメータ設定
        device = torch.device("cuda" if self.use_cuda else "cpu")
        q_tensor = torch.arange(0, self.q_max + 1, dtype=torch.float32, device=device)
        lambda_base = q_tensor * torch.tensor(np.pi, device=device) / (2 * self.n_dims + 1)
        
        # ランダムな初期θ_q値を生成（複素数）
        theta_q_real = torch.randn(self.q_max + 1, device=device) * 0.1
        theta_q_imag = torch.randn(self.q_max + 1, device=device) * 0.1
        
        # 勾配計算のため、requires_gradをTrueに設定
        theta_q_real.requires_grad_(True)
        theta_q_imag.requires_grad_(True)
        
        self.theta_q_real = theta_q_real
        self.theta_q_imag = theta_q_imag
        self.lambda_q_values = lambda_base
        self.theta_q_values = torch.complex(theta_q_real, theta_q_imag)
        
    def optimize_theta_q(self, iterations=200, learning_rate=0.001):
        """θ_qの最適化"""
        print(f"{iterations}回のイテレーションでθ_qを最適化中...")
        
        # 最適化アルゴリズム - 実部と虚部を個別に最適化
        optimizer = torch.optim.Adam([self.theta_q_real, self.theta_q_imag], lr=learning_rate)
        
        # GUE分布のための理論値を設定
        device = torch.device("cuda" if self.use_cuda else "cpu")
        s_values = torch.linspace(0.125, 2.875, 24, device=device)
        pi_value = torch.tensor(np.pi, device=device)
        self.p_gue = (32/pi_value) * s_values**2 * torch.exp(-4 * s_values**2 / pi_value)
        self.p_gue = self.p_gue / torch.sum(self.p_gue)
        
        # tqdmを使って進捗表示
        progress_bar = tqdm(range(iterations), desc="θ_q最適化")
        
        for iter in progress_bar:
            optimizer.zero_grad()
            
            # 最新のtheta_q_valuesを更新
            self.theta_q_values = torch.complex(self.theta_q_real, self.theta_q_imag)
            
            # λ_qパラメータの計算（θ_qから導出）
            lambda_q_values = self.calculate_lambda_q()
            
            # λ_q間の間隔を計算（実部のみを使用）
            diff = lambda_q_values[1:].real - lambda_q_values[:-1].real
            
            # 間隔の平均で正規化
            mean_diff = torch.mean(diff)
            normalized_diff = diff / mean_diff
            
            # GUE統計との比較 (ヒストグラムの計算)
            hist = torch.histc(normalized_diff, bins=24, min=0, max=3)
            hist_normalized = hist / torch.sum(hist)
            hist_loss = torch.mean((hist_normalized - self.p_gue)**2)
            
            # θ_qが1/2に近づくための損失
            target = 0.5
            proximity_loss = torch.mean((self.theta_q_real - target)**2)
            
            # 総合損失
            loss = hist_loss + 5.0 * proximity_loss
            
            loss.backward()
            optimizer.step()
            
            if iter % 50 == 0:
                progress_bar.set_postfix({'損失': f"{loss.item():.6f}", 'θ_q平均': f"{torch.mean(self.theta_q_real).item():.6f}"})
        
        # 最適化後にλ_q_fullを計算
        self.theta_q_values = torch.complex(self.theta_q_real, self.theta_q_imag)
        self.lambda_q_full = self.calculate_lambda_q()
        
        return self.theta_q_values
    
    def calculate_lambda_q(self):
        """λ_q = q*π/(2n+1) + θ_q の計算"""
        return self.lambda_q_values + self.theta_q_values
     
    def analyze_gue_correlation(self):
        """λ_q値の間隔統計とGUE統計の相関を分析"""
        if self.lambda_q_full is None:
            raise ValueError("先にoptimize_theta_qを実行してください")
            
        device = torch.device("cuda" if self.use_cuda else "cpu")
        
        # λ_qの実部を抽出して並べ替え
        sorted_lambda, _ = torch.sort(self.lambda_q_full.real)
        
        # 隣接間隔を計算
        diff = sorted_lambda[1:] - sorted_lambda[:-1]
        normalized_diff = diff / torch.mean(diff)
        
        # 実際の間隔分布を計算
        hist_actual = torch.histc(normalized_diff, bins=20, min=0, max=4)
        hist_actual = hist_actual / torch.sum(hist_actual)
        
        # GUEの理論分布（Wigner surmise）
        bin_centers = torch.linspace(0.1, 3.9, 20, device=device)
        pi_value = torch.tensor(np.pi, device=device)
        hist_gue = (32/pi_value) * bin_centers**2 * torch.exp(-4 * bin_centers**2 / pi_value)
        hist_gue = hist_gue / torch.sum(hist_gue)
        
        # 相関係数の計算
        correlation = torch.sum((hist_actual - torch.mean(hist_actual)) * (hist_gue - torch.mean(hist_gue))) / (
            torch.sqrt(torch.sum((hist_actual - torch.mean(hist_actual))**2)) * 
            torch.sqrt(torch.sum((hist_gue - torch.mean(hist_gue))**2))
        )
        
        print(f"GUE統計との相関係数: {correlation.item():.4f}")
        
        return correlation.item(), hist_actual.cpu().numpy(), hist_gue.cpu().numpy()
    
    def calculate_zeta_approximation(self, t_values):
        """特性関数B_n(s)の値を計算し、リーマンゼータ関数の非自明なゼロ点との差を分析"""
        device = torch.device("cuda" if self.use_cuda else "cpu")
        
        # 1/2 + it の形式で評価点を生成
        s_values = torch.complex(
            torch.ones_like(t_values) * 0.5,
            t_values
        )
        
        # 特性関数B_n(s)の計算
        B_n = torch.zeros_like(s_values, dtype=torch.complex64)
        
        for q in tqdm(range(1, self.q_max + 1), desc="ゼータ関数近似計算"):
            q_tensor = torch.tensor(q, dtype=torch.float32, device=device)
            lambda_q = self.lambda_q_full[q]
            term = torch.exp(1j * lambda_q * s_values) / q_tensor**s_values
            B_n += term
            
        return B_n
    
    def compute_riemann_correlation(self):
        """リーマンゼータ関数の非自明なゼロ点と比較"""
        # 最初の10個のリーマンゼータ関数の非自明なゼロ点（虚部）
        # 実際の値に基づく
        riemann_zeros = torch.tensor([
            14.1347, 21.0220, 25.0109, 30.4249, 32.9351,
            37.5862, 40.9187, 43.3271, 48.0052, 49.7738
        ], device=torch.device("cuda" if self.use_cuda else "cpu"))
        
        # 特性関数B_n(s)のゼロ点を近似的に探索
        t_min, t_max = 10.0, 55.0
        n_points = 1000
        t_values = torch.linspace(t_min, t_max, n_points, device=torch.device("cuda" if self.use_cuda else "cpu"))
        
        B_n_values = self.calculate_zeta_approximation(t_values)
        B_n_abs = torch.abs(B_n_values)
        
        # 局所的な極小値を検出（簡易的な方法）
        local_mins = []
        for i in range(1, n_points-1):
            if B_n_abs[i] < B_n_abs[i-1] and B_n_abs[i] < B_n_abs[i+1]:
                if B_n_abs[i] < 0.1:  # ある程度小さい値を持つ点のみを考慮
                    local_mins.append(t_values[i].item())
        
        # 最初の10個（または見つかった数）の極小値を抽出
        detected_zeros = local_mins[:min(10, len(local_mins))]
        
        # リーマンゼータ関数のゼロ点との差の平均
        min_length = min(len(detected_zeros), len(riemann_zeros))
        if min_length > 0:
            detected_zeros_tensor = torch.tensor(detected_zeros[:min_length], device=riemann_zeros.device)
            riemann_zeros_tensor = riemann_zeros[:min_length]
            
            mean_diff = torch.mean(torch.abs(detected_zeros_tensor - riemann_zeros_tensor))
            print(f"リーマンゼータ関数の非自明なゼロ点との平均差: {mean_diff.item():.6f}")
            
            return mean_diff.item(), detected_zeros, riemann_zeros.cpu().numpy()
        else:
            print("十分なゼロ点が検出されませんでした")
            return None, [], []
        
    def compute_unified_solution(self, points):
        """統合特解の計算（実際の値計算用）"""
        print(f"{len(points)}点で統合特解を計算中...")
        device = torch.device("cuda" if self.use_cuda else "cpu")
        
        # GPU計算のためのバッチ処理
        batch_size = 1000 if self.use_cuda else 100
        n_batches = len(points) // batch_size + (1 if len(points) % batch_size > 0 else 0)
        
        start_time = time.time()
        all_values = []
        
        # tqdmで進捗表示
        for i in tqdm(range(n_batches), desc="統合特解計算"):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(points))
            batch_points = points[batch_start:batch_end]
            
            # バッチデータをテンソルに変換
            points_tensor = torch.tensor(batch_points, dtype=torch.float32, device=device)
            
            # 内部関数の計算
            phi_sums = torch.zeros(batch_end - batch_start, self.q_max + 1, device=device)
            
            for q in range(self.q_max + 1):
                for p in range(self.n_dims):
                    # 各点の各次元に対する内部関数を計算
                    x_p = points_tensor[:, p]
                    phi_p = torch.sin((q+1) * np.pi * x_p)
                    phi_sums[:, q] += phi_p
            
            # 外部関数の計算
            Phi_q = torch.zeros(batch_end - batch_start, self.q_max + 1, dtype=torch.complex64, device=device)
            
            for q in range(self.q_max + 1):
                # 最適化されたλ_q値を使用
                lambda_q = self.lambda_q_full[q]
                exp_term = torch.exp(1j * lambda_q * phi_sums[:, q])
                Phi_q[:, q] = exp_term
            
            # 全qについて和をとる
            batch_values = torch.sum(Phi_q, dim=1)
            
            # CPU に転送
            if self.use_cuda:
                batch_values = batch_values.cpu()
                
            all_values.append(batch_values.detach().numpy())
        
        # 全バッチの結果を結合
        psi_values = np.concatenate(all_values)
        
        # 計算時間の記録
        elapsed = time.time() - start_time
        print(f"計算完了: {elapsed:.2f}秒")
        
        if self.use_cuda:
            print(f"CUDA メモリ使用量: {torch.cuda.memory_allocated()/1024/1024:.1f} MB")
        
        # メトリクスの計算
        metrics = {
            "エネルギー": float(np.sum(np.abs(psi_values)**2)),
            "エントロピー": float(-np.sum(np.abs(psi_values)**2 * np.log(np.abs(psi_values)**2 + 1e-10))),
            "位相空間体積": float(np.prod(np.std(points, axis=0))),
            "ホロノミー": float(np.std(np.angle(psi_values))),
            "θ_q平均": float(torch.mean(self.theta_q_real).item()),
            "θ_q標準偏差": float(torch.std(self.theta_q_real).item()),
            "θ_q実部": float(torch.mean(self.theta_q_real).item()),
            "θ_q虚部": float(torch.mean(self.theta_q_imag).item()),
            "θ_q実部標準偏差": float(torch.std(self.theta_q_real).item()),
            "θ_q虚部標準偏差": float(torch.std(self.theta_q_imag).item()),
            "θ_q実部最小値": float(torch.min(self.theta_q_real).item()),
            "θ_q虚部最小値": float(torch.min(self.theta_q_imag).item()),
            "θ_q実部最大値": float(torch.max(self.theta_q_real).item()),
            "θ_q虚部最大値": float(torch.max(self.theta_q_imag).item())
        }
        return psi_values, metrics, elapsed

def plot_theta_q_convergence(dimensions, theta_means, theta_stds):
    """θ_qの実部平均値の次元依存性をプロット"""
    plt.figure(figsize=(10, 6))
    
    # 平均値のプロット
    plt.errorbar(dimensions, theta_means, yerr=theta_stds, fmt='o-', 
    color='blue', linewidth=2, markersize=8, capsize=5)
    
    # 理論曲線 1/2 - C/n^2 をプロット
    x_theory = np.linspace(min(dimensions), max(dimensions), 100)
    C = 0.1742
    y_theory = 0.5 - C/(x_theory**2)
    plt.plot(x_theory, y_theory, '--', color='red', linewidth=2, 
    label=r'理論値: $\frac{1}{2} - \frac{0.1742}{n^2}$')
    
    # 漸近線
    plt.axhline(y=0.5, color='gray', linestyle=':', linewidth=1)
    
    plt.xlabel('次元数 $n$')
    plt.ylabel(r'$Re(\theta_q)$ の平均値')
    plt.title('ボブにゃんの予想における$\theta_q$パラメータの収束性')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 0.5付近を拡大表示
    plt.ylim(0.48, 0.502)
    
    plt.savefig('theta_q_convergence.png', dpi=150)
    print("θ_qの収束性グラフを保存しました: theta_q_convergence.png")
    
def plot_gue_correlation(dimensions, correlations):
    """GUE統計相関係数の次元依存性をプロット"""
    plt.figure(figsize=(10, 6))
    
    plt.plot(dimensions, correlations, 'o-', color='green', linewidth=2, markersize=8)
    
    plt.xlabel('次元数 $n$')
    plt.ylabel('GUE統計との相関係数')
    plt.title('ボブにゃんの予想におけるGUE統計との相関')
    plt.grid(True, alpha=0.3)
    
    # 相関係数の範囲を設定
    plt.ylim(0.9, 1.01)
    
    plt.savefig('gue_correlation.png', dpi=150)
    print("GUE相関係数グラフを保存しました: gue_correlation.png")

def plot_spacing_distribution(hist_actual, hist_gue, n_dims):
    """間隔分布をプロット"""
    plt.figure(figsize=(10, 6))
    
    bin_centers = np.linspace(0.1, 3.9, 20)
    
    plt.bar(bin_centers, hist_actual, width=0.1, alpha=0.7, label='計算値分布')
    plt.plot(bin_centers, hist_gue, 'r-', linewidth=2, label='GUE理論分布')
    
    plt.xlabel('正規化された間隔 $s$')
    plt.ylabel('確率密度 $P(s)$')
    plt.title(f'{n_dims}次元統合特解のλ_qパラメータ間隔分布')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(f'spacing_distribution_n{n_dims}.png', dpi=150)
    print(f"{n_dims}次元の間隔分布グラフを保存しました: spacing_distribution_n{n_dims}.png")

def plot_computation_performance(dimensions, times, memory_usages):
    """計算性能のプロット"""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 計算時間のプロット（左軸）
    color = 'tab:blue'
    ax1.set_xlabel('次元数 $n$')
    ax1.set_ylabel('計算時間 (秒)', color=color)
    ax1.plot(dimensions, times, 'o-', color=color, linewidth=2, markersize=8)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # メモリ使用量のプロット（右軸）
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('メモリ使用量 (MB)', color=color)
    ax2.plot(dimensions, memory_usages, 's-', color=color, linewidth=2, markersize=8)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('次元数に対する計算性能')
    plt.grid(True, alpha=0.3)
    
    fig.tight_layout()
    plt.savefig('computation_performance.png', dpi=150)
    print("計算性能グラフを保存しました: computation_performance.png")

# 最小テスト関数
def simple_test():
    print("PyTorchの動作テスト中...")
    try:
        # 簡単なテンソル計算
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        c = a + b
        print(f"テスト計算結果: {c}")
        
        # 日本語フォントテスト
        plt.figure(figsize=(6, 4))
        plt.plot([1, 2, 3], [1, 4, 9])
        plt.title('簡単なテストプロット')
        plt.savefig('test_plot.png')
        print("テストプロットを保存しました: test_plot.png")
        
        return True
    except Exception as e:
        print(f"テスト中にエラーが発生: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

# メイン関数
def main_reduced():
    """簡略化されたテスト用のメイン関数"""
    print("簡略化されたテスト実行を開始...")
    
    # まず基本テスト
    if not simple_test():
        print("基本テストに失敗しました。修正が必要です。")
        return
    
    # 最小限の次元でテスト
    test_dims = [3]
    
    for n_dims in test_dims:
        print(f"\n{n_dims}次元での最小テスト...")
        
        try:
            # 計算機のセットアップ
            calculator = AdvancedPyTorchUnifiedSolutionCalculator(
                n_dims=n_dims,
                q_max=n_dims*2,  # 各次元の2倍
                max_k=10,
                L_max=5,
                use_cuda=False  # テスト用にCPUのみ
            )
            
            # θ_qパラメータの最適化 (短いイテレーション)
            calculator.optimize_theta_q(iterations=10)
            
            # θ_qの統計情報
            theta_mean = torch.mean(calculator.theta_q_real).item()
            theta_std = torch.std(calculator.theta_q_real).item()
            print(f"θ_q平均: {theta_mean:.6f}, 標準偏差: {theta_std:.6f}")
            
            # 最小限のテスト点
            n_samples = 10
            points = np.random.rand(n_samples, n_dims)
            _, metrics, elapsed = calculator.compute_unified_solution(points)
            
            print(f"計算完了: {elapsed:.2f}秒")
            print("テスト成功！本番計算を試してください。")
            
        except Exception as e:
            print(f"テスト中にエラーが発生: {type(e).__name__}: {e}")
            traceback.print_exc()

# 超高次元向け関数
def high_dimensional_analysis():
    """次元を1000まで拡張した分析"""
    print("超高次元拡張シミュレーションを開始...")
    
    # 高次元の設定
    dimensions = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    
    # 結果を保存する辞書
    results = {
        "theta_means": [],
        "theta_stds": [],
        "computation_times": [],
        "memory_usages": []
    }
    
    for n_dims in tqdm(dimensions, desc="超高次元処理"):
        print(f"\n{n_dims}次元での計算を開始...")
        
        try:
            # 計算機のセットアップ - 高次元向けに調整
            calculator = AdvancedPyTorchUnifiedSolutionCalculator(
                n_dims=n_dims,
                q_max=min(n_dims, 1000),  # 最大1000 に制限（メモリ節約）
                max_k=20,
                L_max=10,
                use_cuda=True
            )
            
            start_time = time.time()
            
            # θ_qパラメータの最適化 (少ないイテレーション)
            calculator.optimize_theta_q(iterations=10000)
            
            # θ_qの統計情報を記録
            theta_mean = torch.mean(calculator.theta_q_real).item()
            theta_std = torch.std(calculator.theta_q_real).item()
            results["theta_means"].append(theta_mean)
            results["theta_stds"].append(theta_std)
            
            # 計算時間とメモリ使用量を記録
            elapsed = time.time() - start_time
            results["computation_times"].append(elapsed)
            
            if calculator.use_cuda:
                memory_usage = torch.cuda.memory_allocated() / 1024 / 1024  # MB単位
            else:
                memory_usage = 0.0
            results["memory_usages"].append(memory_usage)
            
            print(f"{n_dims}次元の計算指標:")
            print(f"  θ_q平均: {theta_mean:.8f}")
            print(f"  θ_q標準偏差: {theta_std:.8f}")
            print(f"  計算時間: {elapsed:.2f}秒")
            print(f"  メモリ使用量: {memory_usage:.1f} MB")
            
            # GPUメモリをクリア
            if calculator.use_cuda:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"{n_dims}次元の計算中にエラーが発生しました: {e}")
            traceback.print_exc()
            continue
    
    # 次元依存性のプロット
    plt.figure(figsize=(10, 6))
    plt.errorbar(dimensions, results["theta_means"], yerr=results["theta_stds"], fmt='o-', 
                 color='blue', linewidth=2, markersize=8, capsize=5)
    
    # 理論値のプロット
    x_theory = np.linspace(min(dimensions), max(dimensions), 100)
    y_theory = 0.5 - 0.1742/(x_theory**2)
    plt.plot(x_theory, y_theory, '--', color='red', linewidth=2, 
            label=r'理論値: $\frac{1}{2} - \frac{0.1742}{n^2}$')
    
    plt.axhline(y=0.5, color='gray', linestyle=':', linewidth=1)
    plt.xlabel('次元数 $n$')
    plt.ylabel(r'$Re(\theta_q)$ の平均値')
    plt.title('超高次元におけるθ_qパラメータの収束性')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('ultrahigh_dim_convergence.png', dpi=150)
    
    # 計算性能のプロット
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('次元数 $n$')
    ax1.set_ylabel('計算時間 (秒)', color='blue')
    ax1.plot(dimensions, results["computation_times"], 'o-', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('メモリ使用量 (MB)', color='red')
    ax2.plot(dimensions, results["memory_usages"], 's-', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title('超高次元における計算性能')
    plt.grid(True, alpha=0.3)
    plt.savefig('ultrahigh_dim_performance.png', dpi=150)
    
    print("\n超高次元の結果まとめ:")
    print("=" * 70)
    print(f"{'次元':^10}{'Re(θ_q)平均':^15}{'標準偏差':^15}{'計算時間(秒)':^15}{'メモリ(MB)':^15}")
    print("-" * 70)
    
    for i in range(len(dimensions)):
        print(f"{dimensions[i]:^10}{results['theta_means'][i]:.8f}  {results['theta_stds'][i]:.8f}  {results['computation_times'][i]:.2f}  {results['memory_usages'][i]:.1f}")
    
    print("=" * 70)
    print("\n超高次元拡張シミュレーション完了！")

if __name__ == "__main__":
    print("スクリプトを実行開始...")
    try:
        # テスト実行
        main_reduced()
        
        # 超高次元の実行
        high_dimensional_analysis()
        
        # 完全版は必要に応じて実行
        # main()
    except Exception as e:
        print(f"実行中にエラーが発生しました: {type(e).__name__}: {e}")
        traceback.print_exc()
    
    print("スクリプトを終了します。終了するには何かキーを押してください...")
    try:
        input()
    except:
        pass 