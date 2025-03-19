import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
from mpl_toolkits.mplot3d import Axes3D

# リーマンゼータ関数の非自明なゼロ点の虚部
riemann_zeros = np.array([
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832
])

class RicciFlowSurgery:
    """リッチフロー方程式と時空曲率を用いた特異点解析クラス"""
    
    def __init__(self, nx=50, ny=50, T_max=1.0, dt=0.01):
        """
        初期化メソッド
        
        引数:
            nx, ny: 空間格子の解像度
            T_max: 最大時間
            dt: 時間ステップ
        """
        self.nx = nx
        self.ny = ny
        self.T_max = T_max
        self.dt = dt
        
        # 計算格子
        self.x = np.linspace(0, 1, nx)
        self.y = np.linspace(0, 1, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # メトリック初期化
        self.g = np.zeros((nx, ny, 2, 2))
        for i in range(nx):
            for j in range(ny):
                self.g[i, j] = np.eye(2)  # 初期メトリックは平坦なユークリッド計量
        
        # リッチ曲率テンソル
        self.Ric = np.zeros_like(self.g)
        
        # スカラー曲率
        self.R = np.zeros((nx, ny))
        
        # 特異点形成時間の推定値
        self.T_singularity = None
        
        # リーマンゼータ関数のゼロ点に基づく時空修正係数
        self.spacetime_correction = self._compute_spacetime_correction()
    
    def _compute_spacetime_correction(self):
        """リーマンゼータ関数のゼロ点に基づく時空修正係数を計算"""
        correction = np.zeros((self.nx, self.ny))
        
        for i, zero in enumerate(riemann_zeros[:5]):  # 最初の5つのゼロ点を使用
            # 複素位相因子
            phase = np.exp(2j * np.pi * i / len(riemann_zeros))
            
            # 空間依存の修正項
            correction += np.real(phase * np.exp(-(self.X**2 + self.Y**2) * zero))
        
        # 正規化
        correction = correction / np.max(np.abs(correction))
        
        return correction
    
    def set_initial_metric(self, metric_type='flat'):
        """
        初期メトリックを設定
        
        引数:
            metric_type: メトリックのタイプ ('flat', 'perturbed', 'curved')
        """
        if metric_type == 'flat':
            # 平坦なユークリッド計量
            for i in range(self.nx):
                for j in range(self.ny):
                    self.g[i, j] = np.eye(2)
        
        elif metric_type == 'perturbed':
            # 摂動を加えた計量
            for i in range(self.nx):
                for j in range(self.ny):
                    x, y = self.X[i, j], self.Y[i, j]
                    perturbation = 0.1 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
                    self.g[i, j] = np.eye(2) + perturbation * np.ones((2, 2))
        
        elif metric_type == 'curved':
            # 曲率のある計量（球面の一部をシミュレート）
            for i in range(self.nx):
                for j in range(self.ny):
                    x, y = self.X[i, j], self.Y[i, j]
                    r2 = (x - 0.5)**2 + (y - 0.5)**2
                    if r2 < 0.2**2:
                        # 球面計量（極座標変換）
                        factor = 1.0 + 5.0 * r2
                        self.g[i, j] = factor * np.eye(2)
                    else:
                        self.g[i, j] = np.eye(2)
        
        # リーマンゼータ関数のゼロ点に基づく修正
        for i in range(self.nx):
            for j in range(self.ny):
                correction = self.spacetime_correction[i, j]
                self.g[i, j] = self.g[i, j] * (1.0 + 0.2 * correction)
    
    def compute_ricci_curvature(self):
        """リッチ曲率テンソルとスカラー曲率を計算"""
        # 数値微分のためのステップサイズ
        h = 1.0 / self.nx
        
        # クリストッフェル記号を計算（数値微分）
        Gamma = np.zeros((self.nx, self.ny, 2, 2, 2))
        for i in range(1, self.nx-1):
            for j in range(1, self.ny-1):
                # メトリックの1階微分（中心差分）
                dg_dx = (self.g[i+1, j] - self.g[i-1, j]) / (2*h)
                dg_dy = (self.g[i, j+1] - self.g[i, j-1]) / (2*h)
                
                # メトリックの逆行列
                g_inv = np.linalg.inv(self.g[i, j])
                
                # クリストッフェル記号の計算
                for k in range(2):
                    for l in range(2):
                        for m in range(2):
                            Gamma[i, j, k, l, m] = 0.5 * (
                                dg_dx[m, l] * (k == 0) + dg_dy[m, l] * (k == 1) +
                                dg_dx[k, m] * (l == 0) + dg_dy[k, m] * (l == 1) -
                                dg_dx[k, l] * (m == 0) - dg_dy[k, l] * (m == 1)
                            )
                            
                            # g^{pq}を掛ける
                            Gamma_sum = 0
                            for p in range(2):
                                for q in range(2):
                                    Gamma_sum += g_inv[p, q] * Gamma[i, j, k, l, m]
                            Gamma[i, j, k, l, m] = Gamma_sum
        
        # リッチ曲率テンソルの計算
        self.Ric = np.zeros_like(self.g)
        for i in range(1, self.nx-1):
            for j in range(1, self.ny-1):
                for k in range(2):
                    for l in range(2):
                        # Rij = dΓ^m_{ij}/dx^m - dΓ^m_{im}/dx^j + Γ^p_{ij}Γ^m_{pm} - Γ^p_{im}Γ^m_{pj}
                        
                        # 第1項: dΓ^m_{kl}/dx^m (数値微分)
                        term1 = 0
                        for m in range(2):
                            if m == 0 and i < self.nx-2:
                                dGamma_dx = (Gamma[i+1, j, m, k, l] - Gamma[i-1, j, m, k, l]) / (2*h)
                                term1 += dGamma_dx
                            elif m == 1 and j < self.ny-2:
                                dGamma_dy = (Gamma[i, j+1, m, k, l] - Gamma[i, j-1, m, k, l]) / (2*h)
                                term1 += dGamma_dy
                        
                        # 第2項: dΓ^m_{km}/dx^l (数値微分)
                        term2 = 0
                        for m in range(2):
                            if l == 0 and i < self.nx-2:
                                dGamma_dx = (Gamma[i+1, j, m, k, m] - Gamma[i-1, j, m, k, m]) / (2*h)
                                term2 += dGamma_dx
                            elif l == 1 and j < self.ny-2:
                                dGamma_dy = (Gamma[i, j+1, m, k, m] - Gamma[i, j-1, m, k, m]) / (2*h)
                                term2 += dGamma_dy
                        
                        # 第3項: Γ^p_{kl}Γ^m_{pm}
                        term3 = 0
                        for p in range(2):
                            for m in range(2):
                                term3 += Gamma[i, j, p, k, l] * Gamma[i, j, m, p, m]
                        
                        # 第4項: Γ^p_{km}Γ^m_{pl}
                        term4 = 0
                        for p in range(2):
                            for m in range(2):
                                term4 += Gamma[i, j, p, k, m] * Gamma[i, j, m, p, l]
                        
                        # リッチテンソル成分
                        self.Ric[i, j, k, l] = term1 - term2 + term3 - term4
        
        # スカラー曲率の計算
        self.R = np.zeros((self.nx, self.ny))
        for i in range(self.nx):
            for j in range(self.ny):
                g_inv = np.linalg.inv(self.g[i, j])
                for k in range(2):
                    for l in range(2):
                        self.R[i, j] += g_inv[k, l] * self.Ric[i, j, k, l]
    
    def evolve_ricci_flow(self, steps=100):
        """リッチフロー方程式に従ってメトリックを発展させる"""
        dt = self.T_max / steps
        
        # 発展履歴を記録
        self.g_history = []
        self.R_history = []
        self.max_curvature_history = []
        
        # 初期状態を記録
        self.compute_ricci_curvature()
        self.g_history.append(self.g.copy())
        self.R_history.append(self.R.copy())
        self.max_curvature_history.append(np.max(np.abs(self.R)))
        
        # リッチフロー: dg/dt = -2*Ric(g)
        for step in range(steps):
            # リッチ曲率を計算
            self.compute_ricci_curvature()
            
            # メトリックを更新
            for i in range(self.nx):
                for j in range(self.ny):
                    self.g[i, j] -= 2.0 * dt * self.Ric[i, j]
                    
                    # 正定値性を保証
                    eigenvalues, eigenvectors = eigh(self.g[i, j])
                    eigenvalues = np.maximum(eigenvalues, 0.1)  # 最小固有値を制限
                    self.g[i, j] = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            
            # 状態を記録
            self.g_history.append(self.g.copy())
            self.R_history.append(self.R.copy())
            self.max_curvature_history.append(np.max(np.abs(self.R)))
            
            # 特異点形成の検出
            if self.max_curvature_history[-1] > 1000:
                self.T_singularity = (step + 1) * dt
                print(f"特異点形成を検出: T ≈ {self.T_singularity:.6f}")
                break
    
    def perform_surgery(self):
        """特異点近傍で手術を実行する"""
        if self.T_singularity is None:
            print("特異点が検出されていないため、手術は実行できません。")
            return False
        
        # 高曲率領域の検出
        threshold = np.max(np.abs(self.R)) * 0.5  # 最大曲率の50%をしきい値とする
        surgery_mask = np.abs(self.R) > threshold
        
        if not np.any(surgery_mask):
            print("手術が必要な高曲率領域が見つかりませんでした。")
            return False
        
        print(f"手術を実行: {np.sum(surgery_mask)} 点で高曲率を検出")
        
        # 手術の実行: 高曲率領域をユークリッド計量で置換
        for i in range(self.nx):
            for j in range(self.ny):
                if surgery_mask[i, j]:
                    # 近傍のメトリックの平均を取る
                    neighbors = []
                    for di in [-2, -1, 1, 2]:
                        for dj in [-2, -1, 1, 2]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < self.nx and 0 <= nj < self.ny and not surgery_mask[ni, nj]:
                                neighbors.append(self.g[ni, nj])
                    
                    if neighbors:
                        # 近傍の平均メトリックを使用
                        avg_metric = sum(neighbors) / len(neighbors)
                        self.g[i, j] = avg_metric
                    else:
                        # 近傍がない場合はユークリッド計量を使用
                        self.g[i, j] = np.eye(2)
        
        # 手術後の曲率を再計算
        self.compute_ricci_curvature()
        
        print(f"手術後の最大曲率: {np.max(np.abs(self.R)):.6f}")
        return True
    
    def apply_spacetime_quantum_correction(self):
        """リーマンゼータ関数のゼロ点に基づく量子補正を適用"""
        for i in range(self.nx):
            for j in range(self.ny):
                # リーマンゼロ点による量子補正
                quantum_factor = 0
                for k, zero in enumerate(riemann_zeros[:5]):
                    # リーマンゼロ点に依存する量子補正
                    phase = np.exp(2j * np.pi * k / 5)
                    r2 = (self.X[i, j] - 0.5)**2 + (self.Y[i, j] - 0.5)**2
                    quantum_factor += np.real(phase * np.exp(-zero * r2))
                
                # 正規化
                quantum_factor = quantum_factor / 5.0
                
                # 量子補正をメトリックに適用
                self.g[i, j] = self.g[i, j] * (1.0 + 0.1 * quantum_factor)
        
        # 補正後の曲率を再計算
        self.compute_ricci_curvature()
    
    def analyze_singularity(self):
        """特異点形成のパターンを解析"""
        if not hasattr(self, 'max_curvature_history'):
            print("リッチフローの発展が実行されていません。")
            return
        
        # 最大曲率の時間発展
        plt.figure(figsize=(10, 6))
        time_points = np.linspace(0, self.T_max, len(self.max_curvature_history))
        plt.semilogy(time_points, self.max_curvature_history)
        plt.grid(True)
        plt.xlabel('時間 t')
        plt.ylabel('最大スカラー曲率 |R|_max')
        plt.title('リッチフロー下での最大曲率の発展')
        
        if self.T_singularity is not None:
            plt.axvline(x=self.T_singularity, color='r', linestyle='--', 
                        label=f'特異点形成時間 T ≈ {self.T_singularity:.4f}')
            
            # 特異点近傍での振る舞いをフィッティング
            idx = len(self.max_curvature_history) // 2
            time_near_singularity = time_points[idx:]
            curvature_near_singularity = self.max_curvature_history[idx:]
            
            # 対数変換してフィッティング
            if len(time_near_singularity) > 2:
                from scipy.optimize import curve_fit
                
                def singularity_model(t, T_s, alpha):
                    return alpha / (T_s - t)
                
                try:
                    # 特異点形成時間と指数のフィッティング
                    popt, _ = curve_fit(singularity_model, time_near_singularity, curvature_near_singularity,
                                       p0=[self.T_singularity, 1.0], maxfev=10000)
                    T_s_fit, alpha_fit = popt
                    
                    # フィッティング曲線のプロット
                    t_fit = np.linspace(time_points[idx], T_s_fit * 0.99, 100)
                    plt.semilogy(t_fit, singularity_model(t_fit, T_s_fit, alpha_fit), 'g--',
                                linewidth=2, label=f'フィット: R ≈ {alpha_fit:.2f}/(T-t), T ≈ {T_s_fit:.4f}')
                    
                    # リーマン予想との関連を分析
                    riemann_time = 0.5 / riemann_zeros[0]  # 最初のリーマンゼロ点との関係
                    plt.axvline(x=riemann_time, color='b', linestyle=':', 
                               label=f'リーマン時間スケール: {riemann_time:.4f}')
                except:
                    print("特異点モデルのフィッティングに失敗しました。")
        
        plt.legend()
        plt.savefig('ricci_flow_singularity_analysis.png')
        plt.show()
        
        # 特異点近傍の曲率分布を可視化
        if self.T_singularity is not None:
            plt.figure(figsize=(10, 6))
            plt.contourf(self.X, self.Y, np.abs(self.R), 50, cmap='hot')
            plt.colorbar(label='スカラー曲率の絶対値 |R|')
            plt.title(f'特異点形成直前 (T = {self.T_singularity:.4f}) の曲率分布')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.savefig('ricci_flow_curvature_near_singularity.png')
            plt.show()
    
    def compare_with_navier_stokes(self):
        """リッチフローとナビエストークス方程式の特異点形成を比較"""
        from ns_quantum_kat_part2 import evaluate_global_existence_condition
        
        # ナビエストークス方程式の大域解存在条件を評価
        c_fluid_values = np.linspace(1.0, 10.0, 100)
        satisfy_condition = []
        
        for c_fluid in c_fluid_values:
            condition_satisfied, ratio, threshold = evaluate_global_existence_condition(c_fluid)
            satisfy_condition.append(condition_satisfied)
        
        # 条件を満たし始める臨界値を見つける
        critical_idx = np.argmax(satisfy_condition)
        if critical_idx > 0:
            critical_c_fluid = c_fluid_values[critical_idx]
        else:
            critical_c_fluid = np.inf
        
        # リッチフローの特異点形成時間との比較
        if self.T_singularity is not None:
            # 理論的関係: T_singularity ∝ 1/(critical_c_fluid)
            theory_constant = self.T_singularity * critical_c_fluid if critical_c_fluid < np.inf else 1.0
            
            plt.figure(figsize=(10, 6))
            plt.plot(c_fluid_values, theory_constant / c_fluid_values, 'b-', 
                     label='理論予測: $T_{sing} \\propto 1/c_{fluid}$')
            plt.axhline(y=self.T_singularity, color='r', linestyle='--',
                        label=f'リッチフロー特異点時間: {self.T_singularity:.4f}')
            plt.axvline(x=critical_c_fluid, color='g', linestyle='--',
                        label=f'臨界 c_fluid: {critical_c_fluid:.4f}')
            
            plt.grid(True)
            plt.xlabel('流体パラメータ $c_{fluid}$')
            plt.ylabel('特異点形成時間 $T_{sing}$')
            plt.title('リッチフローとナビエストークス方程式の特異点形成比較')
            plt.legend()
            plt.savefig('ricci_flow_navier_stokes_comparison.png')
            plt.show()
            
            # リーマンゼータ関数のゼロ点との関係
            first_zero = riemann_zeros[0]
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(riemann_zeros)+1), riemann_zeros, 'bo', label='リーマンゼータ関数のゼロ点 $\\gamma_n$')
            plt.axhline(y=1/self.T_singularity, color='r', linestyle='--',
                      label=f'1/T_{{sing}} = {1/self.T_singularity:.4f}')
            plt.grid(True)
            plt.xlabel('ゼロ点のインデックス n')
            plt.ylabel('$\\gamma_n$')
            plt.title('リーマンゼータ関数のゼロ点と特異点形成時間の関係')
            plt.legend()
            plt.savefig('ricci_flow_riemann_zeros_relation.png')
            plt.show()

# 実行例
if __name__ == "__main__":
    # リッチフローソルバーを初期化
    solver = RicciFlowSurgery(nx=50, ny=50, T_max=1.0, dt=0.01)
    
    # 初期メトリックを設定
    solver.set_initial_metric('perturbed')
    
    # リッチフローを発展させる
    print("リッチフロー方程式を解いています...")
    solver.evolve_ricci_flow(steps=500)
    
    # 特異点形成の解析
    solver.analyze_singularity()
    
    # 特異点が形成された場合、手術を実行
    if solver.T_singularity is not None:
        solver.perform_surgery()
        
        # 量子補正を適用
        solver.apply_spacetime_quantum_correction()
        
        # 手術後の特異点形成を解析
        print("\n手術後のリッチフローを発展させています...")
        solver.evolve_ricci_flow(steps=500)
        solver.analyze_singularity()
    
    # ナビエストークス方程式との比較
    solver.compare_with_navier_stokes() 