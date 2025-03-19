import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
from scipy.special import sph_harm
import matplotlib.animation as animation

# リーマンゼータ関数の非自明なゼロ点の虚部
riemann_zeros = np.array([
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832
])

class EinsteinWormhole:
    """アインシュタイン方程式に基づく計算論的ワームホールのシミュレーションクラス"""
    
    def __init__(self, n_modes=10, t_max=10.0, dt=0.01, throat_radius=1.0, quantum_factor=0.1):
        """
        初期化メソッド
        
        引数:
            n_modes: 球面調和関数のモード数
            t_max: 最大時間
            dt: 時間ステップ
            throat_radius: ワームホールの喉の初期半径
            quantum_factor: 量子効果の強さ
        """
        self.n_modes = n_modes
        self.t_max = t_max
        self.dt = dt
        self.throat_radius = throat_radius
        self.quantum_factor = quantum_factor
        
        # 三次元球面S³の座標系パラメータ
        self.n_theta = 50  # 天頂角分割数
        self.n_phi = 50    # 方位角分割数
        self.n_psi = 50    # 第3角度分割数
        
        # S³の座標グリッド
        self.theta = np.linspace(0, np.pi, self.n_theta)
        self.phi = np.linspace(0, 2*np.pi, self.n_phi)
        self.psi = np.linspace(0, 2*np.pi, self.n_psi)
        
        # 計量テンソル初期化（単位球面計量に喉のサイズを掛ける）
        self.initialize_metric()
        
        # アインシュタインテンソル
        self.G = np.zeros((4, 4, self.n_theta, self.n_phi))
        
        # リッチテンソル
        self.Ric = np.zeros((4, 4, self.n_theta, self.n_phi))
        
        # スカラー曲率
        self.R = np.zeros((self.n_theta, self.n_phi))
        
        # 量子エンタングルメントエントロピー
        self.entropy = np.zeros((self.n_theta, self.n_phi))
        
        # リーマンゼータ関数に基づく量子補正
        self.quantum_correction = self._compute_quantum_correction()
    
    def _compute_quantum_correction(self):
        """リーマンゼータ関数のゼロ点に基づく量子補正を計算"""
        correction = np.zeros((self.n_theta, self.n_phi))
        
        # 球面調和関数ベースの補正
        for l in range(1, min(6, self.n_modes)):
            for m in range(-l, l+1):
                THETA, PHI = np.meshgrid(self.theta, self.phi, indexing='ij')
                
                # 球面調和関数
                Y_lm = sph_harm(m, l, PHI, THETA)
                
                # 対応するリーマンゼロ点を使用
                zero_idx = (l - 1) % len(riemann_zeros)
                gamma = riemann_zeros[zero_idx]
                
                # 複素位相因子とゼロ点に基づく重み付け
                phase = np.exp(2j * np.pi * l / len(riemann_zeros))
                weight = 1.0 / (l**2 + gamma**2 + 0.25)
                
                # 実部を取って補正に加算
                correction += weight * np.real(phase * Y_lm)
        
        # 正規化
        correction = correction / np.max(np.abs(correction))
        
        return correction
    
    def initialize_metric(self):
        """三次元球面と同相なワームホールの喉の計量を初期化"""
        # 4次元時空の計量テンソル (t, θ, φ, ψ)
        self.g = np.zeros((4, 4, self.n_theta, self.n_phi))
        
        # 各点での計量テンソルを設定
        for i in range(self.n_theta):
            for j in range(self.n_phi):
                # 時間成分
                self.g[0, 0, i, j] = -1.0
                
                # 球面成分にワームホールの喉の半径を組み込む
                r_throat = self.throat_radius * (1.0 + 0.1 * np.sin(self.theta[i]) * np.cos(self.phi[j]))
                
                # 三次元球面の計量
                self.g[1, 1, i, j] = r_throat**2  # dθ²
                self.g[2, 2, i, j] = r_throat**2 * np.sin(self.theta[i])**2  # sin²θ dφ²
                self.g[3, 3, i, j] = r_throat**2 * np.sin(self.theta[i])**2 * np.sin(self.phi[j])**2  # sin²θ sin²φ dψ²
    
    def compute_ricci_tensor(self):
        """リッチテンソルとスカラー曲率を計算"""
        # 数値微分のためのステップサイズ
        dtheta = np.pi / (self.n_theta - 1)
        dphi = 2 * np.pi / (self.n_phi - 1)
        
        # クリストッフェル記号の計算（省略 - 実際には数値微分で計算する）
        # ここでは単純化のため、直接リッチテンソルを解析解から計算
        
        # 三次元球面の場合、リッチテンソルは計量に比例
        for i in range(self.n_theta):
            for j in range(self.n_phi):
                # リッチテンソルの球面成分（3次元球面のリッチテンソルは R_ab = 2 g_ab）
                self.Ric[1, 1, i, j] = 2 * self.g[1, 1, i, j]
                self.Ric[2, 2, i, j] = 2 * self.g[2, 2, i, j]
                self.Ric[3, 3, i, j] = 2 * self.g[3, 3, i, j]
                
                # スカラー曲率（3次元球面のスカラー曲率は R = 6/r²）
                r_throat = self.throat_radius * (1.0 + 0.1 * np.sin(self.theta[i]) * np.cos(self.phi[j]))
                self.R[i, j] = 6.0 / (r_throat**2)
        
        # 量子補正を適用
        self.R = self.R * (1.0 + self.quantum_factor * self.quantum_correction)
    
    def compute_einstein_tensor(self):
        """アインシュタインテンソルを計算"""
        # G_μν = R_μν - 1/2 g_μν R
        for i in range(self.n_theta):
            for j in range(self.n_phi):
                for mu in range(4):
                    for nu in range(4):
                        self.G[mu, nu, i, j] = self.Ric[mu, nu, i, j] - 0.5 * self.g[mu, nu, i, j] * self.R[i, j]
    
    def compute_entropy(self):
        """量子エンタングルメントエントロピーを計算"""
        # リーマン予想に基づくエントロピー計算
        c_fluid = 3.0  # 流体定数
        
        for i in range(self.n_theta):
            for j in range(self.n_phi):
                # 曲率からエントロピーを計算 (Bekenstein-Hawkingタイプの関係)
                area = self.throat_radius**2 * np.sin(self.theta[i])  # 対応する面積要素
                self.entropy[i, j] = (c_fluid / 4.0) * area * (1.0 + self.quantum_correction[i, j])
    
    def time_evolution(self, steps=100):
        """アインシュタイン方程式に従ってワームホールを時間発展させる"""
        # 時間ステップ
        dt = self.t_max / steps
        
        # 結果を保存する配列
        self.radius_history = np.zeros(steps)
        self.curvature_history = np.zeros(steps)
        self.entropy_history = np.zeros(steps)
        
        # 喉の半径の時間発展式（単純化したモデル）
        def throat_dynamics(t, r):
            # ワームホールのダイナミクスを記述する微分方程式
            # dr/dt = f(r) - 量子補正項
            
            # 古典項: アインシュタイン方程式から導かれる項
            classical_term = -0.1 * (r - self.throat_radius)  # 平衡点に戻る傾向
            
            # 量子項: リーマンゼータ関数の零点に基づく量子効果
            quantum_term = 0
            for i, zero in enumerate(riemann_zeros[:5]):
                quantum_term += 0.01 * np.sin(zero * t) / zero
            
            return classical_term + self.quantum_factor * quantum_term
        
        # 時間発展
        t_span = (0, self.t_max)
        t_eval = np.linspace(0, self.t_max, steps)
        
        # 初期条件
        r0 = [self.throat_radius]
        
        # 微分方程式を解く
        sol = solve_ivp(throat_dynamics, t_span, r0, t_eval=t_eval, method='RK45')
        
        # 結果を保存
        self.time_points = sol.t
        self.radius_history = sol.y[0]
        
        # 各時点での曲率とエントロピーを計算
        for i, r in enumerate(self.radius_history):
            # 平均曲率
            self.curvature_history[i] = 6.0 / (r**2)
            
            # 平均エントロピー
            self.entropy_history[i] = (3.0 / 4.0) * 4 * np.pi * r**2  # S = (c/4) * Area
    
    def analyze_throat_topology(self):
        """ワームホールの喉のトポロジーを解析"""
        # トポロジカル指数の計算（三次元球面のオイラー標数は2）
        euler_characteristic = 2
        
        # ベッチ数の計算（三次元球面のベッチ数は b_0=1, b_1=0, b_2=0, b_3=1）
        betti_numbers = [1, 0, 0, 1]
        
        # トーラスとの比較（情報トポロジー）
        information_genus = 0  # 球面のトポロジーはジーナス0
        
        # トポロジー解析結果を表示
        print(f"三次元球面と同相なワームホールの喉のトポロジー解析:")
        print(f"オイラー標数: {euler_characteristic}")
        print(f"ベッチ数: {betti_numbers}")
        print(f"情報ジーナス: {information_genus}")
        
        # 量子エンタングルメント構造
        print("\n量子エンタングルメント構造:")
        
        # リーマンゼータ関数のゼロ点から導かれるエンタングルメントスペクトル
        entanglement_eigenvalues = 1.0 / (1.0 + riemann_zeros[:5]**2)
        print(f"エンタングルメント固有値: {entanglement_eigenvalues}")
        
        # エンタングルメントエントロピー
        S_ent = -np.sum(entanglement_eigenvalues * np.log(entanglement_eigenvalues))
        print(f"エンタングルメントエントロピー: {S_ent}")
        
        # ミューチュアルインフォメーション（両側のワームホール間）
        I_mutual = 2 * S_ent
        print(f"ミューチュアルインフォメーション: {I_mutual}")
    
    def analyze_computational_complexity(self):
        """計算複雑性とワームホール構造の関係を解析"""
        # 計算複雑性の時間発展
        complexity = np.zeros_like(self.time_points)
        
        # リーマン予想に基づく複雑性-体積関係
        for i, r in enumerate(self.radius_history):
            # 複雑性 ∝ 体積
            volume = 2 * np.pi**2 * r**3  # 三次元球面の体積
            complexity[i] = volume / (self.throat_radius**3)  # 相対的な複雑性
        
        # 複雑性の時間変化をプロット
        plt.figure(figsize=(10, 6))
        plt.plot(self.time_points, complexity, 'b-', linewidth=2)
        plt.grid(True)
        plt.xlabel('時間')
        plt.ylabel('相対的計算複雑性')
        plt.title('ワームホールの喉の計算複雑性の時間発展')
        plt.savefig('wormhole_complexity.png')
        plt.show()
        
        # 複雑性と曲率の関係
        plt.figure(figsize=(10, 6))
        plt.scatter(self.curvature_history, complexity, c=self.time_points, cmap='viridis')
        plt.colorbar(label='時間')
        plt.grid(True)
        plt.xlabel('平均曲率')
        plt.ylabel('相対的計算複雑性')
        plt.title('ワームホールの曲率と計算複雑性の関係')
        plt.savefig('wormhole_complexity_curvature.png')
        plt.show()
        
        # リーマンゼータ関数のゼロ点との関連
        print("\n計算複雑性とリーマンゼータ関数の関係:")
        
        # 複雑性の成長率とリーマンゼータ関数の第一ゼロ点
        complexity_growth_rate = np.mean(np.diff(complexity) / np.diff(self.time_points))
        relation_to_first_zero = complexity_growth_rate * riemann_zeros[0]
        print(f"複雑性成長率: {complexity_growth_rate:.4f}")
        print(f"第一リーマンゼロ点との積: {relation_to_first_zero:.4f}")
        
        # 予測される臨界時間
        if complexity_growth_rate > 0:
            critical_time = riemann_zeros[0] / complexity_growth_rate
            print(f"予測される臨界時間: {critical_time:.4f}")
    
    def visualize_throat(self):
        """ワームホールの喉を可視化"""
        # 三次元球面を三次元に埋め込んで可視化
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # S³を立体射影してR³に埋め込む
        u = np.linspace(0, np.pi, 30)
        v = np.linspace(0, 2*np.pi, 30)
        
        U, V = np.meshgrid(u, v)
        
        # 立体射影（三次元球面から三次元ユークリッド空間へ）
        denominator = 1.0 + np.cos(U)
        X = self.throat_radius * np.sin(U) * np.cos(V) / denominator
        Y = self.throat_radius * np.sin(U) * np.sin(V) / denominator
        Z = self.throat_radius * np.sin(U) / denominator
        
        # 曲率で色付け
        curvature = 6.0 / (self.throat_radius**2) * (1.0 + 0.1 * np.sin(U) * np.cos(V))
        
        # 曲面描画
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', 
                              facecolors=plt.cm.viridis(curvature/np.max(curvature)),
                              linewidth=0, antialiased=True, alpha=0.8)
        
        # 軸ラベル
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('三次元球面と同相なワームホールの喉の構造')
        
        # カラーバー
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='相対的曲率')
        
        plt.savefig('wormhole_throat_visualization.png')
        plt.show()
    
    def visualize_time_evolution(self):
        """ワームホールの時間発展を可視化"""
        plt.figure(figsize=(12, 8))
        
        # 喉の半径の時間変化
        plt.subplot(2, 2, 1)
        plt.plot(self.time_points, self.radius_history, 'b-', linewidth=2)
        plt.grid(True)
        plt.xlabel('時間')
        plt.ylabel('喉の半径')
        plt.title('ワームホールの喉の半径の時間発展')
        
        # 曲率の時間変化
        plt.subplot(2, 2, 2)
        plt.plot(self.time_points, self.curvature_history, 'r-', linewidth=2)
        plt.grid(True)
        plt.xlabel('時間')
        plt.ylabel('平均曲率')
        plt.title('ワームホールの曲率の時間発展')
        
        # エントロピーの時間変化
        plt.subplot(2, 2, 3)
        plt.plot(self.time_points, self.entropy_history, 'g-', linewidth=2)
        plt.grid(True)
        plt.xlabel('時間')
        plt.ylabel('エンタングルメントエントロピー')
        plt.title('ワームホールのエントロピーの時間発展')
        
        # 曲率とエントロピーの関係
        plt.subplot(2, 2, 4)
        plt.scatter(self.radius_history, self.entropy_history, c=self.time_points, cmap='viridis')
        plt.colorbar(label='時間')
        plt.grid(True)
        plt.xlabel('喉の半径')
        plt.ylabel('エントロピー')
        plt.title('半径とエントロピーの関係')
        
        plt.tight_layout()
        plt.savefig('wormhole_evolution.png')
        plt.show()
    
    def create_animation(self):
        """ワームホールの喉の時間発展アニメーションを作成"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 球面のグリッド
        u = np.linspace(0, np.pi, 30)
        v = np.linspace(0, 2*np.pi, 30)
        U, V = np.meshgrid(u, v)
        
        # アニメーションのフレーム更新関数
        def update(frame):
            ax.clear()
            
            # 現在の半径
            r = self.radius_history[frame]
            
            # 立体射影
            denominator = 1.0 + np.cos(U)
            X = r * np.sin(U) * np.cos(V) / denominator
            Y = r * np.sin(U) * np.sin(V) / denominator
            Z = r * np.sin(U) / denominator
            
            # 曲率で色付け
            curvature = 6.0 / (r**2) * (1.0 + 0.1 * np.sin(U) * np.cos(V))
            
            # 曲面描画
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', 
                                  facecolors=plt.cm.viridis(curvature/np.max(curvature)),
                                  linewidth=0, antialiased=True, alpha=0.8)
            
            # 軸ラベル
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'ワームホールの喉の時間発展 (t={self.time_points[frame]:.2f})')
            
            # 軸の範囲を固定
            ax_limit = max(np.max(self.radius_history) * 2, 2.0)
            ax.set_xlim([-ax_limit, ax_limit])
            ax.set_ylim([-ax_limit, ax_limit])
            ax.set_zlim([-ax_limit, ax_limit])
            
            return surf,
        
        # アニメーション作成
        ani = animation.FuncAnimation(fig, update, frames=len(self.time_points), 
                                      interval=100, blit=False)
        
        # アニメーションを保存
        ani.save('wormhole_evolution.gif', writer='pillow', fps=10)
        
        plt.close()
        
        print("アニメーションを 'wormhole_evolution.gif' に保存しました。")

# 実行例
if __name__ == "__main__":
    # ワームホールシミュレーターを初期化
    wormhole = EinsteinWormhole(n_modes=10, t_max=10.0, throat_radius=1.0, quantum_factor=0.2)
    
    # リッチテンソルとアインシュタインテンソルを計算
    wormhole.compute_ricci_tensor()
    wormhole.compute_einstein_tensor()
    
    # エントロピーを計算
    wormhole.compute_entropy()
    
    # ワームホールの時間発展をシミュレート
    print("ワームホールの時間発展をシミュレートしています...")
    wormhole.time_evolution(steps=200)
    
    # 解析
    wormhole.analyze_throat_topology()
    wormhole.analyze_computational_complexity()
    
    # 可視化
    wormhole.visualize_throat()
    wormhole.visualize_time_evolution()
    
    # アニメーション作成
    wormhole.create_animation() 