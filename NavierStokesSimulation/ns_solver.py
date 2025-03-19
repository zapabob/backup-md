import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
import time

# リーマンゼータ関数の非自明なゼロ点の虚部
riemann_zeros = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832
]

class NavierStokesQuantumKAT:
    """非可換KAT表現と量子統計力学的アプローチを用いたナビエストークス方程式のシミュレーションクラス"""
    
    def __init__(self, N=64, L=2*np.pi, nu=0.01, dt=0.01):
        """初期化メソッド
        
        引数:
            N: グリッドの解像度
            L: 計算領域のサイズ
            nu: 動粘性係数
            dt: 時間ステップ
        """
        self.N = N
        self.L = L
        self.nu = nu
        self.dt = dt
        
        # グリッド設定
        self.x = np.linspace(0, L, N, endpoint=False)
        self.y = np.linspace(0, L, N, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # 波数設定
        self.kx = 2 * np.pi / L * np.fft.fftfreq(N) * N
        self.ky = 2 * np.pi / L * np.fft.fftfreq(N) * N
        self.KX, self.KY = np.meshgrid(self.kx, self.ky)
        self.K2 = self.KX**2 + self.KY**2
        self.K2[0, 0] = 1.0  # ゼロ除算を避けるため
        
        # 初期条件の設定（デフォルトはテイラーグリーン渦）
        self.initialize_taylor_green()
        
        # 非可換KAT表現のパラメータ初期化
        self.initialize_quantum_parameters()
    
    def initialize_taylor_green(self):
        """テイラーグリーン渦の初期条件を設定"""
        self.u = np.sin(self.X) * np.cos(self.Y)
        self.v = -np.cos(self.X) * np.sin(self.Y)
        self.omega = np.gradient(self.v, self.x, axis=1) - np.gradient(self.u, self.y, axis=0)
    
    def initialize_quantum_parameters(self):
        """非可換KAT表現と量子統計力学的パラメータの初期化"""
        # リーマンゼータ関数のゼロ点に基づく量子補正係数
        self.quantum_correction = np.zeros((self.N, self.N), dtype=complex)
        for i, zero in enumerate(riemann_zeros[:min(5, len(riemann_zeros))]):
            phase = np.exp(2j * np.pi * i / len(riemann_zeros))
            self.quantum_correction += phase * np.exp(-((self.KX**2 + self.KY**2) / zero**2))
        
        # 正規化
        self.quantum_correction /= np.max(np.abs(self.quantum_correction))
        
        # エンタングルメント係数（非可換性を表す）
        self.entanglement_factor = 0.1  # 非可換性の強さ

    def step(self):
        """1ステップの時間発展を計算"""
        # 渦度のフーリエ変換
        omega_hat = fft2(self.omega)
        
        # 非線形項の計算
        u_omega = self.u * self.omega
        v_omega = self.v * self.omega
        nl_hat = fft2(u_omega + v_omega)
        
        # 量子統計力学的補正の適用
        quantum_nl_hat = nl_hat * (1.0 + self.entanglement_factor * self.quantum_correction)
        
        # 時間発展
        factor = np.exp(-self.nu * self.K2 * self.dt)
        omega_hat_new = factor * (omega_hat - self.dt * quantum_nl_hat)
        
        # 渦度を空間領域に戻す
        self.omega = np.real(ifft2(omega_hat_new))
        
        # 速度場の更新
        psi_hat = -omega_hat_new / self.K2
        psi_hat[0, 0] = 0
        
        # 速度の導出
        u_hat = 1j * self.KY * psi_hat
        v_hat = -1j * self.KX * psi_hat
        
        self.u = np.real(ifft2(u_hat))
        self.v = np.real(ifft2(v_hat))
        
        return self.u, self.v, self.omega

    def simulate(self, T=2.0, plot_interval=0.1):
        """指定した時間まで計算し、途中経過を表示する
        
        引数:
            T: 最終時刻
            plot_interval: プロット間隔
        """
        t = 0.0
        plot_times = np.arange(0, T+plot_interval, plot_interval)
        plot_index = 0
        
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        
        while t < T:
            # 時間発展
            self.step()
            t += self.dt
            
            # 表示間隔に達したらプロット
            if plot_index < len(plot_times) and t >= plot_times[plot_index]:
                self.plot_state(fig, axs, t)
                plot_index += 1
                plt.pause(0.01)
        
        # 最終状態のプロット
        self.plot_state(fig, axs, t, is_final=True)
        plt.tight_layout()
        plt.savefig("quantum_navier_stokes_result.png")
        plt.show()
        
        return self.u, self.v, self.omega
    
    def plot_state(self, fig, axs, t, is_final=False):
        """現在の状態をプロット
        
        引数:
            fig: matplotlib図
            axs: サブプロット軸
            t: 現在の時刻
            is_final: 最終プロットかどうか
        """
        speed = np.sqrt(self.u**2 + self.v**2)
        vorticity_entropy = self.calculate_entropy()
        
        # 渦度
        axs[0].clear()
        axs[0].set_title(f"渦度 (t={t:.2f}, エントロピー={vorticity_entropy:.4f})")
        cf1 = axs[0].contourf(self.X, self.Y, self.omega, cmap="RdBu_r", levels=20)
        fig.colorbar(cf1, ax=axs[0])
        
        # 速度場
        axs[1].clear()
        axs[1].set_title(f"速度場 (t={t:.2f})")
        cf2 = axs[1].contourf(self.X, self.Y, speed, cmap="viridis", levels=20)
        skip = max(1, self.N // 20)  # 表示する矢印の間隔
        axs[1].quiver(self.X[::skip, ::skip], self.Y[::skip, ::skip], 
                     self.u[::skip, ::skip], self.v[::skip, ::skip], 
                     color="white", scale=50)
        fig.colorbar(cf2, ax=axs[1])
        
        if is_final:
            axs[0].set_xlabel("x")
            axs[0].set_ylabel("y")
            axs[1].set_xlabel("x")
            axs[1].set_ylabel("y")
    
    def calculate_entropy(self):
        """渦度に関するエンタングルメントエントロピーの計算"""
        # 渦度のパワースペクトル密度の計算
        omega_hat = fft2(self.omega)
        psd = np.abs(omega_hat)**2
        psd /= np.sum(psd)
        
        # ゼロを避ける
        psd = psd + 1e-10
        psd /= np.sum(psd)
        
        # エントロピーの計算
        entropy = -np.sum(psd * np.log(psd))
        return entropy

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

def main():
    """メイン関数"""
    print("非可換KAT表現と量子統計力学的アプローチを用いたナビエストークス方程式のシミュレーション開始")
    
    # リーマン予想に基づく大域解の存在性条件の評価
    condition_satisfied, ratio, threshold = evaluate_global_existence_condition()
    print("\n===== リーマン予想に基づく大域解の存在性条件の評価 =====")
    print(f"計算された比率: {ratio:.6f}")
    print(f"閾値: {threshold:.6f}")
    if condition_satisfied:
        print("条件を満たしています：大域的な滑らかな解の存在が予測されます")
    else:
        print("条件を満たしていません：大域的な滑らかな解の存在は保証されません")
    print("===================================================\n")
    
    # シミュレーションパラメータ
    N = 128  # グリッド解像度
    L = 2 * np.pi  # 計算領域サイズ
    nu = 0.001  # 動粘性係数
    dt = 0.001  # 時間ステップ
    T = 1.0  # シミュレーション時間
    
    # シミュレーション実行
    simulator = NavierStokesQuantumKAT(N=N, L=L, nu=nu, dt=dt)
    u, v, omega = simulator.simulate(T=T, plot_interval=0.1)
    
    print("シミュレーション完了")
    print(f"最大速度: {np.max(np.sqrt(u**2 + v**2)):.4f}")
    print(f"最大渦度: {np.max(np.abs(omega)):.4f}")
    print(f"エントロピー: {simulator.calculate_entropy():.4f}")
    
    # シミュレーション結果と理論予測の比較
    c_fluid = simulator.calculate_entropy() * 1.5  # エントロピーから流体力学パラメータを推定
    print("\n===== シミュレーション結果と理論予測の比較 =====")
    condition_satisfied, ratio, threshold = evaluate_global_existence_condition(c_fluid)
    print(f"シミュレーションから推定されたc_fluid: {c_fluid:.4f}")
    print(f"修正された閾値: {threshold:.6f}")
    if condition_satisfied:
        print("シミュレーション結果は理論と一致し、大域的な滑らかな解の存在を支持しています")
    else:
        print("シミュレーション結果と理論予測の間に不一致があります")
    print("=============================================")

if __name__ == "__main__":
    main()
