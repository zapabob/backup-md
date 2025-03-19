import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
import time

# リーマンゼータ関数の非自明なゼロ点の虚部（最初の5つ）
riemann_zeros = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062]

def solve_navier_stokes_quantum(N=64, L=2*np.pi, nu=0.01, dt=0.01, T=1.0, quantum_factor=0.1):
    """
    非可換KAT表現と量子統計力学的アプローチを用いたナビエストークス方程式の数値解法
    
    引数:
        N: グリッド解像度
        L: 計算領域のサイズ
        nu: 動粘性係数
        dt: 時間ステップ
        T: シミュレーション終了時間
        quantum_factor: 量子効果の強さ
    
    戻り値:
        u, v: 最終的な速度場
        omega: 最終的な渦度場
    """
    # 開始時間
    start_time = time.time()
    
    # グリッド設定
    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, y)
    
    # 波数設定
    kx = 2 * np.pi / L * np.fft.fftfreq(N) * N
    ky = 2 * np.pi / L * np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2
    K2[0, 0] = 1.0  # ゼロ除算を避けるため
    
    # 量子補正係数の計算
    quantum_correction = np.zeros((N, N), dtype=complex)
    for i, zero in enumerate(riemann_zeros):
        phase = np.exp(2j * np.pi * i / len(riemann_zeros))
        quantum_correction += phase * np.exp(-((KX**2 + KY**2) / zero**2))
    
    # 正規化
    quantum_correction /= np.max(np.abs(quantum_correction))
    
    # 初期条件: テイラーグリーン渦
    u = np.sin(X) * np.cos(Y)
    v = -np.cos(X) * np.sin(Y)
    omega = np.gradient(v, x, axis=1) - np.gradient(u, y, axis=0)
    
    # 時間発展
    t = 0.0
    while t < T:
        # 渦度のフーリエ変換
        omega_hat = fft2(omega)
        
        # 非線形項
        u_omega = u * omega
        v_omega = v * omega
        nl_hat = fft2(u_omega + v_omega)
        
        # 量子統計力学的補正の適用
        quantum_nl_hat = nl_hat * (1.0 + quantum_factor * quantum_correction)
        
        # 時間発展
        factor = np.exp(-nu * K2 * dt)
        omega_hat_new = factor * (omega_hat - dt * quantum_nl_hat)
        
        # 空間領域に戻す
        omega = np.real(ifft2(omega_hat_new))
        
        # 速度場の更新
        psi_hat = -omega_hat_new / K2
        psi_hat[0, 0] = 0
        
        u_hat = 1j * KY * psi_hat
        v_hat = -1j * KX * psi_hat
        
        u = np.real(ifft2(u_hat))
        v = np.real(ifft2(v_hat))
        
        t += dt
        
        if t % 0.2 < dt:  # 0.2秒ごとに進捗表示
            print(f"t = {t:.2f} / {T:.2f}")
    
    # 計算時間
    elapsed_time = time.time() - start_time
    print(f"計算時間: {elapsed_time:.2f}秒")
    
    # 結果のプロット
    plt.figure(figsize=(12, 5))
    
    # 渦度
    plt.subplot(1, 2, 1)
    plt.title(f"渦度 (t={t:.2f})")
    plt.contourf(X, Y, omega, cmap="RdBu_r", levels=20)
    plt.colorbar()
    
    # 速度場
    plt.subplot(1, 2, 2)
    plt.title(f"速度場 (t={t:.2f})")
    speed = np.sqrt(u**2 + v**2)
    plt.contourf(X, Y, speed, cmap="viridis", levels=20)
    skip = max(1, N // 20)
    plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
              u[::skip, ::skip], v[::skip, ::skip], 
              color="white", scale=50)
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig("quantum_ns_result.png")
    print(f"結果を quantum_ns_result.png に保存しました")
    plt.close()  # ウィンドウを表示せずに閉じる
    
    return u, v, omega

if __name__ == "__main__":
    print("非可換KAT表現と量子統計力学的アプローチを用いたナビエストークス方程式のシミュレーション開始")
    
    # シミュレーションパラメータ
    N = 64  # グリッド解像度（小さめに設定）
    L = 2 * np.pi  # 計算領域サイズ
    nu = 0.01  # 動粘性係数
    dt = 0.01  # 時間ステップ
    T = 1.0  # シミュレーション時間
    quantum_factor = 0.1  # 量子効果の強さ
    
    # シミュレーション実行
    u, v, omega = solve_navier_stokes_quantum(N=N, L=L, nu=nu, dt=dt, T=T, quantum_factor=quantum_factor)
    
    print("シミュレーション完了")
    print(f"最大速度: {np.max(np.sqrt(u**2 + v**2)):.4f}")
    print(f"最大渦度: {np.max(np.abs(omega)):.4f}") 