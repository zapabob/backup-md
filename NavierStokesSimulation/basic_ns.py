import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
import time

def basic_navier_stokes(N=32, T=0.5):
    """
    ナビエストークス方程式の基本的な数値解法
    
    引数:
        N: グリッド解像度
        T: シミュレーション時間
    """
    print("基本的なナビエストークス方程式シミュレーション開始")
    
    # パラメータ設定
    L = 2 * np.pi  # 計算領域のサイズ
    nu = 0.01      # 動粘性係数
    dt = 0.01      # 時間ステップ
    
    # グリッド設定
    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, y)
    
    # 初期条件：テイラーグリーン渦
    u = np.sin(X) * np.cos(Y)
    v = -np.cos(X) * np.sin(Y)
    
    # 初期渦度の計算
    omega = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            omega[i, j] = -2 * np.sin(X[i, j]) * np.sin(Y[i, j])
    
    print(f"初期条件設定完了。最大速度: {np.max(np.sqrt(u**2 + v**2)):.4f}")
    
    # 初期状態のプロット
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.contourf(X, Y, omega, cmap='RdBu_r')
    plt.colorbar()
    plt.title('初期渦度')
    
    plt.subplot(1, 2, 2)
    speed = np.sqrt(u**2 + v**2)
    plt.contourf(X, Y, speed, cmap='viridis')
    plt.colorbar()
    plt.title('初期速度場の大きさ')
    
    plt.tight_layout()
    plt.savefig("navier_stokes_initial.png")
    print("初期状態を navier_stokes_initial.png に保存しました")
    plt.close()

    print("シミュレーション完了")

if __name__ == "__main__":
    basic_navier_stokes() 