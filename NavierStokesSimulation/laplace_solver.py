import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class LaplaceSolver:
    """ラプラス方程式 ∇²φ = 0 の数値解法クラス"""
    
    def __init__(self, nx=50, ny=50, max_iter=1000, tolerance=1e-6):
        """
        初期化メソッド
        
        引数:
            nx: x方向の格子点数
            ny: y方向の格子点数
            max_iter: 最大反復回数
            tolerance: 収束判定の許容誤差
        """
        self.nx = nx
        self.ny = ny
        self.max_iter = max_iter
        self.tolerance = tolerance
        
        # 格子点の座標
        self.x = np.linspace(0, 1, nx)
        self.y = np.linspace(0, 1, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # ポテンシャル場の初期化
        self.phi = np.zeros((ny, nx))
        
        # 境界条件の設定（デフォルトは全て0）
        self.boundary_conditions = {
            'top': np.zeros(nx),
            'bottom': np.zeros(nx),
            'left': np.zeros(ny),
            'right': np.zeros(ny)
        }
    
    def set_boundary_conditions(self, top=None, bottom=None, left=None, right=None):
        """
        境界条件を設定
        
        引数:
            top: 上側境界の値
            bottom: 下側境界の値
            left: 左側境界の値
            right: 右側境界の値
        """
        if top is not None:
            self.boundary_conditions['top'] = np.array(top)
        if bottom is not None:
            self.boundary_conditions['bottom'] = np.array(bottom)
        if left is not None:
            self.boundary_conditions['left'] = np.array(left)
        if right is not None:
            self.boundary_conditions['right'] = np.array(right)
        
        # 境界条件を初期場に反映
        self.apply_boundary_conditions()
    
    def apply_boundary_conditions(self):
        """境界条件を場に適用"""
        # 上下左右の境界を設定
        self.phi[0, :] = self.boundary_conditions['top']
        self.phi[-1, :] = self.boundary_conditions['bottom']
        self.phi[:, 0] = self.boundary_conditions['left']
        self.phi[:, -1] = self.boundary_conditions['right']
    
    def solve_jacobi(self):
        """ヤコビ法でラプラス方程式を解く"""
        phi_old = self.phi.copy()
        
        for iteration in range(self.max_iter):
            # 境界条件を再適用
            self.apply_boundary_conditions()
            
            # 内部点の更新（ヤコビ法）
            phi_new = phi_old.copy()
            phi_new[1:-1, 1:-1] = 0.25 * (
                phi_old[1:-1, 0:-2] +  # 左隣
                phi_old[1:-1, 2:]   +  # 右隣
                phi_old[0:-2, 1:-1] +  # 上隣
                phi_old[2:, 1:-1]      # 下隣
            )
            
            # 収束判定
            error = np.max(np.abs(phi_new - phi_old))
            phi_old = phi_new.copy()
            
            if iteration % 100 == 0:
                print(f"反復: {iteration}, 誤差: {error:.6e}")
            
            if error < self.tolerance:
                print(f"収束しました（反復: {iteration}, 誤差: {error:.6e}）")
                break
        
        self.phi = phi_old
        
        if iteration == self.max_iter - 1:
            print(f"最大反復回数に達しました。最終誤差: {error:.6e}")
        
        return self.phi
    
    def solve_sor(self, omega=1.5):
        """SOR法（逐次過緩和法）でラプラス方程式を解く"""
        for iteration in range(self.max_iter):
            # 境界条件を再適用
            self.apply_boundary_conditions()
            
            error = 0.0
            
            # 内部点を更新（SOR法）- 赤黒順序法
            for i in range(1, self.ny-1):
                for j in range(1, self.nx-1):
                    if (i + j) % 2 == 0:  # 赤点
                        phi_old = self.phi[i, j]
                        self.phi[i, j] = (1.0 - omega) * phi_old + omega * 0.25 * (
                            self.phi[i, j-1] + self.phi[i, j+1] +
                            self.phi[i-1, j] + self.phi[i+1, j]
                        )
                        error = max(error, abs(self.phi[i, j] - phi_old))
            
            for i in range(1, self.ny-1):
                for j in range(1, self.nx-1):
                    if (i + j) % 2 == 1:  # 黒点
                        phi_old = self.phi[i, j]
                        self.phi[i, j] = (1.0 - omega) * phi_old + omega * 0.25 * (
                            self.phi[i, j-1] + self.phi[i, j+1] +
                            self.phi[i-1, j] + self.phi[i+1, j]
                        )
                        error = max(error, abs(self.phi[i, j] - phi_old))
            
            if iteration % 100 == 0:
                print(f"反復: {iteration}, 誤差: {error:.6e}")
            
            if error < self.tolerance:
                print(f"収束しました（反復: {iteration}, 誤差: {error:.6e}）")
                break
        
        if iteration == self.max_iter - 1:
            print(f"最大反復回数に達しました。最終誤差: {error:.6e}")
        
        return self.phi
    
    def solve_fft(self):
        """高速フーリエ変換（FFT）を用いてラプラス方程式を解く"""
        from numpy.fft import fft2, ifft2
        
        # 境界条件を適用
        self.apply_boundary_conditions()
        
        # 内部点を0に設定（ディリクレ境界条件用）
        phi_interior = self.phi.copy()
        phi_interior[1:-1, 1:-1] = 0
        
        # 残差を計算
        residual = np.zeros_like(self.phi)
        residual[0, :] = self.phi[0, :]
        residual[-1, :] = self.phi[-1, :]
        residual[:, 0] = self.phi[:, 0]
        residual[:, -1] = self.phi[:, -1]
        
        # ラプラシアン作用素のフーリエ表現
        kx = np.fft.fftfreq(self.nx) * 2.0 * np.pi * self.nx
        ky = np.fft.fftfreq(self.ny) * 2.0 * np.pi * self.ny
        kx2, ky2 = np.meshgrid(kx**2, ky**2)
        laplacian_k = -(kx2 + ky2)
        laplacian_k[0, 0] = 1.0  # ゼロ除算を避ける（定数成分は境界条件で決まる）
        
        # FFTで解く
        residual_hat = fft2(residual)
        phi_hat = residual_hat / laplacian_k
        phi_solution = np.real(ifft2(phi_hat))
        
        # 境界条件を再適用
        phi_solution[0, :] = self.phi[0, :]
        phi_solution[-1, :] = self.phi[-1, :]
        phi_solution[:, 0] = self.phi[:, 0]
        phi_solution[:, -1] = self.phi[:, -1]
        
        self.phi = phi_solution
        return self.phi
    
    def plot_solution(self, title="ラプラス方程式の解"):
        """解をプロット"""
        fig = plt.figure(figsize=(12, 5))
        
        # 3Dサーフェスプロット
        ax1 = fig.add_subplot(121, projection='3d')
        surf = ax1.plot_surface(self.X, self.Y, self.phi, cmap=cm.viridis,
                              linewidth=0, antialiased=False)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('$\phi$')
        ax1.set_title(f'3D表示: {title}')
        fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
        
        # 2D等高線プロット
        ax2 = fig.add_subplot(122)
        contour = ax2.contourf(self.X, self.Y, self.phi, 20, cmap=cm.viridis)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title(f'等高線: {title}')
        fig.colorbar(contour, ax=ax2)
        
        plt.tight_layout()
        plt.savefig('laplace_solution.png')
        plt.show()

# 実行例：静電ポテンシャルの計算
if __name__ == "__main__":
    # 100x100の格子でソルバーを初期化
    solver = LaplaceSolver(nx=100, ny=100, max_iter=5000, tolerance=1e-6)
    
    # 境界条件：上部に正の電位、底部に負の電位を設定
    top_values = np.ones(100)  # 上部: φ = 1
    bottom_values = -np.ones(100)  # 底部: φ = -1
    
    # 左右は周期的になるように正弦波で変化
    x = np.linspace(0, 1, 100)
    left_values = np.sin(2 * np.pi * x)
    right_values = -np.sin(2 * np.pi * x)
    
    # 境界条件を設定
    solver.set_boundary_conditions(
        top=top_values,
        bottom=bottom_values,
        left=left_values,
        right=right_values
    )
    
    # ラプラス方程式を解く（SOR法）
    print("SOR法で解いています...")
    phi_sor = solver.solve_sor(omega=1.5)
    
    # 解をプロット
    solver.plot_solution("SOR法による解")
    
    # FFT法でも解く
    print("\nFFT法で解いています...")
    solver = LaplaceSolver(nx=100, ny=100)
    solver.set_boundary_conditions(
        top=top_values,
        bottom=bottom_values,
        left=left_values,
        right=right_values
    )
    phi_fft = solver.solve_fft()
    
    # 解をプロット
    solver.plot_solution("FFT法による解") 