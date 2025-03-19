#python 3.12.9
#pytorch 2.3.1
#cuda 12.1
#RTX3080
#utf-8
# 行列の対数を計算（PyTorchで利用可能な方法を使用）
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# GPU設定！RTX3080対応！
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 
def generate_gue_matrix(N):
    A = torch.randn(N, N, dtype=torch.cfloat, device=device)
    H = (A + A.conj().T) / 2  #     
    return H

#   
def compute_fisher_matrix(H):
    # 密度行列の計算
    rho = torch.matrix_exp(-H)
    rho /= torch.trace(rho)
    
    # 行列を実数化して簡略計算
    rho_real = rho.real
    F = rho_real @ rho_real
    F = (F + F.T) / 2  # 対称化
    return F

# 固有値計算関数    
def compute_eigenvalues(F):
    eigvals = torch.linalg.eigvalsh(F)
    return eigvals.cpu().numpy()


N = 3000

            
num_matrices = 100
all_eigvals_spacing = []

for _ in tqdm(range(num_matrices), desc="行列の生成と固有値計算"):
    
    H = generate_gue_matrix(N).to(device)
    
    # Fisher行列を生成
    F = compute_fisher_matrix(H=H)
    
    # 固有値
    eigvals = compute_eigenvalues(F)
    
    # 固有値の間隔を正規化
    eigvals_spacing = np.diff(np.sort(eigvals))
    eigvals_spacing /= np.mean(eigvals_spacing)
    
    all_eigvals_spacing.extend(eigvals_spacing)

# 
plt.figure(figsize=(10,6))
plt.hist(all_eigvals_spacing, bins=50, density=True, alpha=0.7, color='purple')
plt.xlabel('Normalized Eigenvalue Spacing')
plt.ylabel('Density')
plt.title('Eigenvalue spacing distribution (PyTorch GPU)')
plt.grid(True)
plt.show()


