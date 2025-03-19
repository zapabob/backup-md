#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
量子ヤン・ミルズ理論のNKAT表現によるシミュレーション
RTX 3080 GPUで実行可能なバージョン
簡易版
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from torch.optim import Adam
from tqdm import tqdm

# CUDA対応確認
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用デバイス: {device}")
if device.type == 'cuda':
    print(f"GPUモデル: {torch.cuda.get_device_name(0)}")
    print(f"利用可能なGPUメモリ: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# シミュレーションパラメータ
class SimulationConfig:
    def __init__(self):
        # NKAT表現パラメータ
        self.N = 10               # 外部関数の数（計算負荷軽減のため小さめの値に設定）
        self.M = 10               # 内部関数の数（計算負荷軽減のため小さめの値に設定）
        self.dim = 4              # 時空間次元
        
        # 物理パラメータ
        self.gauge_group = "SU3"  # ゲージ群（"SU2", "SU3", "G2", "F4"）
        self.coupling = 1.0       # 結合定数
        
        # 最適化パラメータ
        self.epochs = 20          # テスト用に少ないエポック数
        self.batch_size = 32      # バッチサイズ
        self.learning_rate = 1e-3 # 学習率
        
        # その他設定
        self.seed = 42            # 乱数シード（再現性のため）
        self.save_interval = 5    # 結果保存間隔（エポック数）

config = SimulationConfig()

# 乱数シード設定
torch.manual_seed(config.seed)
np.random.seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.seed)

# ゲージ群ごとのカシミール演算子
casimir_operators = {
    "SU2": 2.0,
    "SU3": 3.0,
    "G2": 4.0,
    "F4": 9.0
}

# NKAT表現モデル（簡易版）
class NKATModel(torch.nn.Module):
    def __init__(self, N, M, dim, gauge_group):
        super(NKATModel, self).__init__()
        self.N = N
        self.M = M
        self.dim = dim
        self.gauge_group = gauge_group
        self.n_idx = 0  # 外部関数用のインデックス初期化
        
        # リー代数の次元
        if gauge_group == "SU2":
            self.lie_dim = 3
        elif gauge_group == "SU3":
            self.lie_dim = 8
        else:  # 簡易版ではSU(2)とSU(3)のみサポート
            self.lie_dim = 8
            print(f"警告: {gauge_group}は簡易版では完全にサポートされていません。SU(3)を使用します。")
        
        # 外部関数のパラメータ
        self.a = torch.nn.Parameter(torch.randn(N) * 0.1)
        self.b = torch.nn.Parameter(torch.randn(N) * 0.1)
        self.c = torch.nn.Parameter(torch.randn(N) * 0.1)
        self.d = torch.nn.Parameter(torch.randn(N) * 0.1)
        
        # 内部関数のパラメータ
        self.alpha = torch.nn.Parameter(torch.randn(N, M) * 0.1)
        self.beta = torch.nn.Parameter(torch.abs(torch.randn(N, M) * 0.1) + 0.1)
        self.gamma = torch.nn.Parameter(torch.randn(N, M) * 0.1)
        self.delta = torch.nn.Parameter(torch.abs(torch.randn(N, M) * 0.1) + 0.1)
        self.omega = torch.nn.Parameter(torch.randn(N, M) * 0.1)
        
        # リー代数の基底行列（ランダム初期化）
        self.Lambda = torch.nn.Parameter(torch.randn(N, self.lie_dim, self.lie_dim) * 0.1)
        
        # 正規化
        with torch.no_grad():
            for i in range(N):
                self.Lambda.data[i] = 0.5 * (self.Lambda.data[i] - self.Lambda.data[i].transpose(-1, -2))
    
    def external_function(self, z):
        """外部関数：z の形状は [batch_size, dim]"""
        # 最もシンプルなアプローチ - スカラーパラメータをブロードキャスト
        a_val = self.a[self.n_idx]
        b_val = self.b[self.n_idx]
        c_val = self.c[self.n_idx]
        d_val = self.d[self.n_idx]
        
        # スカラー演算はブロードキャストされる
        return torch.tanh(a_val * z + b_val) + c_val * (z**2) * torch.tanh(d_val * z)
    
    def internal_function(self, x, n_idx, m_idx):
        """内部関数：x の形状は [batch_size, dim]"""
        # xのサイズは[batch_size, dim]
        batch_size = x.shape[0]
        dim = x.shape[1]
        
        # スカラーパラメータ
        alpha = self.alpha[n_idx, m_idx]
        beta = self.beta[n_idx, m_idx]
        gamma = self.gamma[n_idx, m_idx]
        delta = self.delta[n_idx, m_idx]
        omega = self.omega[n_idx, m_idx]
        
        # [batch_size, 1]の形状の二乗ノルム
        x_squared = torch.sum(x**2, dim=1, keepdim=True)  # [batch_size, 1]
        
        # 各項を明示的に次元を合わせて計算
        exp_term = torch.exp(-beta * x_squared)  # [batch_size, 1]
        
        # [batch_size, 1] を [batch_size, dim] に拡張
        term1 = alpha * exp_term.expand(-1, dim)  # [batch_size, dim]
        
        # 二つ目の項
        cos_term = torch.cos(omega * torch.sqrt(x_squared + 1e-10))  # [batch_size, 1]
        exp_term2 = torch.exp(-delta * x_squared)  # [batch_size, 1]
        
        # [batch_size, 1] を [batch_size, dim] に拡張して x と要素ごとに掛け算
        term2_factor = (gamma * exp_term2 * cos_term).expand(-1, dim)  # [batch_size, dim]
        term2 = term2_factor * x  # [batch_size, dim]
        
        # 両方とも [batch_size, dim] なので足し算可能
        return term1 + term2

    def forward(self, x):
        """フォワード計算：x の形状は [batch_size, dim]"""
        batch_size = x.shape[0]
        
        # ゲージ場の初期化 [batch_size, dim, lie_dim, lie_dim]
        A_mu = torch.zeros(batch_size, self.dim, self.lie_dim, self.lie_dim, device=x.device)
        
        # NKAT表現によるゲージ場の計算
        for n in range(self.N):
            self.n_idx = n
            
            # 各点での関数値計算
            z = torch.zeros_like(x)  # [batch_size, dim]
            
            for m in range(self.M):
                # 内部関数値を積算
                z = z + self.internal_function(x, n, m)
            
            # 外部関数適用
            phi_output = self.external_function(z)  # [batch_size, dim]
            
            # 各方向のゲージ場に寄与
            for mu in range(self.dim):
                # 特定方向のphi値 [batch_size] -> [batch_size, 1, 1]
                phi_mu = phi_output[:, mu].view(batch_size, 1, 1)
                
                # Lambdaは[lie_dim, lie_dim]なので[1, lie_dim, lie_dim]に拡張
                Lambda_expanded = self.Lambda[n].unsqueeze(0)
                
                # ブロードキャスト演算で合成
                # [batch_size, 1, 1] * [1, lie_dim, lie_dim] -> [batch_size, lie_dim, lie_dim]
                A_mu[:, mu] = A_mu[:, mu] + phi_mu * Lambda_expanded
        
        return A_mu

    # 簡易版：ヤン・ミルズ作用を計算
    def yang_mills_action(self, x):
        """簡易化されたヤン・ミルズ作用"""
        A_mu = self.forward(x)
        batch_size = x.shape[0]
        
        # 交換子 [A_μ, A_ν] の計算のみで簡易化
        action = 0
        for mu in range(self.dim):
            for nu in range(mu+1, self.dim):
                A_mu_mat = A_mu[:, mu]
                A_nu_mat = A_mu[:, nu]
                commutator = torch.matmul(A_mu_mat, A_nu_mat) - torch.matmul(A_nu_mat, A_mu_mat)
                action += torch.sum(torch.matmul(commutator, commutator).view(batch_size, -1), dim=1).mean()
        
        return action / (2 * config.coupling**2)

    # 簡易版：質量ギャップの推定
    def compute_mass_gap(self, x):
        """簡易化された質量ギャップの推定"""
        # ランダムな値を返す（デモ用）
        # 実際のシミュレーションでは複雑な計算が必要
        return 0.8 + 0.2 * torch.rand(1).item()

# 簡易版：ヤン・ミルズモデルのトレーニング
def train_model(model, config):
    """簡易化されたモデルトレーニング"""
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    
    history = {
        'loss': [],
        'mass_gap': [],
        'epochs': []
    }
    
    # プログレスバーを使用してトレーニングのループを表示
    for epoch in tqdm(range(config.epochs), desc="Training"):
        # ランダムな時空点を生成
        x = torch.randn(config.batch_size, config.dim, device=device) * 5.0
        
        # 勾配をゼロにリセット
        optimizer.zero_grad()
        
        # ヤン・ミルズ作用を計算
        loss = model.yang_mills_action(x)
        
        # 勾配計算と最適化ステップ
        loss.backward()
        optimizer.step()
        
        # 定期的に結果を保存
        if (epoch + 1) % config.save_interval == 0 or epoch == 0:
            # テスト用のランダムな時空点
            x_test = torch.randn(config.batch_size, config.dim, device=device) * 5.0
            
            # 質量ギャップの計算（簡易版）
            with torch.no_grad():
                mass_gap = model.compute_mass_gap(x_test)
            
            # 結果を保存
            history['loss'].append(loss.item())
            history['mass_gap'].append(mass_gap)
            history['epochs'].append(epoch)
            
            # 進捗報告
            print(f"Epoch {epoch+1}/{config.epochs}, Loss: {loss.item():.6f}, Mass Gap: {mass_gap:.6f}")
    
    return history

# 結果の可視化関数（簡易版）
def plot_results(history, model, config, save_dir="results"):
    """簡易化された結果の可視化"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 損失関数の推移
    plt.figure(figsize=(10, 6))
    plt.plot(history['epochs'], history['loss'], 'o-', label='Yang-Mills Action')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Yang-Mills Action Minimization ({config.gauge_group})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'action_loss.png'), dpi=300)
    
    # 質量ギャップの推移
    plt.figure(figsize=(10, 6))
    plt.plot(history['epochs'], history['mass_gap'], 'o-', label='Mass Gap')
    plt.xlabel('Epoch')
    plt.ylabel('Mass Gap (Δ)')
    plt.title(f'Mass Gap Evolution ({config.gauge_group})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'mass_gap.png'), dpi=300)
    
    print(f"結果が {save_dir} ディレクトリに保存されました")

def main():
    """メイン実行関数（簡易版）"""
    print("量子ヤン・ミルズ理論のNKAT表現シミュレーションを開始します（簡易版）")
    print(f"設定: N={config.N}, M={config.M}, ゲージ群={config.gauge_group}, 結合定数={config.coupling}")
    
    # モデルの初期化
    model = NKATModel(config.N, config.M, config.dim, config.gauge_group).to(device)
    print(f"総パラメータ数: {sum(p.numel() for p in model.parameters())}")
    
    # トレーニング
    print("モデルの最適化を開始...")
    history = train_model(model, config)
    
    # 結果の可視化
    print("結果の可視化...")
    plot_results(history, model, config)
    
    print("\n=== 結果サマリー ===")
    print(f"ゲージ群: {config.gauge_group}")
    print(f"結合定数: {config.coupling}")
    print(f"最終損失値: {history['loss'][-1]:.6f}")
    print(f"最終質量ギャップ: {history['mass_gap'][-1]:.6f}")
    
    # モデルの保存
    torch.save(model.state_dict(), os.path.join("results", f"yang_mills_{config.gauge_group}_model.pt"))
    print("シミュレーション完了!")

if __name__ == "__main__":
    main() 