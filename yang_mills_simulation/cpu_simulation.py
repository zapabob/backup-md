#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CPU専用：量子ヤン・ミルズ理論のNKAT表現シミュレーション（超シンプル版）
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time

print("量子ヤン・ミルズ理論のNKATシミュレーション（CPU専用超シンプル版）")
print(f"PyTorch バージョン: {torch.__version__}")

# CPUを強制的に使用
device = torch.device('cpu')
print(f"使用デバイス: {device}")

# シンプルな設定
N = 3  # 外部関数数
M = 3  # 内部関数数
dim = 4  # 時空間次元
lie_dim = 2  # SU(2)の簡略版
batch_size = 8
epochs = 5
os.makedirs("results", exist_ok=True)

# 乱数シード設定
torch.manual_seed(42)
np.random.seed(42)

# 超シンプルなモデル定義
class SimpleCPUModel(torch.nn.Module):
    def __init__(self):
        super(SimpleCPUModel, self).__init__()
        # 少ないパラメータで単純な行列を作成
        self.weights = torch.nn.Parameter(torch.randn(N, M) * 0.1)
        # シンプルな反対称行列
        self.Lambda = torch.tensor([[0, 1], [-1, 0]], dtype=torch.float32)
    
    def forward(self, x):
        # バッチサイズを取得
        batch_size = x.shape[0]
        # 単純化したゲージ場
        A_mu = torch.zeros(batch_size, dim, lie_dim, lie_dim)
        
        # 簡略化した計算
        for mu in range(dim):
            # 単純な係数計算
            coef = torch.sin(x[:, mu]).unsqueeze(1)
            # 全バッチに同じ行列を適用
            A_mu[:, mu] = coef * self.Lambda
        
        return A_mu
    
    def loss_function(self, x):
        # より単純な損失関数
        A_mu = self.forward(x)
        loss = 0
        for mu in range(dim-1):
            # 単純に隣接する方向の差の2乗で近似
            diff = A_mu[:, mu] - A_mu[:, mu+1]
            loss += torch.sum(diff**2) / batch_size
        return loss

# モデルと最適化器を初期化
start_time = time.time()
print("モデル初期化中...")
model = SimpleCPUModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# トレーニングループ
print("トレーニング開始...")
losses = []
for epoch in range(epochs):
    epoch_start = time.time()
    # ランダムな入力
    x = torch.randn(batch_size, dim)
    
    # 順伝播と損失計算
    optimizer.zero_grad()
    loss = model.loss_function(x)
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    epoch_time = time.time() - epoch_start
    print(f"エポック {epoch+1}/{epochs}, 損失: {loss.item():.6f}, 時間: {epoch_time:.2f}秒")

# 合計時間
total_time = time.time() - start_time
print(f"シミュレーション合計時間: {total_time:.2f}秒")

# 結果をプロット
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs+1), losses, 'o-')
plt.xlabel('エポック')
plt.ylabel('損失')
plt.title('ヤン・ミルズシンプルモデル学習曲線')
plt.grid(True)
plt.savefig(os.path.join("results", "cpu_simple_loss.png"))
plt.close()

print("シミュレーション完了！結果はresultsディレクトリに保存されました。") 