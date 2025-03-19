#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
超シンプル版：量子ヤン・ミルズ理論のNKAT表現シミュレーション
RTX 3080 GPUで実行可能
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

print("量子ヤン・ミルズ理論のNKATシミュレーション（超シンプル版）")

# CUDA確認
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用デバイス: {device}")
if device.type == 'cuda':
    print(f"GPUモデル: {torch.cuda.get_device_name(0)}")

# シンプルな設定
N = 5  # 外部関数数
M = 5  # 内部関数数
dim = 4  # 時空間次元
lie_dim = 3  # SU(2)のリー代数次元
batch_size = 16
epochs = 10
os.makedirs("results", exist_ok=True)

# 乱数シード設定
torch.manual_seed(42)
np.random.seed(42)

# 超シンプルなモデル定義
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        # すべてのパラメータを一つのテンソルに
        self.params = torch.nn.Parameter(torch.randn(N * M * 3, device=device) * 0.1)
        # リー代数の基底（単に反対称行列として扱う）
        self.Lambda = torch.nn.Parameter(torch.randn(lie_dim, lie_dim, device=device) * 0.1)
        with torch.no_grad():
            self.Lambda.data = 0.5 * (self.Lambda.data - self.Lambda.data.transpose(-1, -2))
    
    def forward(self, x):
        # 単にバッチごとにランダムな行列を生成
        batch_size = x.shape[0]
        A_mu = torch.zeros(batch_size, dim, lie_dim, lie_dim, device=device)
        
        # 簡略化した計算
        for mu in range(dim):
            # 各方向に対して適当な係数を計算
            coef = torch.sin(torch.matmul(x, self.params[:dim].unsqueeze(1))).mean(dim=1, keepdim=True)
            A_mu[:, mu] = coef * self.Lambda
        
        return A_mu
    
    def loss_function(self, x):
        # シンプルな損失関数（交換子のノルムの平均）
        A_mu = self.forward(x)
        loss = 0
        for mu in range(dim):
            for nu in range(mu+1, dim):
                commutator = torch.matmul(A_mu[:, mu], A_mu[:, nu]) - torch.matmul(A_mu[:, nu], A_mu[:, mu])
                loss += torch.norm(commutator, dim=(-2,-1)).mean()
        return loss

# モデルと最適化器を初期化
model = SimpleModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# トレーニングループ
losses = []
for epoch in range(epochs):
    # ランダムな入力
    x = torch.randn(batch_size, dim, device=device)
    
    # 順伝播と損失計算
    optimizer.zero_grad()
    loss = model.loss_function(x)
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

# 結果をプロット
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs+1), losses, 'o-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Yang-Mills Simplified Model Training')
plt.grid(True)
plt.savefig(os.path.join("results", "simple_loss.png"))
plt.close()

print("シミュレーション完了！結果はresultsディレクトリに保存されました。") 