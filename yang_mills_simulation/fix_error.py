#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
テンソルサイズの不一致エラーを修正するためのデバッグスクリプト
"""

import torch
import numpy as np

# 小さなモデルで動作確認
N, M = 5, 5
dim = 4
batch_size = 32

# パラメータ初期化
a = torch.randn(N)
b = torch.randn(N)
c = torch.randn(N)
d = torch.randn(N)

# バッチサイズのテンソル
x = torch.randn(batch_size, dim)

# エラーが発生する操作
z = torch.zeros(batch_size, 1)
print(f"z shape: {z.shape}")

# 修正前: エラーが発生するコード
try:
    # view(-1, 1)はテンソルのサイズを変更するが、バッチサイズと互換性がない
    result1 = torch.tanh(a.view(-1, 1) * z + b.view(-1, 1))
    print("修正前のコードは成功しました（予期しない結果）")
except RuntimeError as e:
    print(f"修正前のコードでエラーが発生: {e}")

# 修正方法1: unsqueeze(0)で次元を追加し、ブロードキャスト
try:
    # unsqueeze(0)で次元を追加し、バッチサイズとブロードキャスト
    result2 = torch.tanh(a.unsqueeze(0) * z + b.unsqueeze(0))
    print(f"修正方法1は成功: 結果のshape = {result2.shape}")
except RuntimeError as e:
    print(f"修正方法1でエラーが発生: {e}")

# 修正方法2: 明示的なブロードキャスト
try:
    # 明示的にバッチサイズに拡張
    a_expanded = a.expand(batch_size, -1)
    b_expanded = b.expand(batch_size, -1)
    result3 = torch.tanh(a_expanded * z + b_expanded)
    print(f"修正方法2は成功: 結果のshape = {result3.shape}")
except RuntimeError as e:
    print(f"修正方法2でエラーが発生: {e}")

print("\n修正した外部関数のコード:")
print("""
def external_function(self, z):
    # 外部関数: Φ_i(z) = tanh(a_i*z + b_i) + c_i*z^2*tanh(d_i*z)
    return torch.tanh(self.a.unsqueeze(0) * z + self.b.unsqueeze(0)) + \\
           self.c.unsqueeze(0) * z**2 * torch.tanh(self.d.unsqueeze(0) * z)
""")

print("\n修正した内部関数のコード:")
print("""
def internal_function(self, x, n_idx, m_idx):
    # 内部関数: φ_{ij}(x_j) = α_{ij}*exp(-β_{ij}*|x_j|^2) + γ_{ij}*x_j*exp(-δ_{ij}*|x_j|^2)*cos(ω_{ij}*|x_j|)
    x_squared = torch.sum(x**2, dim=-1, keepdim=True)
    term1 = self.alpha[n_idx, m_idx].unsqueeze(0) * torch.exp(-self.beta[n_idx, m_idx].unsqueeze(0) * x_squared)
    term2 = self.gamma[n_idx, m_idx].unsqueeze(0) * x * torch.exp(-self.delta[n_idx, m_idx].unsqueeze(0) * x_squared) * \\
            torch.cos(self.omega[n_idx, m_idx].unsqueeze(0) * torch.sqrt(x_squared + 1e-10))
    return term1 + term2
""")

print("\nこの修正を適用してエラーを解決してください。") 