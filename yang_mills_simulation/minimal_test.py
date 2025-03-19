#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyTorch動作確認用の最小限のテストスクリプト
"""

import torch
import time

print("PyTorch最小限テスト")
print(f"PyTorchバージョン: {torch.__version__}")

# CPU/GPU確認
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用デバイス: {device}")
if device.type == 'cuda':
    print(f"GPUモデル: {torch.cuda.get_device_name(0)}")
    print(f"CUDAバージョン: {torch.version.cuda}")

# 単純なテンソル演算
start_time = time.time()
print("テンソル演算テスト中...")

# テンソル生成
a = torch.randn(1000, 1000, device=device)
b = torch.randn(1000, 1000, device=device)

# 行列積
c = torch.matmul(a, b)

# 単純なニューラルネットワーク層のテスト
linear = torch.nn.Linear(1000, 100).to(device)
d = linear(a)

end_time = time.time()
print(f"テスト完了！計算時間: {end_time - start_time:.4f}秒")
print(f"生成された行列のサイズ: {c.shape}, {d.shape}")
print(f"サンプル値: {c[0, 0].item():.4f}, {d[0, 0].item():.4f}")

print("テスト正常終了") 