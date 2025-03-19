import numpy as np
import torch
import time
import matplotlib.pyplot as plt

print("シンプルなテストスクリプトを実行中...")

# 簡単なテンソル計算
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
c = a + b
print(f"テスト計算結果: {c}")

# メトリクス計算のテスト
values = np.random.rand(10)
metrics = {
    "エネルギー": float(np.sum(values**2)),
    "エントロピー": float(-np.sum(values**2 * np.log(values**2 + 1e-10))),
    "平均値": float(np.mean(values)),
    "標準偏差": float(np.std(values))
}

print("計算されたメトリクス:")
for key, value in metrics.items():
    print(f"  {key}: {value}")

print("テスト完了！") 