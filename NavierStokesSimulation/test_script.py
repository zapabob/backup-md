print("このテストは正常に実行されています")

import numpy as np
print(f"NumPy バージョン: {np.__version__}")

import matplotlib.pyplot as plt
print(f"グラフィックライブラリの読み込み成功")

# 簡単な計算とファイル生成
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 4))
plt.plot(x, y)
plt.title('正弦波テスト')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.savefig('test_plot.png')
plt.close()

print("テストプロットを test_plot.png として保存しました") 