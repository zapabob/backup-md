#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
純粋なPythonのみのテストスクリプト
"""

import time
import os
import sys

print("純粋なPythonテスト")
print(f"Pythonバージョン: {sys.version}")
print(f"実行パス: {sys.executable}")

# システム情報
print(f"OS: {os.name}, {sys.platform}")
print(f"現在の作業ディレクトリ: {os.getcwd()}")

# 単純な計算のテスト
start_time = time.time()
print("計算テスト中...")

# 行列計算のシミュレーション
def matrix_mult(size):
    a = [[0.0] * size for _ in range(size)]
    b = [[0.0] * size for _ in range(size)]
    c = [[0.0] * size for _ in range(size)]
    
    # 初期化
    for i in range(size):
        for j in range(size):
            a[i][j] = i * 0.1 + j * 0.2
            b[i][j] = i * 0.3 + j * 0.1
    
    # 行列乗算
    for i in range(size):
        for j in range(size):
            for k in range(size):
                c[i][j] += a[i][k] * b[k][j]
    
    return c

# テスト実行
result = matrix_mult(100)  # 小さいサイズで実行
sample_value = result[10][10]

end_time = time.time()
print(f"テスト完了！計算時間: {end_time - start_time:.4f}秒")
print(f"サンプル値: {sample_value:.4f}")

# ファイル書き込みテスト
try:
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "pure_python_test.txt"), "w") as f:
        f.write(f"テスト時刻: {time.ctime()}\n")
        f.write(f"計算時間: {end_time - start_time:.4f}秒\n")
        f.write(f"サンプル値: {sample_value:.4f}\n")
    print("ファイル書き込みテスト成功!")
except Exception as e:
    print(f"ファイル書き込みエラー: {e}")

print("テスト正常終了") 