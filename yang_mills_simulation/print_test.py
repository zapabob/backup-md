#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
標準出力テスト用の最小スクリプト
"""

import sys

print("標準出力テスト")
print("これは標準出力に表示されるはずです")
sys.stdout.write("sys.stdoutを使った出力テスト\n")
sys.stdout.flush()

print(f"Pythonバージョン: {sys.version}")
print(f"実行パス: {sys.executable}")

# エラーテスト
try:
    print("エラー出力テスト")
    sys.stderr.write("これはエラー出力です\n")
    sys.stderr.flush()
except Exception as e:
    print(f"エラー: {e}")

print("テスト終了") 