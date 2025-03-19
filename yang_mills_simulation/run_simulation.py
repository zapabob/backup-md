#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
量子ヤン・ミルズ理論のNKAT表現シミュレーション実行ファイル
"""

import os
import sys
import time
import torch
from config import SimulationConfig
from yang_mills_nkat_simulation import main

if __name__ == "__main__":
    print("=== 量子ヤン・ミルズ理論のNKAT表現シミュレーション ===")
    print("NVIDIA RTX 3080 GPU向けに最適化されたバージョン")
    
    # GPUが利用可能か確認
    if not torch.cuda.is_available():
        print("警告: CUDAデバイスが見つかりません。CPUで実行します（非常に遅くなります）")
        proceed = input("続行しますか？ (y/n): ")
        if proceed.lower() != 'y':
            sys.exit(0)
    
    # 実行時間計測開始
    start_time = time.time()
    
    # メイン関数実行
    main()
    
    # 実行時間計測終了
    elapsed = time.time() - start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n合計実行時間: {int(hours)}時間 {int(minutes)}分 {seconds:.2f}秒")
    print("シミュレーションが完了しました。")
    print("結果はresultsディレクトリに保存されています。") 