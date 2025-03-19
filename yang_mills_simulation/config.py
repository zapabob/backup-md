#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
量子ヤン・ミルズ理論の数値シミュレーション設定
"""

class SimulationConfig:
    """シミュレーション設定パラメータ"""
    def __init__(self):
        # NKAT表現パラメータ
        self.N = 20               # 外部関数の数（計算負荷軽減のため小さめの値に設定）
        self.M = 20               # 内部関数の数（計算負荷軽減のため小さめの値に設定）
        self.dim = 4              # 時空間次元
        
        # 物理パラメータ
        self.gauge_group = "SU3"  # ゲージ群（"SU2", "SU3", "G2", "F4"）
        self.coupling = 1.0       # 結合定数
        
        # 最適化パラメータ
        self.epochs = 200         # RTX 3080の計算能力に合わせて低めに設定
        self.batch_size = 64      # バッチサイズ
        self.learning_rate = 1e-3 # 学習率
        
        # その他設定
        self.seed = 42            # 乱数シード（再現性のため）
        self.save_interval = 20   # 結果保存間隔（エポック数）
        self.plot_figures = True  # グラフを生成するかどうか
        self.save_model = True    # モデルを保存するかどうか 