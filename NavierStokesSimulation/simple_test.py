import numpy as np
import matplotlib.pyplot as plt

def main():
    print("テスト実行中...")
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    
    print("テスト完了")

if __name__ == "__main__":
    main() 