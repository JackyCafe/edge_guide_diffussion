"""學習率排程。

本檔案使用繁體中文說明研究程式的用途。"""
import math

def learning_rate(epoch, total, base, warmup=5, min_ratio=0.1):
    """learning_rate：執行此模組的對應運算或資料處理。"""
    if epoch < warmup:
        return base * (min_ratio + (1 - min_ratio) * epoch / max(1, warmup))
    progress = min(1, (epoch - warmup) / max(1, total - warmup - 1))
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))
