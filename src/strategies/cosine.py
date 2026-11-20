"""Cosine Loss 權重排程。

本檔案使用繁體中文說明研究程式的用途。"""
import math

class CosineWeights:
    """CosineWeights：封裝對應模型或研究流程。"""

    def __init__(self, start, end, epochs):
        """初始化元件並儲存必要的模型參數與執行狀態。"""
        self.start = start
        self.end = end
        self.epochs = max(1, epochs)

    def get_weights(self, epoch):
        """get_weights：執行此模組的對應運算或資料處理。"""
        c = 0.5 * (1 + math.cos(math.pi * min(epoch / self.epochs, 1)))
        return {
            k: self.end.get(k, v) + (v - self.end.get(k, v)) * c for k,
            v in self.start.items()
        }
