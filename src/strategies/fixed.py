"""fixed。

本檔案使用繁體中文說明研究程式的用途。"""

class FixedWeights:
    """FixedWeights：封裝對應模型或研究流程。"""

    def __init__(self, weights):
        """初始化元件並儲存必要的模型參數與執行狀態。"""
        self.weights = dict(weights)

    def get_weights(self, epoch):
        """get_weights：執行此模組的對應運算或資料處理。"""
        return dict(self.weights)
