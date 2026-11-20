"""Stage 1 邊緣判別器。

本檔案使用繁體中文說明研究程式的用途。"""
from torch import nn

class EdgeDiscriminator(nn.Module):
    """EdgeDiscriminator：封裝對應模型或研究流程。"""

    def __init__(self, input_channels=3, base=32):
        """初始化元件並儲存必要的模型參數與執行狀態。"""
        super().__init__()
        self.blocks = nn.ModuleList([nn.Sequential(nn.Conv2d(input_channels, base, 4, 2, 1), nn.LeakyReLU(0.2)), nn.Sequential(nn.Conv2d(base, base * 2, 4, 2, 1), nn.LeakyReLU(0.2)), nn.Sequential(nn.Conv2d(base * 2, base * 4, 4, 2, 1), nn.LeakyReLU(0.2))])
        self.out = nn.Conv2d(base * 4, 1, 3, padding=1)

    def forward(self, x):
        """執行模型前向傳播，回傳對應的預測張量。"""
        feats = []
        for block in self.blocks:
            x = block(x)
            feats.append(x)
        return (self.out(x), feats)
