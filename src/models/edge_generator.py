"""Stage 1 邊緣生成器。

本檔案使用繁體中文說明研究程式的用途。"""
import torch
from torch import nn

class ResidualBlock(nn.Module):
    """ResidualBlock：封裝對應模型或研究流程。"""

    def __init__(self, c):
        """初始化元件並儲存必要的模型參數與執行狀態。"""
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.InstanceNorm2d(c),
            nn.ReLU(True),
            nn.Conv2d(c, c, 3, padding=1),
            nn.InstanceNorm2d(c)
        )

    def forward(self, x):
        """執行模型前向傳播，回傳對應的預測張量。"""
        return x + self.net(x)

class EdgeGenerator(nn.Module):
    """EdgeGenerator：封裝對應模型或研究流程。"""

    def __init__(self, input_channels=5, base=32, residual_blocks=4):
        """初始化元件並儲存必要的模型參數與執行狀態。"""
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(input_channels, base, 7, padding=3),
            nn.ReLU(True),
            nn.Conv2d(base, base * 2, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1),
            nn.ReLU(True)
        )
        self.mid = nn.Sequential(*[ResidualBlock(base * 4) for _ in range(residual_blocks)])
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(base, 1, 7, padding=3)
        )

    def forward(self, x):
        """執行模型前向傳播，回傳對應的預測張量。"""
        return self.dec(self.mid(self.enc(x)))
