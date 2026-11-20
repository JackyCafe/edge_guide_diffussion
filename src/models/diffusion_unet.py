"""Stage 2 時間條件 U-Net。

本檔案使用繁體中文說明研究程式的用途。"""
import math
import torch
from torch import nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    """TimeEmbedding：封裝對應模型或研究流程。"""

    def __init__(self, dim):
        """初始化元件並儲存必要的模型參數與執行狀態。"""
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, t):
        """執行模型前向傳播，回傳對應的預測張量。"""
        half = self.dim // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(1, half - 1))
        angles = t.float()[:, None] * freq[None]
        emb = torch.cat([angles.sin(), angles.cos()], 1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return self.net(emb)

class TimeBlock(nn.Module):
    """TimeBlock：封裝對應模型或研究流程。"""

    def __init__(self, inc, outc, td):
        """初始化元件並儲存必要的模型參數與執行狀態。"""
        super().__init__()
        self.conv1 = nn.Conv2d(inc, outc, 3, padding=1)
        self.conv2 = nn.Conv2d(outc, outc, 3, padding=1)
        self.tproj = nn.Linear(td, outc)
        self.norm1 = nn.GroupNorm(8, outc)
        self.norm2 = nn.GroupNorm(8, outc)
        self.skip = nn.Conv2d(inc, outc, 1) if inc != outc else nn.Identity()

    def forward(self, x, t):
        """執行模型前向傳播，回傳對應的預測張量。"""
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.tproj(t)[:, :, None, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)

class DiffusionUNet(nn.Module):
    """DiffusionUNet：封裝對應模型或研究流程。"""

    def __init__(self, in_channels=8, out_channels=3, base=32, time_dim=128):
        """初始化元件並儲存必要的模型參數與執行狀態。"""
        super().__init__()
        self.time = TimeEmbedding(time_dim)
        self.down1 = TimeBlock(in_channels, base, time_dim)
        self.down2 = TimeBlock(base, base * 2, time_dim)
        self.mid = TimeBlock(base * 2, base * 2, time_dim)
        self.up1 = TimeBlock(base * 3, base, time_dim)
        self.out = nn.Conv2d(base, out_channels, 3, padding=1)

    def forward(self, x_t, t, condition):
        """執行模型前向傳播，回傳對應的預測張量。"""
        te = self.time(t)
        x = torch.cat([x_t, condition], 1)
        a = self.down1(x, te)
        b = self.down2(F.avg_pool2d(a, 2), te)
        m = self.mid(b, te)
        u = F.interpolate(m, size=a.shape[-2:], mode='nearest')
        return self.out(self.up1(torch.cat([u, a], 1), te))
