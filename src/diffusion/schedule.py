"""擴散模型時間排程。

本檔案使用繁體中文說明研究程式的用途。"""
import torch
from torch import nn

class DiffusionSchedule(nn.Module):
    """DiffusionSchedule：封裝對應模型或研究流程。"""

    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        """初始化元件並儲存必要的模型參數與執行狀態。"""
        super().__init__()
        self.timesteps = timesteps
        beta = torch.linspace(beta_start, beta_end, timesteps)
        alpha = 1 - beta
        abar = torch.cumprod(alpha, 0)
        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('abar', abar)

    def extract(self, values, t, x):
        """extract：執行此模組的對應運算或資料處理。"""
        return values[t].reshape(-1, *[1] * (x.ndim - 1))

    def snr(self, t):
        """snr：執行此模組的對應運算或資料處理。"""
        a = self.abar[t]
        return a / (1 - a).clamp_min(1e-08)
