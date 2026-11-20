"""正向加噪與單步影像估計。

本檔案使用繁體中文說明研究程式的用途。"""
import torch

def noise_images(x0, t, schedule, noise=None):
    """noise_images：執行此模組的對應運算或資料處理。"""
    if noise is None:
        noise = torch.randn_like(x0)
    a = schedule.extract(schedule.abar, t, x0)
    return (a.sqrt() * x0 + (1 - a).sqrt() * noise, noise)

def predict_x0(x_t, t, eps, schedule):
    """由目前時間步的帶噪影像及預測雜訊估算單步 x0。"""
    a = schedule.extract(schedule.abar, t, x_t)
    return (x_t - (1 - a).sqrt() * eps) / a.clamp_min(1e-08).sqrt()
