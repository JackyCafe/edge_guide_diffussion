"""影像 PSNR、SSIM 與 MAE 指標。

本檔案使用繁體中文說明研究程式的用途。"""
import torch
import torch.nn.functional as F
from src.losses.diffusion_loss import boundary_mask

def masked_psnr(pred, gt, mask):
    """masked_psnr：執行此模組的對應運算或資料處理。"""
    error = ((pred.clamp(-1, 1) - gt) ** 2 * mask).sum((1, 2, 3))
    mse = error / (mask.sum((1, 2, 3)) * pred.shape[1]).clamp_min(1)
    return (10 * torch.log10(4 / mse.clamp_min(1e-10))).mean().item()

def masked_mae(pred, gt, mask):
    """masked_mae：執行此模組的對應運算或資料處理。"""
    error = (pred.clamp(-1, 1) - gt).abs()
    return ((error * mask).sum((1, 2, 3)) / (mask.sum((1, 2, 3)) * pred.shape[1]).clamp_min(1)).mean().item()

def masked_ssim(pred, gt, mask, window=11):
    """masked_ssim：執行此模組的對應運算或資料處理。"""
    x = (pred.clamp(-1, 1) + 1) / 2
    y = (gt.clamp(-1, 1) + 1) / 2
    pool = lambda z: F.avg_pool2d(
        z,
        window,
        stride=1,
        padding=window // 2,
        count_include_pad=False
    )
    ux, uy = (pool(x), pool(y))
    vx, vy = (
        (pool(x * x) - ux * ux).clamp_min(0),
        (pool(y * y) - uy * uy).clamp_min(0)
    )
    cov = pool(x * y) - ux * uy
    ssim = (2 * ux * uy + 0.01 ** 2) * (2 * cov + 0.03 ** 2) / ((ux * ux + uy * uy + 0.01 ** 2) * (vx + vy + 0.03 ** 2)).clamp_min(1e-12)
    return ((ssim * mask).sum((1, 2, 3)) / (mask.sum((1, 2, 3)) * x.shape[1]).clamp_min(1)).mean().item()

def boundary_mae(pred, gt, mask, kernel=11):
    """boundary_mae：執行此模組的對應運算或資料處理。"""
    band = boundary_mask(mask, kernel)
    valid = band.sum((1, 2, 3)) > 0
    if not valid.any():
        return float('nan')
    err = ((pred - gt).abs() * band).sum((1, 2, 3)) / (band.sum((1, 2, 3)) * gt.shape[1]).clamp_min(1)
    return err[valid].mean().item()
