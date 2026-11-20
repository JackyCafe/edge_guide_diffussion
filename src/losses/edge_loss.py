"""Stage 1 邊緣生成損失。

本檔案使用繁體中文說明研究程式的用途。"""
import torch
import torch.nn.functional as F

def masked_mean(x, mask):
    """masked_mean：執行此模組的對應運算或資料處理。"""
    mask = mask.expand_as(x)
    return (x * mask).sum() / mask.sum().clamp_min(1)

def edge_losses(logits, target, mask, pos_weight=5.0):
    """edge_losses：執行此模組的對應運算或資料處理。"""
    weight = torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    wbce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=weight,
        reduction='none'
    )
    wbce = masked_mean(wbce, mask)
    p = logits.sigmoid()
    intersection = (p * target * mask).sum(dim=(1, 2, 3))
    denom = ((p + target) * mask).sum(dim=(1, 2, 3))
    dice = (1 - (2 * intersection + 1e-06) / (denom + 1e-06)).mean()
    return {'wbce': wbce, 'dice': dice}

def feature_matching(fake, real):
    """feature_matching：執行此模組的對應運算或資料處理。"""
    return sum(
        (F.l1_loss(a, b.detach()) for a, b in zip(fake, real))) / max(1,
        len(fake)
    )
