"""RC1：修補指標與 Boundary Mask 的基本驗收。"""

import torch

from src.evaluation.image_metrics import masked_mae, masked_psnr
from src.losses.diffusion_loss import boundary_mask


def test_known_region_composite_has_zero_error():
    """合成後已知區域應完整保留原始影像。"""
    image = torch.rand(2, 1, 16, 16) * 2 - 1
    generated = torch.rand_like(image) * 2 - 1
    mask = torch.zeros(2, 1, 16, 16)
    mask[:, :, 3:10, 4:11] = 1
    output = image * (1 - mask) + generated * mask
    assert masked_mae(output, image, 1 - mask) == 0.0


def test_hole_psnr_uses_only_mask_pixels():
    """非缺損區域的誤差不應改變 Hole PSNR。"""
    image = torch.zeros(1, 1, 16, 16)
    prediction = image.clone()
    mask = torch.zeros_like(image)
    mask[:, :, 3:10, 3:10] = 1
    prediction[:, :, 0, 0] = 1.0
    assert masked_psnr(prediction, image, mask) > 90


def test_boundary_band_is_inside_hole():
    """Boundary 主指標只評估 Mask 的內側接縫。"""
    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, 8:24, 8:24] = 1
    band = boundary_mask(mask, kernel=11)
    assert torch.all(band <= mask)
    assert band.sum().item() > 0
