"""Stage 2 單步 x0 重建、邊界及 VGG 感知損失。"""

import torch
import torch.nn.functional as F


def boundary_mask(mask: torch.Tensor, kernel: int = 11) -> torch.Tensor:
    """以膨脹減侵蝕取得接縫帶，再限制於缺損區內側。"""
    if kernel < 1 or kernel % 2 != 1:
        raise ValueError("Boundary kernel 必須為正奇數。")

    dilated = F.max_pool2d(mask, kernel, stride=1, padding=kernel // 2)
    eroded = -F.max_pool2d(-mask, kernel, stride=1, padding=kernel // 2)
    return (dilated - eroded).clamp(0.0, 1.0) * mask


def per_image_masked_mean(
    error: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """先計算各張影像的遮罩平均誤差，再交由上層跨樣本平均。"""
    numerator = (error * mask).sum(dim=(1, 2, 3))
    denominator = mask.sum(dim=(1, 2, 3)) * error.shape[1]
    return numerator / denominator.clamp_min(1e-8)


def diffusion_losses(
    pred_noise,
    noise,
    pred_x0,
    gt,
    mask,
    t,
    schedule,
    vgg=None,
    snr_gamma=5.0,
    snr_weighting=True,
    vgg_max_t=500,
    boundary_kernel=11,
):
    """計算 epsilon MSE 與單步 x0 的像素空間損失。

    注意：單步 pred_x0 只用於訓練 Loss，正式驗證必須執行完整 DDIM。
    Hole/Boundary 不做硬性 clamp，避免飽和區域梯度直接歸零。
    VGG 則限制時間步並在特徵輸入前將影像裁切至 [-1, 1]。
    """
    noise_loss = F.mse_loss(pred_noise, noise)

    if snr_weighting:
        # 此處是重建項目專用權重，不是 epsilon Min-SNR-gamma。
        sample_weights = (schedule.snr(t) / snr_gamma).clamp(max=1.0)
    else:
        sample_weights = torch.ones_like(t, dtype=gt.dtype)

    error = (pred_x0 - gt).abs()
    hole_error = per_image_masked_mean(error, mask)
    band = boundary_mask(mask, boundary_kernel)
    boundary_error = per_image_masked_mean(error, band)

    losses = {
        "noise": noise_loss,
        "hole": (sample_weights * hole_error).mean(),
        "boundary": (sample_weights * boundary_error).mean(),
        "vgg": noise_loss.new_zeros(()),
    }

    if vgg is not None:
        eligible = t <= vgg_max_t
        if eligible.any():
            # 逐張影像計算，保留每個時間步對應的 SNR 權重。
            indices = torch.nonzero(eligible).flatten().tolist()
            perceptual = [
                vgg(pred_x0[i:i + 1].clamp(-1.0, 1.0), gt[i:i + 1])
                for i in indices
            ]
            losses["vgg"] = (
                sample_weights[eligible] * torch.stack(perceptual)
            ).sum() / t.numel()

    return losses
