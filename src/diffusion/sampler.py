"""DDIM 反向取樣與已知區域保留。

本檔案使用繁體中文說明研究程式的用途。"""
import torch
from .forward import predict_x0

@torch.no_grad()
def ddim_sample(
    model,
    schedule,
    condition,
    steps=50,
    seed=42,
    known_image=None,
    mask=None
):
    """以 DDIM eta=0 反向取樣；每個時間步維持已知區域一致。"""
    device = condition.device
    b, _, h, w = condition.shape
    gen = torch.Generator(device=device).manual_seed(int(seed))
    x = torch.randn((b, model.out.out_channels, h, w), generator=gen, device=device)
    if (known_image is None) != (mask is None):
        raise ValueError('known_image and mask must be provided together')
    if known_image is not None:
        known_noise = torch.randn(known_image.shape, generator=gen, device=device)
    times = torch.linspace(
        schedule.timesteps - 1,
        0,
        steps,
        device=device).long().unique_consecutive(
    )
    # DDIM 依時間步由大至小反向去噪。
    for j, time in enumerate(times):
        t = torch.full((b,), int(time), device=device, dtype=torch.long)
        if known_image is not None:
            alpha = schedule.abar[time]
            known_t = alpha.sqrt() * known_image + (1 - alpha).sqrt() * known_noise
            # 已知區域在每個時間步使用對應噪聲狀態，避免接縫錯位。
            x = x * mask + known_t * (1 - mask)
        eps = model(x, t, condition)
        x0 = predict_x0(x, t, eps, schedule).clamp(-1, 1)
        if j == len(times) - 1:
            x = x0
            break
        next_alpha = schedule.abar[times[j + 1]]
        x = next_alpha.sqrt() * x0 + (1 - next_alpha).sqrt() * eps
    if known_image is not None:
        x = x * mask + known_image * (1 - mask)
    return x.clamp(-1, 1)
