
"""測試 Stage 1 與 Stage 2 Loss 的反向傳播。
這是 Edge-Diffusion 專案的 Loss 梯度測試程式，包含兩個測試：
- test_edge_gradient()：確認 Stage 1 的 Edge Loss 可以正常反向傳播。
- test_snr_loss()：確認 Stage 2 的 Diffusion Loss 可以正常反向傳播
"""

import torch

from src.diffusion.schedule import DiffusionSchedule
from src.losses.diffusion_loss import diffusion_losses
from src.losses.edge_loss import edge_losses


def test_edge_gradient() -> None:
    """確認 Edge Loss 可以產生有效且有限的梯度。"""

    # 固定隨機種子，確保測試結果可重現
    torch.manual_seed(42)

    # 建立需要計算梯度的 Edge Logits
    logits = torch.randn(
        2,
        1,
        16,
        16,
        requires_grad=True,
    )

    # 建立二值化的邊緣真值
    target = (
        torch.rand_like(logits) > 0.8
    ).float()

    # 建立全為 1 的 Mask
    mask = torch.ones_like(logits)

    # 計算 Stage 1 各項 Loss
    loss_dict = edge_losses(
        logits,
        target,
        mask,
    )

    # 將各項 Loss 加總
    total_loss = torch.stack(
        list(loss_dict.values())
    ).sum()

    # 執行反向傳播
    total_loss.backward()

    # 確認梯度存在
    assert logits.grad is not None

    # 確認梯度沒有 NaN 或 Infinity
    assert torch.isfinite(logits.grad).all()


def test_snr_loss() -> None:
    """確認 Diffusion Loss 可以產生有效且有限的梯度。"""

    torch.manual_seed(42)

    # 建立 20 個時間步的 Diffusion Schedule
    schedule = DiffusionSchedule(timesteps=20)

    # 模擬模型預測的雜訊
    pred_noise = torch.randn(
        2,
        3,
        16,
        16,
        requires_grad=True,
    )

    # 建立真實雜訊
    noise = torch.randn_like(pred_noise)

    # 建立原始影像
    ground_truth = torch.zeros_like(pred_noise)

    # 指定不同時間步
    timesteps = torch.tensor([0, 19])

    # 建立全為 1 的 Mask
    mask = torch.ones(
        2,
        1,
        16,
        16,
    )

    # 計算 Stage 2 各項 Loss
    loss_dict = diffusion_losses(
        pred_noise=pred_noise,
        noise=noise,
        pred_x0=pred_noise,
        gt=ground_truth,
        mask=mask,
        t=timesteps,
        sch=schedule,
    )

    # 將各項 Loss 加總
    total_loss = torch.stack(
        list(loss_dict.values())
    ).sum()

    # 執行反向傳播
    total_loss.backward()

    # 確認梯度存在
    assert pred_noise.grad is not None

    # 確認梯度沒有 NaN 或 Infinity
    assert torch.isfinite(pred_noise.grad).all()
