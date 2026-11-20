"""Stage 1 邊緣與 Stage 2 修補影像的正式驗證流程。"""

import torch

from src.diffusion.sampler import ddim_sample
from src.evaluation.edge_metrics import threshold_sweep
from src.evaluation.image_metrics import (
    boundary_mae,
    masked_mae,
    masked_psnr,
    masked_ssim,
)


@torch.no_grad()
def evaluate_stage1(model, loader, device):
    """在固定驗證集搜尋最佳邊緣閾值，並回傳完整指標。"""
    model.eval()
    predictions = []
    targets = []
    masks = []

    for batch in loader:
        masked = batch["masked_image"].to(device)
        mask = batch["mask"].to(device)
        known_edge = batch["known_edge"].to(device)

        # G1 的輸出是 logits；只有評估時才轉為機率。
        logits = model(torch.cat([masked, mask, known_edge], dim=1))
        predictions.append(logits.sigmoid().cpu())
        targets.append(batch["edge"].cpu())
        masks.append(mask.cpu())

    if not predictions:
        raise ValueError("Stage 1 驗證資料集不可為空。")

    results = threshold_sweep(
        torch.cat(predictions),
        torch.cat(targets),
        torch.cat(masks),
    )
    threshold, metrics = max(
        results.items(),
        key=lambda item: (item[1]["f1"], -item[0]),
    )
    return {**metrics, "threshold": threshold}


@torch.no_grad()
def evaluate_stage2(model, g1, schedule, loader, device, steps=50):
    """以完整 DDIM Sampling 驗證，並以影像數量加權彙整指標。

    此函式不用訓練階段的單步 pred_x0；最佳 PSNR 僅由此流程選擇。
    """
    model.eval()
    g1.eval()
    totals = {
        "psnr": 0.0,
        "ssim": 0.0,
        "mae": 0.0,
        "hole_psnr": 0.0,
        "hole_mae": 0.0,
        "boundary_mae": 0.0,
        "known_mae": 0.0,
    }
    sample_count = 0

    for batch in loader:
        image = batch["image"].to(device)
        masked = batch["masked_image"].to(device)
        mask = batch["mask"].to(device)
        known_edge = batch["known_edge"].to(device)
        batch_size = image.shape[0]

        # 完整 Edge Guide：已知區域使用 Canny，缺損區域使用 G1。
        edge_logits = g1(torch.cat([masked, mask, known_edge], dim=1))
        completed_edge = known_edge + mask * edge_logits.sigmoid()
        condition = torch.cat([masked, mask, completed_edge], dim=1)

        generated = ddim_sample(
            model,
            schedule,
            condition,
            steps=steps,
            seed=42,
            known_image=image,
            mask=mask,
        )
        output = mask * generated + (1.0 - mask) * image
        full_mask = torch.ones_like(mask)
        known_mask = 1.0 - mask

        batch_metrics = {
            "psnr": masked_psnr(output, image, full_mask),
            "ssim": masked_ssim(output, image, full_mask),
            "mae": masked_mae(output, image, full_mask),
            "hole_psnr": masked_psnr(output, image, mask),
            "hole_mae": masked_mae(output, image, mask),
            "boundary_mae": boundary_mae(output, image, mask),
            "known_mae": masked_mae(output, image, known_mask),
        }
        for name, value in batch_metrics.items():
            totals[name] += value * batch_size
        sample_count += batch_size

    if sample_count == 0:
        raise ValueError("Stage 2 驗證資料集不可為空。")

    return {name: total / sample_count for name, total in totals.items()}
