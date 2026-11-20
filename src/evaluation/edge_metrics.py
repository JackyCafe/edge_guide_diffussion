"""Stage 1 的閾值掃描、Pixel F1、IoU 與 2px Tolerance F1。"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def threshold_sweep(prob, target, mask, thresholds=None, tolerance=2):
    """依整個驗證集累積 TP/FP/FN，計算最佳二值化閾值。

    Tolerance F1 採雙向膨脹匹配：預測邊緣落在真值容差內算命中，
    真值邊緣落在預測容差內算召回；不等同一對一像素配對。
    """
    if thresholds is None:
        thresholds = [round(i * 0.05, 2) for i in range(1, 20)]

    area = mask > 0.5
    true_edge = (target > 0.5) & area
    results = {}

    for threshold in thresholds:
        predicted = (prob >= threshold) & area
        tp = (predicted & true_edge).sum().item()
        fp = (predicted & ~true_edge & area).sum().item()
        fn = (~predicted & true_edge).sum().item()

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        iou = tp / max(1, tp + fp + fn)

        kernel = 2 * tolerance + 1
        dilated_true = F.max_pool2d(
            true_edge.float(), kernel, stride=1, padding=tolerance
        ) > 0.5
        dilated_pred = F.max_pool2d(
            predicted.float(), kernel, stride=1, padding=tolerance
        ) > 0.5

        tolerance_precision = (
            (predicted & dilated_true).sum().item()
            / max(1, predicted.sum().item())
        )
        tolerance_recall = (
            (true_edge & dilated_pred).sum().item()
            / max(1, true_edge.sum().item())
        )
        tolerance_f1 = (
            2 * tolerance_precision * tolerance_recall
            / max(1e-12, tolerance_precision + tolerance_recall)
        )

        results[threshold] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "tolerance_f1": tolerance_f1,
        }

    return results
