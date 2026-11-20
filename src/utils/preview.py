"""每個 Epoch 的影像比較預覽。

本檔案使用繁體中文說明研究程式的用途。"""
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def save_preview(path, images, names):
    """save_preview：執行此模組的對應運算或資料處理。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(images), figsize=(3.5 * len(images), 3.8))
    for ax, image, name in zip(axes, images, names):
        image = image.detach().cpu().clamp(-1, 1)
        if image.shape[0] == 1:
            ax.imshow(image[0].numpy(), cmap='gray', vmin=-1, vmax=1)
        else:
            ax.imshow(image.permute(1, 2, 0).numpy() * 0.5 + 0.5)
        ax.set_title(name)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
