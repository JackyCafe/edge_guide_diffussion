"""影像視覺化工具。

本檔案使用繁體中文說明研究程式的用途。"""
from pathlib import Path
import torch
from torchvision.utils import save_image

def save_rgb(tensor, path):
    """Save normalized RGB or grayscale tensor as an image."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image((tensor.detach().cpu().clamp(-1, 1) + 1) / 2, path)
