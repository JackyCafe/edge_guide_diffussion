"""Canny 邊緣擷取。

本檔案使用繁體中文說明研究程式的用途。"""
import cv2
import numpy as np
import torch

def canny_edge(image_tensor, low=100, high=200):
    """canny_edge：執行此模組的對應運算或資料處理。"""
    if image_tensor.ndim != 3 or image_tensor.shape[0] not in (1, 3):
        raise ValueError(f'Expected CHW tensor with 1 or 3 channels, got {tuple(image_tensor.shape)}')
    pixels = ((image_tensor.detach().cpu().float() + 1) * 127.5).clamp(0, 255).byte()
    if pixels.shape[0] == 1:
        gray = pixels[0].numpy()
    else:
        rgb = pixels.permute(1, 2, 0).numpy()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(np.ascontiguousarray(gray), low, high)
    return torch.from_numpy(edge.copy()).unsqueeze(0).float() / 255
