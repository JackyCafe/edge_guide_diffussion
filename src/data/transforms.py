"""影像通道偵測與 [-1, 1] 張量正規化。"""

import numpy as np
import torch
from PIL import Image


def detect_channels(image: Image.Image) -> int:
    """依原始 PIL 模式判斷灰階 1ch 或彩色 3ch。"""
    if image.mode in ("1", "L", "LA"):
        return 1

    if image.mode in ("RGB", "RGBA", "P", "CMYK", "YCbCr", "HSV"):
        if image.mode == "P":
            palette = image.getpalette()
            if palette is not None:
                colors = range(0, min(len(palette), 768) - 2, 3)
                grayscale_palette = all(
                    palette[i] == palette[i + 1] == palette[i + 2]
                    for i in colors
                )
                if grayscale_palette:
                    return 1
        return 3

    raise ValueError(
        f"不支援的影像模式：{image.mode!r}，請轉為 8-bit 灰階或 RGB。"
    )


def image_tensor(image: Image.Image, size: int, channels=None):
    """縮放影像並輸出 [C, H, W]，數值範圍固定 [-1, 1]。"""
    if channels is None or channels == "auto":
        channels = detect_channels(image)
    if channels not in (1, 3):
        raise ValueError(f"channels 只能是 1、3 或 auto，收到 {channels!r}")

    mode = "L" if channels == 1 else "RGB"
    resized = image.convert(mode).resize(
        (size, size),
        Image.Resampling.BICUBIC,
    )
    array = np.asarray(resized).copy()

    if channels == 1:
        tensor = torch.from_numpy(array).unsqueeze(0)
    else:
        tensor = torch.from_numpy(array).permute(2, 0, 1)

    return tensor.float() / 127.5 - 1.0


def rgb_tensor(image: Image.Image, size: int):
    """明確指定 RGB 三通道的相容介面。"""
    return image_tensor(image, size, channels=3)


def gray_tensor(image: Image.Image, size: int):
    """明確指定灰階單通道的相容介面。"""
    return image_tensor(image, size, channels=1)
