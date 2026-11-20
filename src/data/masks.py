"""修補遮罩生成。

本檔案使用繁體中文說明研究程式的用途。"""
import torch

def rectangle_mask(height, width, generator=None, min_frac=0.2, max_frac=0.45):
    """rectangle_mask：執行此模組的對應運算或資料處理。"""

    def uniform(a, b):
        """uniform：執行此模組的對應運算或資料處理。"""
        return a + (b - a) * torch.rand((), generator=generator).item()
    mh = max(1, int(height * uniform(min_frac, max_frac)))
    mw = max(1, int(width * uniform(min_frac, max_frac)))
    top = int(torch.randint(0, height - mh + 1, (1,), generator=generator).item())
    left = int(torch.randint(0, width - mw + 1, (1,), generator=generator).item())
    m = torch.zeros(1, height, width)
    m[:, top:top + mh, left:left + mw] = 1
    return m
