"""test_dataset。

本檔案使用繁體中文說明研究程式的用途。"""
import torch
from src.data.masks import rectangle_mask
from src.data.edges import canny_edge

def test_mask():
    """test_mask：執行此模組的對應運算或資料處理。"""
    m = rectangle_mask(64, 64, torch.Generator().manual_seed(42))
    assert m.shape == (1, 64, 64)
    assert set(m.unique().tolist()) == {0.0, 1.0}

def test_edge():
    """test_edge：執行此模組的對應運算或資料處理。"""
    x = torch.zeros(3, 64, 64)
    e = canny_edge(x)
    assert e.shape == (1, 64, 64)
    assert e.min() >= 0 and e.max() <= 1
