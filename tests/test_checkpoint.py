"""test_checkpoint。

本檔案使用繁體中文說明研究程式的用途。"""
import torch
from src.utils.checkpoint import save_checkpoint, load_checkpoint

def test_checkpoint(tmp_path):
    """test_checkpoint：執行此模組的對應運算或資料處理。"""
    p = tmp_path / 'test.pth'
    save_checkpoint(p, epoch=3, value=torch.tensor(5))
    assert load_checkpoint(p)['epoch'] == 3
