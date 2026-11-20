"""模型與最佳權重儲存。

本檔案使用繁體中文說明研究程式的用途。"""
import os
from pathlib import Path
import torch

def save_checkpoint(path, **states):
    """save_checkpoint：執行此模組的對應運算或資料處理。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(states, temporary)
    os.replace(temporary, path)

def load_checkpoint(path, device='cpu'):
    """load_checkpoint：執行此模組的對應運算或資料處理。"""
    return torch.load(path, map_location=device, weights_only=False)
