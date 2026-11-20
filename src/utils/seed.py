"""seed。

本檔案使用繁體中文說明研究程式的用途。"""
import random
import numpy as np
import torch

def seed_everything(seed):
    """seed_everything：執行此模組的對應運算或資料處理。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
