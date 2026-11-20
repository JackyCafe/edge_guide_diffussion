"""YAML 設定載入與合併。

本檔案使用繁體中文說明研究程式的用途。"""
from pathlib import Path
import yaml

def merge(a, b):
    """merge：執行此模組的對應運算或資料處理。"""
    result = dict(a)
    for k, v in b.items():
        result[
            k] = merge(result[k],
            v) if k in result and isinstance(result[k],
            dict) and isinstance(v,
            dict
        ) else v
    return result

def load_config(path):
    """load_config：執行此模組的對應運算或資料處理。"""
    path = Path(path).resolve()
    with path.open(encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    parent = data.pop('include', None)
    return merge(load_config(path.parent / parent), data) if parent else data
