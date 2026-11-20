"""CSV 指標紀錄。

本檔案使用繁體中文說明研究程式的用途。"""
import csv
from pathlib import Path

def append_metrics(path, record):
    """append_metrics：執行此模組的對應運算或資料處理。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(record))
        if not exists:
            writer.writeheader()
        writer.writerow(record)
