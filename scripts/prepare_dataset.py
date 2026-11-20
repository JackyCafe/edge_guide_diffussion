"""prepare_dataset。

本檔案使用繁體中文說明研究程式的用途。"""
import argparse, csv, random
from pathlib import Path

def main():
    """main：執行此模組的對應運算或資料處理。"""
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--out', default='datasets/manifests')
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    paths = sorted((str(x.resolve()) for x in Path(a.root).rglob('*') if x.suffix.lower() in {'.png', '.jpg','.pgm', '.jpeg', '.webp', '.bmp'}))
    if len(paths) < 3:
        raise ValueError('Need at least 3 images')
    random.Random(a.seed).shuffle(paths)
    n = len(paths)
    n_test = max(1, int(n * 0.1))
    n_val = max(1, int(n * 0.1))
    parts = {
        'test': paths[:n_test],
        'val': paths[n_test:n_test + n_val],
        'train': paths[n_test + n_val:]
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, files in parts.items():
        with (out / f'{name}.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['path'])
            writer.writeheader()
            writer.writerows(({'path': x} for x in files))
        print(name, len(files))
if __name__ == '__main__':
    main()
