"""資料集載入、遮罩與邊緣條件建構。

本檔案使用繁體中文說明研究程式的用途。"""
import csv
import hashlib
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset
from .transforms import image_tensor, detect_channels
from .edges import canny_edge
from .masks import rectangle_mask

class InpaintingDataset(Dataset):
    """InpaintingDataset：封裝對應模型或研究流程。"""

    def __init__(
        self,
        manifest,
        size=256,
        seed=42,
        train=True,
        min_frac=0.2,
        max_frac=0.45,
        channels='auto'
    ):
        """初始化元件並儲存必要的模型參數與執行狀態。"""
        manifest = Path(manifest)
        with manifest.open(newline='', encoding='utf-8') as f:
            self.paths = [r['path'] for r in csv.DictReader(f)]
        if not self.paths:
            raise ValueError(f'Empty manifest: {manifest}')
        self.size = size
        self.seed = seed
        self.train = train
        self.epoch = 0
        self.min_frac = min_frac
        self.max_frac = max_frac
        if channels == 'auto':
            modes = {}
            for path in self.paths:
                with Image.open(path) as im:
                    c = detect_channels(im)
                modes.setdefault(c, []).append(path)
            if len(modes) != 1:
                examples = {k: v[:2] for k, v in modes.items()}
                raise ValueError(f'Mixed grayscale/RGB images in one manifest. Use separate datasets/runs or explicitly set channels=1/3. Examples: {examples}')
            self.channels = next(iter(modes))
        else:
            if channels not in (1, 3):
                raise ValueError("channels must be 'auto', 1 or 3")
            self.channels = channels

    def set_epoch(self, epoch):
        """set_epoch：執行此模組的對應運算或資料處理。"""
        self.epoch = epoch

    def __len__(self):
        """__len__：執行此模組的對應運算或資料處理。"""
        return len(self.paths)

    def __getitem__(self, index):
        """__getitem__：執行此模組的對應運算或資料處理。"""
        path = self.paths[index]
        with Image.open(path) as im:
            image = image_tensor(im, self.size, channels=self.channels)
        edge = canny_edge(image)
        stable = int.from_bytes(
            hashlib.sha256(path.encode()).digest()[:8],
            'little'
        ) % 2 ** 31
        generator = torch.Generator().manual_seed(self.seed + stable + (self.epoch * 1000003 if self.train else 0))
        mask = rectangle_mask(
            self.size,
            self.size,
            generator,
            self.min_frac,
            self.max_frac
        )
        masked = image * (1 - mask) - mask
        return {
            'image': image,
            'mask': mask,
            'edge': edge,
            'masked_image': masked,
            'known_edge': edge * (1 - mask),
            'sample_id': path
        }
