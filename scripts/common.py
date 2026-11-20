"""common。

本檔案使用繁體中文說明研究程式的用途。"""
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.data.dataset import InpaintingDataset
from src.utils.config import load_config
from src.utils.seed import seed_everything

def setup(config):
    """setup：執行此模組的對應運算或資料處理。"""
    cfg = load_config(config)
    seed_everything(cfg['seed'])
    device = torch.device('cuda' if cfg['training']['device'] == 'auto' and torch.cuda.is_available() else 'cpu' if cfg['training']['device'] == 'auto' else cfg['training']['device'])
    ds = cfg['dataset']
    manifest = Path(ds['manifest_dir'])
    kwargs = dict(
        size=ds['image_size'],
        seed=cfg['seed'],
        min_frac=ds['mask_min'],
        max_frac=ds['mask_max'],
        channels=ds.get('channels', 'auto')
    )
    train = InpaintingDataset(manifest / 'train.csv', train=True, **kwargs)
    val = InpaintingDataset(manifest / 'val.csv', train=False, **kwargs)
    if train.channels != val.channels:
        raise ValueError(f'Train/val channels differ: train={train.channels}, val={val.channels}. Use matching image types.')
    cfg['dataset']['resolved_channels'] = train.channels
    print(f'[Dataset] detected {train.channels} channel(s)')
    loader = lambda dataset, shuffle: DataLoader(
        dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=shuffle,
        num_workers=cfg['training']['num_workers'],
        pin_memory=device.type == 'cuda'
    )
    return (cfg, device, loader(train, True), loader(val, False))
