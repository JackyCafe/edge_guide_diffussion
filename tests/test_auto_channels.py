"""test_auto_channels。

本檔案使用繁體中文說明研究程式的用途。"""
import csv
from PIL import Image
import numpy as np
import pytest
import torch
from src.data.dataset import InpaintingDataset
from src.data.edges import canny_edge
from src.models.edge_generator import EdgeGenerator
from src.models.diffusion_unet import DiffusionUNet
from src.diffusion.schedule import DiffusionSchedule
from src.diffusion.sampler import ddim_sample

@pytest.mark.parametrize('mode,channels', [('L', 1), ('RGB', 3)])
def test_auto_image_and_models(tmp_path, mode, channels):
    """test_auto_image_and_models：執行此模組的對應運算或資料處理。"""
    img = tmp_path / f'{mode}.png'
    Image.fromarray(
        np.full((32, 32) if channels == 1 else (32, 32, 3), 128, dtype=np.uint8),
        mode=mode).save(img
    )
    manifest = tmp_path / 'train.csv'
    with manifest.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path'])
        writer.writerow([img])
    ds = InpaintingDataset(manifest, size=32)
    sample = ds[0]
    assert ds.channels == channels
    assert sample['image'].shape == (channels, 32, 32)
    assert canny_edge(sample['image']).shape == (1, 32, 32)
    g1 = EdgeGenerator(input_channels=channels + 2, base=8, residual_blocks=1)
    g2 = DiffusionUNet(
        in_channels=2 * channels + 2,
        out_channels=channels,
        base=8,
        time_dim=16
    )
    mask = sample['mask'].unsqueeze(0)
    masked = sample['masked_image'].unsqueeze(0)
    known = sample['known_edge'].unsqueeze(0)
    edge = g1(torch.cat([masked, mask, known], 1))
    assert edge.shape == (1, 1, 32, 32)
    cond = torch.cat([masked, mask, edge.sigmoid()], 1)
    sched = DiffusionSchedule(timesteps=10, beta_start=0.0001, beta_end=0.02)
    with torch.no_grad():
        output = ddim_sample(g2, sched, cond, steps=2)
    assert output.shape == (1, channels, 32, 32)

def test_mixed_manifest_rejected(tmp_path):
    """test_mixed_manifest_rejected：執行此模組的對應運算或資料處理。"""
    paths = []
    for mode in ('L', 'RGB'):
        p = tmp_path / f'{mode}.png'
        Image.new(mode, (32, 32)).save(p)
        paths.append(p)
    manifest = tmp_path / 'mixed.csv'
    with manifest.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path'])
        writer.writerows([[p] for p in paths])
    with pytest.raises(ValueError, match='Mixed grayscale/RGB'):
        InpaintingDataset(manifest, size=32)
    assert InpaintingDataset(manifest, size=32, channels=1).channels == 1
