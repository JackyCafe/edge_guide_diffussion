"""infer。

本檔案使用繁體中文說明研究程式的用途。"""
import argparse
import torch
from PIL import Image
from src.data.transforms import image_tensor, detect_channels
from src.data.masks import rectangle_mask
from src.models.edge_generator import EdgeGenerator
from src.models.diffusion_unet import DiffusionUNet
from src.diffusion.schedule import DiffusionSchedule
from src.pipelines.inference import inpaint
from src.utils.config import load_config
from src.utils.checkpoint import load_checkpoint
from src.utils.visualization import save_rgb

def main():
    """main：執行此模組的對應運算或資料處理。"""
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True)
    p.add_argument('--stage1', required=True)
    p.add_argument('--stage2', required=True)
    p.add_argument('--config', default='configs/stage2.yaml')
    p.add_argument('--out', default='outputs/samples/inference.png')
    a = p.parse_args()
    cfg = load_config(a.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base = cfg['model']['base_channels']
    with Image.open(a.image) as im:
        channels = detect_channels(
            im) if cfg['dataset'].get('channels',
            'auto') == 'auto' else cfg['dataset']['channels'
        ]
        image = image_tensor(
            im,
            cfg['dataset']['image_size'],
            channels=channels).unsqueeze(0).to(device
        )
    g1 = EdgeGenerator(input_channels=channels + 2, base=base).to(device)
    g2 = DiffusionUNet(
        in_channels=2 * channels + 2,
        out_channels=channels,
        base=base,
        time_dim=cfg['model']['time_dim']).to(device
    )
    for model, path in [(g1, a.stage1), (g2, a.stage2)]:
        state = load_checkpoint(path, device)
        model.load_state_dict(state['G'] if 'G' in state else state)
        model.eval()
    mask = rectangle_mask(
        image.shape[-2],
        image.shape[-1],
        torch.Generator().manual_seed(cfg['seed'])).unsqueeze(0).to(device
    )
    d = cfg['diffusion']
    schedule = DiffusionSchedule(
        d['timesteps'],
        d['beta_start'],
        d['beta_end']).to(device
    )
    output, _ = inpaint(image, mask, g1, g2, schedule, d['sample_steps'], cfg['seed'])
    save_rgb(output, a.out)
    print(a.out)
if __name__ == '__main__':
    main()
