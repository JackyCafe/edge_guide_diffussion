"""V19-inspired dashboard with dynamic plots and five latest visual previews."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

def update_dashboard(csv_path, png_path, stage):
    """update_dashboard：執行此模組的對應運算或資料處理。"""
    csv_path, png_path = (Path(csv_path), Path(png_path))
    if not csv_path.exists():
        return
    with csv_path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    stage1 = '1' in str(stage)
    fields = [
        'train_loss',
        'precision',
        'recall',
        'f1',
        'tolerance_f1',
        'iou',
        'weight_wbce',
        'weight_dice',
        'weight_fm',
        'weight_adv',
        'g_lr',
        'd_lr'] if stage1 else ['train_loss',
        'psnr',
        'ssim',
        'mae',
        'boundary_mae',
        'weight_noise',
        'weight_hole',
        'weight_boundary',
        'weight_vgg',
        'weight_adv',
        'weight_fm',
        'g_lr'
    ]
    previews = sorted((png_path.parent / 'visuals').glob('epoch_*_preview.png'))[-5:]
    fig = plt.figure(figsize=(17, 15 + 2.3 * len(previews)), constrained_layout=True)
    gs = fig.add_gridspec(
        6 + len(previews),
        3,
        height_ratios=[1, 1, 1, 1, 1.2, 0.15] + [1.2] * len(previews)
    )
    epochs = [int(row['epoch']) + 1 for row in rows]
    for i, field in enumerate(fields):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        data = [
            (ep, float(row[field])) for ep,
            row in zip(epochs, rows) if row.get(field) not in (None, '', 'nan')
        ]
        if data:
            xs, ys = zip(*data)
            ax.plot(xs, ys, linewidth=1.8)
            ax.scatter(xs[-1], ys[-1], s=18)
        ax.set_title(field.replace('_', ' ').title(), fontsize=10)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=8)
    ax = fig.add_subplot(gs[4, :])
    keys = ['f1', 'tolerance_f1', 'iou'] if stage1 else ['psnr', 'ssim', 'train_loss']
    for key in keys:
        pairs = [
            (ep, float(row[key])) for ep,
            row in zip(epochs, rows) if row.get(key) not in ('', None)
        ]
        if pairs:
            x, y = zip(*pairs)
            y0, y1 = (min(y), max(y))
            ax.plot(x, [(z - y0) / (y1 - y0) if y1 > y0 else 0.5 for z in y], label=key)
    ax.set_title('Performance Analysis (normalized trends)')
    ax.grid(alpha=0.2)
    ax.legend(loc='best', fontsize=8)
    divider = fig.add_subplot(gs[5, :])
    divider.axis('off')
    divider.text(
        0.5,
        0.5,
        'LATEST EPOCH VISUAL FEEDBACK',
        ha='center',
        va='center',
        fontsize=13
    )
    for idx, path in enumerate(previews):
        preview = fig.add_subplot(gs[6 + idx, :])
        with Image.open(path) as im:
            preview.imshow(im.convert('RGB'))
        preview.set_title(path.stem, fontsize=10)
        preview.axis('off')
    fig.suptitle(f'Edge-Diffusion {stage} Multi-Visual Monitor', fontsize=17)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
