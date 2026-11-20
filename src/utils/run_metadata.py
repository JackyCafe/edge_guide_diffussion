"""run_metadata。

本檔案使用繁體中文說明研究程式的用途。"""
import json
import platform
import subprocess
from pathlib import Path
import torch
import yaml

def save_run_metadata(run_dir, cfg):
    """save_run_metadata：執行此模組的對應運算或資料處理。"""
    folder = Path(run_dir)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'resolved_config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    try:
        git = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL,
            text=True).strip(
        )
    except (OSError, subprocess.CalledProcessError):
        git = 'unknown'
    info = {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'git_commit': git,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'seed': cfg['seed'],
        'channels': cfg['dataset']['resolved_channels']
    }
    (folder / 'audit.json').write_text(json.dumps(info, ensure_ascii=False, indent=2))
