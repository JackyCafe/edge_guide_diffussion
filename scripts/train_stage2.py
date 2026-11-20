"""Stage 2 訓練命令列入口。

本檔案使用繁體中文說明研究程式的用途。"""
from src.utils.run_metadata import save_run_metadata
import argparse
from scripts.common import setup
from src.trainers.stage2_trainer import Stage2Trainer

def main():
    """main：執行此模組的對應運算或資料處理。"""
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/stage2.yaml')
    p.add_argument('--run-dir', default='outputs/experiments/stage2_fixed')
    a = p.parse_args()
    cfg, device, train, val = setup(a.config)
    if cfg['strategy']['kind'] == 'sa':
        raise ValueError('For SA use scripts/run_ablation.py')
    save_run_metadata(a.run_dir, cfg)
    trainer = Stage2Trainer(cfg, train, val, device, a.run_dir)
    trainer.train_epochs(0, cfg['training']['epochs'], cfg['strategy']['weights'])
if __name__ == '__main__':
    main()
