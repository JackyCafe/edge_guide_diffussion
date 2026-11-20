"""evaluate。

本檔案使用繁體中文說明研究程式的用途。"""
import argparse
from scripts.common import setup
from src.trainers.stage1_trainer import Stage1Trainer
from src.trainers.stage2_trainer import Stage2Trainer
from src.evaluation.evaluator import evaluate_stage1, evaluate_stage2
from src.utils.checkpoint import load_checkpoint

def main():
    """main：執行此模組的對應運算或資料處理。"""
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    a = p.parse_args()
    cfg, device, train, val = setup(a.config)
    stage = cfg['stage']
    trainer = (
        Stage1Trainer if stage == 1 else Stage2Trainer)(cfg,
        train,
        val,
        device,
        'outputs/eval'
    )
    state = load_checkpoint(a.checkpoint, device)
    trainer.G.load_state_dict(state['G'] if 'G' in state else state)
    print(evaluate_stage1(trainer.G, val, device) if stage == 1 else evaluate_stage2(trainer.G, trainer.G1, trainer.schedule, val, device, cfg['diffusion']['sample_steps']))
if __name__ == '__main__':
    main()
