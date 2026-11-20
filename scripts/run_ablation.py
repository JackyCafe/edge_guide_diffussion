"""消融實驗執行入口。

本檔案使用繁體中文說明研究程式的用途。"""
import argparse, time
from pathlib import Path
from scripts.common import setup
from src.trainers.stage1_trainer import Stage1Trainer
from src.trainers.stage2_trainer import Stage2Trainer
from src.trainers.candidate_evaluator import CandidateEvaluator
from src.strategies.simulated_annealing import SimulatedAnnealing
from src.utils.checkpoint import save_checkpoint
from src.utils.logging import append_metrics

def main():
    """main：執行此模組的對應運算或資料處理。"""
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--run-dir', default='outputs/experiments/sa')
    a = p.parse_args()
    raise NotImplementedError('v2.2 SA candidate evaluation (200 optimizer steps, active mask, DDIM20 subset) is not yet implemented. Do not run the legacy epoch-branch SA experiment.')
    cfg, device, train, val = setup(a.config)
    run = Path(a.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    if cfg['strategy']['kind'] != 'sa':
        raise ValueError('Requires SA config')
    trainer = Stage1Trainer(
        cfg,
        train,
        val,
        device,
        run) if cfg['stage'] == 1 else Stage2Trainer(cfg,
        train,
        val,
        device,
        run
    )
    sa = SimulatedAnnealing(
        cfg['strategy']['weights'],
        cfg['strategy']['bounds'],
        cfg['sa']['temperature'],
        cfg['sa']['cooling'],
        cfg['sa']['min_temperature'],
        cfg['seed']
    )
    evaluator = CandidateEvaluator(trainer, 'f1' if cfg['stage'] == 1 else 'score')
    k = cfg['sa']['branch_epochs']
    max_candidates = min(cfg['sa']['candidates'], cfg['training']['epochs'] // k)
    for i in range(max_candidates):
        start = i * k
        candidate = sa.propose()
        tick = time.monotonic()
        old, old_state, new, new_state = evaluator.compare(
            start,
            k,
            sa.weights,
            candidate
        )
        accepted = sa.decide(candidate, old, new)
        trainer.load_state_dict(new_state if accepted else old_state)
        append_metrics(
            run / 'sa_history.csv',
            {'step': i, 'epoch': start + k, 'current_score': old, 'candidate_score': new, 'accepted': accepted, 'temperature': sa.temperature, 'gpu_wall_seconds': time.monotonic() - tick, 'weights': repr(sa.weights)}
        )
        save_checkpoint(
            run / 'sa_latest.pth',
            step=i,
            trainer=trainer.state_dict(),
            sa=sa.state_dict()
        )
        print('SA', i, 'accepted', accepted, 'old', old, 'new', new)
    remaining = max_candidates * k
    if remaining < cfg['training']['epochs']:
        trainer.train_epochs(remaining, cfg['training']['epochs'], sa.weights)
if __name__ == '__main__':
    main()
