"""candidate_evaluator。

本檔案使用繁體中文說明研究程式的用途。"""
import copy
import random
import numpy as np
import torch

def capture_rng():
    """capture_rng：執行此模組的對應運算或資料處理。"""
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }

def restore_rng(s):
    """restore_rng：執行此模組的對應運算或資料處理。"""
    random.setstate(s['python'])
    np.random.set_state(s['numpy'])
    torch.set_rng_state(s['torch'])
    if s['cuda'] is not None:
        torch.cuda.set_rng_state_all(s['cuda'])

class CandidateEvaluator:
    """CandidateEvaluator：封裝對應模型或研究流程。"""

    def __init__(self, trainer, score_key):
        """初始化元件並儲存必要的模型參數與執行狀態。"""
        self.trainer = trainer
        self.score_key = score_key

    def compare(self, start, branch_epochs, current, candidate):
        """compare：執行此模組的對應運算或資料處理。"""
        base = copy.deepcopy(self.trainer.state_dict())
        rng = capture_rng()

        def run(weights):
            """run：執行此模組的對應運算或資料處理。"""
            self.trainer.load_state_dict(copy.deepcopy(base))
            restore_rng(rng)
            metrics = self.trainer.train_epochs(
                start,
                start + branch_epochs,
                weights,
                save_artifacts=False
            )
            return (metrics[self.score_key], copy.deepcopy(self.trainer.state_dict()))
        score_current, state_current = run(current)
        score_candidate, state_candidate = run(candidate)
        return (score_current, state_current, score_candidate, state_candidate)
