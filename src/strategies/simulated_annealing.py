"""模擬退火搜尋器：提供有界候選、Active Mask 與接受規則。"""

import copy
import math
import random


class SimulatedAnnealing:
    """依照驗證目標最大化的 Metropolis 準則更新候選參數。"""

    def __init__(
        self,
        weights,
        bounds,
        temperature=1.0,
        cooling=0.9,
        min_temperature=0.01,
        seed=42,
        score_scale=1.0,
    ):
        """初始化權重、搜尋邊界、退火溫度與獨立亂數產生器。"""
        self.weights = copy.deepcopy(weights)
        self.bounds = dict(bounds)
        self.temperature = float(temperature)
        self.cooling = float(cooling)
        self.min_temperature = float(min_temperature)
        self.score_scale = float(score_scale)
        self.rng = random.Random(seed)
        self.best_score = float("-inf")
        self.best_weights = copy.deepcopy(weights)

        if not 0.0 < self.cooling <= 1.0:
            raise ValueError("cooling 必須介於 (0, 1]。")
        if self.score_scale <= 0.0:
            raise ValueError("score_scale 必須大於零。")

    def propose(self, active_keys=None):
        """只擾動當前 Phase 開放的參數，避免搜尋尚未啟用的 GAN。"""
        active = set(active_keys) if active_keys is not None else set(self.bounds)
        candidates = sorted(active.intersection(self.bounds))
        if not candidates:
            raise ValueError("沒有可搜尋的參數；請檢查 Active Parameter Mask。")

        candidate = copy.deepcopy(self.weights)
        name = self.rng.choice(candidates)
        lower, upper = self.bounds[name]
        current = candidate[name]

        # 學習率跨數量級變動，因此採用對數空間擾動。
        if name.endswith("_lr"):
            if lower <= 0 or upper <= 0:
                raise ValueError("LR 搜尋邊界必須大於零。")
            current_log = math.log(max(lower, min(upper, current)))
            span = math.log(upper) - math.log(lower)
            proposal = math.exp(current_log + self.rng.gauss(0, span * 0.1))
        else:
            span = upper - lower
            proposal = current + self.rng.gauss(0, span * 0.1)

        candidate[name] = max(lower, min(upper, proposal))
        return candidate

    def decide(self, candidate, current_score, candidate_score):
        """候選較差時仍可依溫度接受，避免過早停在局部最佳解。"""
        delta = (candidate_score - current_score) / self.score_scale
        probability = math.exp(min(0.0, delta / max(self.temperature, 1e-12)))
        accepted = delta >= 0.0 or self.rng.random() < probability

        if accepted:
            self.weights = copy.deepcopy(candidate)
        if candidate_score > self.best_score:
            self.best_score = candidate_score
            self.best_weights = copy.deepcopy(candidate)
        if current_score > self.best_score:
            self.best_score = current_score

        self.temperature = max(
            self.min_temperature,
            self.temperature * self.cooling,
        )
        return accepted

    def state_dict(self):
        """保存完整 SA 狀態，確保中斷後可以重現搜尋順序。"""
        return {
            "weights": copy.deepcopy(self.weights),
            "temperature": self.temperature,
            "best_score": self.best_score,
            "best_weights": copy.deepcopy(self.best_weights),
            "rng": self.rng.getstate(),
        }

    def load_state_dict(self, state):
        """恢復候選權重、溫度與亂數產生器。"""
        self.weights = copy.deepcopy(state["weights"])
        self.temperature = state["temperature"]
        self.best_score = state["best_score"]
        self.best_weights = copy.deepcopy(state["best_weights"])
        self.rng.setstate(state["rng"])
