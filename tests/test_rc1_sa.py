"""
RC1：驗證 SA Phase 遮罩、學習率邊界及狀態恢復。
這支程式是你的 Edge-Guided Diffusion v2.2 RC1 的 Simulated Annealing（SA，模擬退火）單元測試。

它主要驗證 Stage 2 的 GAN/FM 啟用時間、SA 搜尋參數限制，以及中斷後能否恢復相同的隨機搜尋狀態。
"""

import pytest

from src.strategies.phase import active_sa_keys, stage2_phase
from src.strategies.simulated_annealing import SimulatedAnnealing


def test_stage2_phase_gan_gate():
    """Epoch 1–150 不啟用 GAN，151–170 漸進增加。"""
    assert stage2_phase(149) == 0.0
    assert 0.0 < stage2_phase(150) < 1.0
    assert stage2_phase(169) == pytest.approx(1.0)
    assert stage2_phase(170) == 1.0


def test_sa_active_parameters():
    """在 Epoch 150 前，SA 不得擾動 Adv、FM、D2 LR。"""
    before = active_sa_keys(stage=2, epoch=149)
    after = active_sa_keys(stage=2, epoch=150)
    assert not {"adv", "fm", "d_lr"}.intersection(before)
    assert {"adv", "fm", "d_lr"}.issubset(after)


def test_sa_candidate_respects_active_mask_and_bounds():
    """候選只變動允許的參數，且 LR 維持正值。"""
    original = {"hole": 0.7, "adv": 0.005, "g_lr": 1e-4}
    bounds = {
        "hole": (0.2, 1.5),
        "adv": (0.0, 0.01),
        "g_lr": (1e-6, 2e-4),
    }
    sa = SimulatedAnnealing(original, bounds, seed=123)
    for _ in range(20):
        candidate = sa.propose(active_keys={"hole", "g_lr"})
        assert candidate["adv"] == original["adv"]
        assert 0.2 <= candidate["hole"] <= 1.5
        assert 1e-6 <= candidate["g_lr"] <= 2e-4


def test_sa_restore_rng():
    """儲存後恢復，下一個候選必須完全一致。"""
    weights = {"hole": 0.7}
    bounds = {"hole": (0.2, 1.5)}
    sa = SimulatedAnnealing(weights, bounds, seed=123)
    state = sa.state_dict()
    expected = sa.propose()
    sa.load_state_dict(state)
    assert sa.propose() == expected
