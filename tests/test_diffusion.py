"""
test_diffusion。

這支 test_diffusion.py 是用來驗證 Diffusion 正向加噪與反推原始影像的數學公式是否正確。程式邏輯基本上正確，可以使用 pytest 執行。
"""
import torch
from src.diffusion.schedule import DiffusionSchedule
from src.diffusion.forward import noise_images, predict_x0

def test_inverse():
    """驗證使用真實雜訊能正確反推原始影像。"""

    # 固定隨機種子，確保實驗結果可重現
    torch.manual_seed(42)

    # 建立 50 個時間步的 Diffusion Schedule
    schedule = DiffusionSchedule(timesteps=50)

    # 建立兩張 RGB 測試影像
    original = torch.randn(2, 3, 16, 16)

    # 分別指定不同的時間步
    timesteps = torch.tensor([1, 49])

    # 正向加噪，取得加噪影像與真實雜訊
    noisy_image, noise = noise_images(
        original,
        timesteps,
        schedule,
    )

    # 利用真實雜訊反推原始影像
    reconstructed = predict_x0(
        noisy_image,
        timesteps,
        noise,
        schedule,
    )

    # 計算最大絕對誤差
    max_error = (
        reconstructed - original
    ).abs().max().item()

    print(f"最大還原誤差：{max_error:.8f}")

    # 驗證正向與反向公式是否一致
    assert torch.allclose(
        reconstructed,
        original,
        atol=1e-4,
    )
