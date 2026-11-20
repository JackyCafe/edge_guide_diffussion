"""GAN/FM 分階段啟用規則。

本檔案使用繁體中文說明研究程式的用途。"""
import math

def stage2_phase(epoch, gan_start=150, gan_warmup=20):
    """epoch is zero-based; epochs 1..150 no GAN, 151..170 ramp."""
    if epoch < gan_start:
        return 0.0
    if epoch >= gan_start + gan_warmup:
        return 1.0
    progress = (epoch + 1 - gan_start) / gan_warmup
    return 0.5 * (1 - math.cos(math.pi * progress))

def effective_weights(stage, epoch, weights, gan_start=150, gan_warmup=20):
    """effective_weights：執行此模組的對應運算或資料處理。"""
    result = dict(weights)
    if stage == 2:
        gate = stage2_phase(epoch, gan_start, gan_warmup)
        for key in ('adv', 'fm'):
            result[key] = result.get(key, 0.0) * gate
    return result

def active_sa_keys(stage, epoch):
    """active_sa_keys：執行此模組的對應運算或資料處理。"""
    if stage == 1:
        return ('wbce', 'dice', 'fm', 'adv', 'g_lr', 'd_lr')
    keys = ('hole', 'boundary', 'vgg', 'g_lr')
    if epoch >= 150:
        keys += ('adv', 'fm', 'd_lr')
    return keys
