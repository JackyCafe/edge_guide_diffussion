"""Stage 2 分階段訓練、驗證與最佳 PSNR 管理。

本檔案使用繁體中文說明研究程式的用途。"""
from pathlib import Path
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
from src.models.diffusion_unet import DiffusionUNet
from src.models.image_discriminator import ImageDiscriminator
from src.models.edge_generator import EdgeGenerator
from src.diffusion.schedule import DiffusionSchedule
from src.diffusion.forward import noise_images, predict_x0
from src.losses.diffusion_loss import diffusion_losses
from src.losses.perceptual import VGGPerceptual
from src.losses.adversarial import hinge_d, hinge_g
from src.strategies.lr_scheduler import learning_rate
from src.evaluation.evaluator import evaluate_stage2
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.logging import append_metrics
from src.utils.dashboard import update_dashboard
from src.utils.preview import save_preview
from src.diffusion.sampler import ddim_sample
from src.strategies.phase import effective_weights
from src.strategies.cosine import CosineWeights

class Stage2Trainer:
    """管理 Stage 2 Diffusion 訓練、後期 GAN/FM 與最佳 PSNR。"""

    def __init__(self, cfg, train_loader, val_loader, device, run_dir):
        """初始化 G2、D2、凍結的 G1、擴散排程與優化器。"""
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        base = cfg['model']['base_channels']
        channels = cfg['dataset']['resolved_channels']
        self.G = DiffusionUNet(
            in_channels=2 * channels + 2,
            out_channels=channels,
            base=base,
            time_dim=cfg['model']['time_dim']).to(device
        )
        self.D = ImageDiscriminator(in_channels=channels, base=base).to(device)
        self.G1 = EdgeGenerator(input_channels=channels + 2, base=base).to(device)
        ckpt = load_checkpoint(cfg['checkpoint']['stage1'], device)
        self.G1.load_state_dict(ckpt['G'] if 'G' in ckpt else ckpt)
        self.G1.eval().requires_grad_(False)
        self.schedule = DiffusionSchedule(
            timesteps=cfg['diffusion']['timesteps'],
            beta_start=cfg['diffusion']['beta_start'],
            beta_end=cfg['diffusion']['beta_end']).to(device
        )
        self.opt_g = AdamW(
            self.G.parameters(),
            lr=cfg['training']['lr'],
            weight_decay=0.0001
        )
        self.opt_d = AdamW(self.D.parameters(), lr=cfg['training']['lr'] * 0.2)
        self.vgg = VGGPerceptual(
            ).to(device) if cfg['loss'].get('vgg_enabled',
            False
        ) else None
        self.best = float('-inf')

    def state_dict(self):
        """state_dict：執行此模組的對應運算或資料處理。"""
        return {
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'opt_g': self.opt_g.state_dict(),
            'opt_d': self.opt_d.state_dict(),
            'best': self.best
        }

    def load_state_dict(self, s):
        """load_state_dict：執行此模組的對應運算或資料處理。"""
        self.G.load_state_dict(s['G'])
        self.D.load_state_dict(s['D'])
        self.opt_g.load_state_dict(s['opt_g'])
        self.opt_d.load_state_dict(s['opt_d'])
        self.best = s.get('best', float('-inf'))

    def train_epochs(self, start, end, weights, save_artifacts=True):
        """依設定執行訓練，並更新驗證指標及模型檢查點。"""
        accum = self.cfg['training'].get('accumulation', 1)
        # 依 Epoch 套用對應的 Loss 權重與學習率。
        for epoch in range(start, end):
            self.train_loader.dataset.set_epoch(epoch)
            self.G.train()
            self.D.train()
            if self.cfg['strategy']['kind'] == 'cosine':
                strategy = self.cfg['strategy']
                weights = CosineWeights(
                    strategy['start'],
                    strategy['end'],
                    self.cfg['training']['epochs'] - 1).get_weights(epoch
                )
            weights_epoch = effective_weights(2, epoch, weights)
            lr = learning_rate(
                epoch,
                self.cfg['training']['epochs'],
                self.cfg['training']['lr'],
                self.cfg['training']['warmup_epochs'],
                self.cfg['training']['min_lr_ratio']
            )
            if self.cfg['strategy']['kind'] == 'cosine':
                import math
                strategy = self.cfg['strategy']
                progress = epoch / max(1, self.cfg['training']['epochs'] - 1)
                lr = strategy['lr_end'] + 0.5 * (strategy['lr_start'] - strategy['lr_end']) * (1 + math.cos(math.pi * progress))
            for pg in self.opt_g.param_groups:
                pg['lr'] = lr
            for pg in self.opt_d.param_groups:
                pg['lr'] = lr * 0.2
            self.opt_g.zero_grad(set_to_none=True)
            epoch_loss_sum = 0.0
            epoch_samples = 0
            for idx, batch in enumerate(tqdm(self.train_loader, desc=f'S2 epoch {epoch}', leave=False)):
                image = batch['image'].to(self.device)
                masked = batch['masked_image'].to(self.device)
                mask = batch['mask'].to(self.device)
                known = batch['known_edge'].to(self.device)
                with torch.no_grad():
                    edge = known + self.G1(torch.cat([masked, mask, known], 1)).sigmoid() * mask
                condition = torch.cat([masked, mask, edge], 1)
                t = torch.randint(
                    0,
                    self.schedule.timesteps,
                    (image.shape[0],),
                    device=self.device
                )
                xt, noise = noise_images(image, t, self.schedule)
                eps = self.G(xt, t, condition)
                # 訓練期以單步 x0 計算像素損失，不等於完整 DDIM 取樣。
                x0 = predict_x0(xt, t, eps, self.schedule)
                parts = diffusion_losses(
                    eps,
                    noise,
                    x0,
                    image,
                    mask,
                    t,
                    self.schedule,
                    self.vgg if weights_epoch.get('vgg', 0) > 0 else None,
                    self.cfg['loss']['snr_gamma'],
                    self.cfg['loss']['snr_weighting'],
                    self.cfg['loss'].get('vgg_max_t', 500),
                    self.cfg['loss'].get('boundary_kernel', 11)
                )
                loss = sum((weights_epoch.get(k, 0) * v for k, v in parts.items()))
                use_gan = weights_epoch.get(
                    'adv',
                    0) > 0 or weights_epoch.get('fm',
                    0
                ) > 0
                if use_gan:
                    low = t < self.schedule.timesteps // 5
                    if low.any():
                        fake = x0[low]
                        real = image[low]
                        self.opt_d.zero_grad(set_to_none=True)
                        ld = hinge_d(self.D(real), self.D(fake.detach()))
                        ld.backward()
                        self.opt_d.step()
                        for p in self.D.parameters():
                            p.requires_grad_(False)
                        fake_score, fake_features = self.D(fake, return_features=True)
                        loss = loss + weights_epoch.get('adv', 0) * hinge_g(fake_score)
                        if weights_epoch.get('fm', 0) > 0:
                            with torch.no_grad():
                                _, real_features = self.D(real, return_features=True)
                            fm = sum(((a - b.detach()).abs().mean() for a, b in zip(fake_features, real_features))) / len(fake_features)
                            loss = loss + weights_epoch['fm'] * fm
                        for p in self.D.parameters():
                            p.requires_grad_(True)
                epoch_loss_sum += loss.detach().item() * image.shape[0]
                epoch_samples += image.shape[0]
                (loss / accum).backward()
                if (idx + 1) % accum == 0 or idx + 1 == len(self.train_loader):
                    nn.utils.clip_grad_norm_(self.G.parameters(), 1)
                    self.opt_g.step()
                    self.opt_g.zero_grad(set_to_none=True)
            # 最佳 PSNR 必須由完整驗證集的反向取樣結果決定。
            metrics = evaluate_stage2(
                self.G,
                self.G1,
                self.schedule,
                self.val_loader,
                self.device,
                self.cfg['diffusion']['sample_steps']
            )
            metrics['train_loss'] = epoch_loss_sum / max(epoch_samples, 1)
            metrics.update({f'weight_{k}': v for k, v in weights_epoch.items()})
            metrics['g_lr'] = self.opt_g.param_groups[0]['lr']
            metrics['d_lr'] = self.opt_d.param_groups[0]['lr']
            score = metrics['psnr']
            append_metrics(
                self.run_dir / 'metrics.csv',
                {'epoch': epoch, **metrics, 'score': score}
            ) if save_artifacts else None
            if score > self.best:
                self.best = score
                save_checkpoint(
                    self.run_dir / 'G2_best_psnr.pth',
                    epoch=epoch,
                    **self.state_dict()
                ) if save_artifacts else None
            if save_artifacts and (epoch + 1) % self.cfg['training']['checkpoint_every'] == 0:
                save_checkpoint(
                    self.run_dir / 'G2_latest.pth',
                    epoch=epoch,
                    **self.state_dict()
                )
            if save_artifacts and epoch == 149:
                save_checkpoint(
                    self.run_dir / 'G2_epoch_0150.pth',
                    epoch=epoch,
                    **self.state_dict()
                )
            if save_artifacts and epoch == 199:
                save_checkpoint(
                    self.run_dir / 'G2_epoch_0200.pth',
                    epoch=epoch,
                    **self.state_dict()
                )
            if save_artifacts:
                batch = next(iter(self.val_loader))
                with torch.no_grad():
                    original = batch['image'][:1].to(self.device)
                    masked = batch['masked_image'][:1].to(self.device)
                    mask = batch['mask'][:1].to(self.device)
                    known = batch['known_edge'][:1].to(self.device)
                    guide = known + self.G1(torch.cat([masked, mask, known], 1)).sigmoid() * mask
                    condition = torch.cat([masked, mask, guide], 1)
                    sampled = ddim_sample(
                        self.G,
                        self.schedule,
                        condition,
                        steps=self.cfg['diffusion']['sample_steps'],
                        known_image=original,
                        mask=mask,
                        seed=self.cfg['seed']
                    )
                save_preview(
                    self.run_dir / 'visuals' / f'epoch_{epoch + 1:04d}_preview.png',
                    [original[0], mask[0] * 2 - 1, guide[0] * 2 - 1, sampled[0]],
                    ['Original', 'Mask', 'G1 Edge Guide', 'DDIM Sampling']
                )
                update_dashboard(
                    self.run_dir / 'metrics.csv',
                    self.run_dir / 'dashboard_latest.png',
                    'Stage 2'
                )
            print('S2', epoch, metrics)
        return {'score': score, **metrics}
