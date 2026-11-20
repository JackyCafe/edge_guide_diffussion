"""Stage 1 訓練、驗證與 Checkpoint 管理。

本檔案使用繁體中文說明研究程式的用途。"""
from pathlib import Path
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from src.models.edge_generator import EdgeGenerator
from src.models.edge_discriminator import EdgeDiscriminator
from src.losses.edge_loss import edge_losses, feature_matching
from src.losses.adversarial import hinge_d, hinge_g
from src.strategies.lr_scheduler import learning_rate
from src.evaluation.evaluator import evaluate_stage1
from src.utils.checkpoint import save_checkpoint
from src.utils.logging import append_metrics
from src.utils.dashboard import update_dashboard
from src.utils.preview import save_preview
from src.strategies.cosine import CosineWeights

class Stage1Trainer:
    """管理 G1/D1 邊緣生成對抗訓練及最佳 F1 儲存。"""

    def __init__(self, cfg, train_loader, val_loader, device, run_dir):
        """初始化元件並儲存必要的模型參數與執行狀態。"""
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        base = cfg['model']['base_channels']
        channels = cfg['dataset']['resolved_channels']
        self.G = EdgeGenerator(input_channels=channels + 2, base=base).to(device)
        self.D = EdgeDiscriminator(base=base).to(device)
        lr = cfg['training']['lr']
        self.opt_g = Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.9))
        self.opt_d = Adam(
            self.D.parameters(),
            lr=lr * cfg['training'].get('d_lr_ratio', 0.2),
            betas=(0.5, 0.9)
        )
        self.best = -1

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
        self.best = s.get('best', -1)

    def train_epochs(self, start, end, weights, save_artifacts=True):
        """依設定執行訓練，並更新驗證指標及模型檢查點。"""
        # 每個 Epoch 記錄邊緣指標，並依 Validation F1 選擇最佳模型。
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
                pg['lr'] = lr * self.cfg['training'].get('d_lr_ratio', 0.2)
            epoch_loss_sum = 0.0
            epoch_samples = 0
            for batch in tqdm(self.train_loader, desc=f'S1 epoch {epoch}', leave=False):
                image = batch['masked_image'].to(self.device)
                mask = batch['mask'].to(self.device)
                edge = batch['edge'].to(self.device)
                known = batch['known_edge'].to(self.device)
                g_input = torch.cat([image, mask, known], 1)
                gray = image if image.shape[
                    1] == 1 else image[:,
                    0:1] * 0.299 + image[:,
                    1:2] * 0.587 + image[:,
                    2:3
                ] * 0.114
                logits = self.G(g_input)
                fake = known + logits.sigmoid() * mask
                real_input = torch.cat([gray, edge, mask], 1)
                fake_input = torch.cat([gray, fake, mask], 1)
                use_d = weights.get('adv', 0) > 0 or weights.get('fm', 0) > 0
                sigma = self.cfg[
                    'loss'].get('instance_noise',
                    0.05) * max(0,
                    1 - epoch / self.cfg['training']['epochs']
                )
                if sigma > 0 and use_d:
                    real_input = torch.cat(
                        [gray, (edge + torch.randn_like(edge) * sigma).clamp(0, 1), mask],
                        1
                    )
                    fake_input = torch.cat(
                        [gray, (fake + torch.randn_like(fake) * sigma).clamp(0, 1), mask],
                        1
                    )
                if use_d:
                    self.opt_d.zero_grad(set_to_none=True)
                    real_score, _ = self.D(real_input)
                    fake_score, _ = self.D(fake_input.detach())
                    ld = hinge_d(real_score, fake_score)
                    ld.backward()
                    self.opt_d.step()
                    for p in self.D.parameters():
                        p.requires_grad_(False)
                self.opt_g.zero_grad(set_to_none=True)
                parts = edge_losses(logits, edge, mask, self.cfg['loss']['pos_weight'])
                loss = weights.get(
                    'wbce',
                    1) * parts['wbce'] + weights.get('dice',
                    0) * parts['dice'
                ]
                if use_d:
                    score, ff = self.D(fake_input)
                    loss = loss + weights.get('adv', 0) * hinge_g(score)
                    if weights.get('fm', 0) > 0:
                        with torch.no_grad():
                            _, rf = self.D(real_input)
                        loss = loss + weights['fm'] * feature_matching(ff, rf)
                epoch_loss_sum += loss.detach().item() * image.shape[0]
                epoch_samples += image.shape[0]
                loss.backward()
                nn.utils.clip_grad_norm_(self.G.parameters(), 1)
                self.opt_g.step()
                if use_d:
                    for p in self.D.parameters():
                        p.requires_grad_(True)
            metrics = evaluate_stage1(self.G, self.val_loader, self.device)
            metrics['train_loss'] = epoch_loss_sum / max(epoch_samples, 1)
            metrics.update({f'weight_{k}': v for k, v in weights.items()})
            metrics['g_lr'] = self.opt_g.param_groups[0]['lr']
            metrics['d_lr'] = self.opt_d.param_groups[0]['lr']
            append_metrics(
                self.run_dir / 'metrics.csv',
                {'epoch': epoch, **metrics}
            ) if save_artifacts else None
            if metrics['f1'] > self.best:
                self.best = metrics['f1']
                save_checkpoint(
                    self.run_dir / 'G1_best_f1.pth',
                    epoch=epoch,
                    **self.state_dict()
                ) if save_artifacts else None
            if save_artifacts and (epoch + 1) % self.cfg['training']['checkpoint_every'] == 0:
                save_checkpoint(
                    self.run_dir / 'G1_latest.pth',
                    epoch=epoch,
                    **self.state_dict()
                )
            if save_artifacts:
                batch = next(iter(self.val_loader))
                with torch.no_grad():
                    image = batch['masked_image'][:1].to(self.device)
                    mask = batch['mask'][:1].to(self.device)
                    known = batch['known_edge'][:1].to(self.device)
                    pred = self.G(torch.cat([image, mask, known], 1)).sigmoid()
                save_preview(
                    self.run_dir / 'visuals' / f'epoch_{epoch + 1:04d}_preview.png',
                    [batch['image'][0], mask[0] * 2 - 1, batch['edge'][0] * 2 - 1, pred[0] * 2 - 1],
                    ['Original', 'Mask', 'Canny GT', 'Predicted Edge']
                )
                update_dashboard(
                    self.run_dir / 'metrics.csv',
                    self.run_dir / 'dashboard_latest.png',
                    'Stage 1'
                )
            print('S1', epoch, metrics)
        return metrics
