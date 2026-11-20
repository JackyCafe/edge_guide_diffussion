"""VGG 感知特徵損失。

本檔案使用繁體中文說明研究程式的用途。"""
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights

class VGGPerceptual(nn.Module):
    """VGGPerceptual：封裝對應模型或研究流程。"""

    def __init__(self):
        """初始化元件並儲存必要的模型參數與執行狀態。"""
        super().__init__()
        self.features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
        for p in self.features.parameters():
            p.requires_grad_(False)
        self.register_buffer(
            'mean',
            torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        )
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        )

    def forward(self, pred, gt):
        """執行模型前向傳播，回傳對應的預測張量。"""
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
        if gt.shape[1] == 1:
            gt = gt.repeat(1, 3, 1, 1)
        a = ((pred.float().clamp(-1, 1) + 1) / 2 - self.mean) / self.std
        b = ((gt.float().clamp(-1, 1) + 1) / 2 - self.mean) / self.std
        return F.l1_loss(self.features(a), self.features(b).detach())
