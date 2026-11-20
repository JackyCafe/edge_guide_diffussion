"""Hinge GAN 損失。

本檔案使用繁體中文說明研究程式的用途。"""
import torch.nn.functional as F

def lsgan_d(real, fake):
    """lsgan_d：執行此模組的對應運算或資料處理。"""
    return 0.5 * (F.mse_loss(real, real.new_ones(real.shape)) + F.mse_loss(fake, fake.new_zeros(fake.shape)))

def lsgan_g(fake):
    """lsgan_g：執行此模組的對應運算或資料處理。"""
    return F.mse_loss(fake, fake.new_ones(fake.shape))

def hinge_d(real, fake):
    """hinge_d：執行此模組的對應運算或資料處理。"""
    return 0.5 * (F.relu(1 - real).mean() + F.relu(1 + fake).mean())

def hinge_g(fake):
    """hinge_g：執行此模組的對應運算或資料處理。"""
    return -fake.mean()
