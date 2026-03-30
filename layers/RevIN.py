# =============================================================================
# RevIN - Reversible Instance Normalization
# 来源: ICLR 2022 "Reversible Instance Normalization for Accurate 
#       Time-Series Forecasting against Distribution Shift"
# =============================================================================

import torch
import torch.nn as nn
from typing import Optional

# code from https://github.com/ts-kim/RevIN, with minor modifications

import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
        
# class RevIN(nn.Module):
#     """
#     Reversible Instance Normalization (可逆实例归一化)
    
#     专为时间序列预测设计的归一化层，解决以下问题：
#     1. 避免跨样本信息泄漏（每个样本独立归一化）
#     2. 处理非平稳时间序列的分布漂移
#     3. 通过可逆设计保留原始数据尺度
    
#     工作流程:
#         输入 → normalize() → 模型处理 → denormalize() → 输出
    
#     Args:
#         num_features: 特征数量
#         eps: 数值稳定性常数
#         affine: 是否使用可学习的仿射变换参数
        
#     Shape:
#         - Input: (Batch, Features, Time) 或 (Batch, Time, Features)
#         - Output: 与输入相同
        
#     Example:
#         >>> revin = RevIN(num_features=23, affine=True)
#         >>> x = torch.randn(32, 23, 100)  # (B, F, T)
#         >>> x_norm = revin(x, mode='norm')
#         >>> # ... 模型处理 ...
#         >>> y = revin(y_norm, mode='denorm')
#     """
    
#     def __init__(
#         self, 
#         num_features: int, 
#         eps: float = 1e-5, 
#         affine: bool = True,
#         dim: int = -1  # 时间维度，-1 表示最后一维
#     ):
#         super().__init__()
#         self.num_features = num_features
#         self.eps = eps
#         self.affine = affine
#         self.dim = dim
        
#         if self.affine:
#             # 可学习的仿射变换参数
#             self.affine_weight = nn.Parameter(torch.ones(num_features))
#             self.affine_bias = nn.Parameter(torch.zeros(num_features))
            
#         # 用于存储归一化时的统计量（denorm时使用）
#         self.register_buffer('mean', None)
#         self.register_buffer('std', None)
        
#     def forward(self, x: torch.Tensor, mode: str = 'norm') -> torch.Tensor:
#         """
#         前向传播
        
#         Args:
#             x: 输入张量
#             mode: 'norm' 归一化 | 'denorm' 逆归一化
            
#         Returns:
#             归一化或逆归一化后的张量
#         """
#         if mode == 'norm':
#             return self._normalize(x)
#         elif mode == 'denorm':
#             return self._denormalize(x)
#         else:
#             raise ValueError(f"Unknown mode: {mode}, expected 'norm' or 'denorm'")
    
#     def _normalize(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         归一化：减去均值，除以标准差
        
#         对于 (B, F, T) 输入，在 T 维度上计算统计量
#         每个样本、每个特征独立计算
#         """
#         # 在时间维度上计算统计量
#         self.mean = x.mean(dim=self.dim, keepdim=True)
#         self.std = x.std(dim=self.dim, keepdim=True) + self.eps
        
#         x_norm = (x - self.mean) / self.std
        
#         if self.affine:
#             # 扩展仿射参数的维度以匹配输入
#             if self.dim == -1 or self.dim == x.dim() - 1:
#                 # (B, F, T) -> weight/bias 需要是 (1, F, 1)
#                 weight = self.affine_weight.view(1, -1, 1)
#                 bias = self.affine_bias.view(1, -1, 1)
#             else:
#                 # (B, T, F) -> weight/bias 需要是 (1, 1, F)
#                 weight = self.affine_weight.view(1, 1, -1)
#                 bias = self.affine_bias.view(1, 1, -1)
                
#             x_norm = x_norm * weight + bias
            
#         return x_norm
    
#     def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         逆归一化：还原到原始尺度
#         """
#         if self.mean is None or self.std is None:
#             raise RuntimeError("Must call normalize before denormalize!")
        
#         if self.affine:
#             # 先逆仿射变换
#             if self.dim == -1 or self.dim == x.dim() - 1:
#                 weight = self.affine_weight.view(1, -1, 1)
#                 bias = self.affine_bias.view(1, -1, 1)
#             else:
#                 weight = self.affine_weight.view(1, 1, -1)
#                 bias = self.affine_bias.view(1, 1, -1)
                
#             x = (x - bias) / (weight + self.eps)
            
#         # 还原尺度
#         x_denorm = x * self.std + self.mean
        
#         return x_denorm
    
#     def reset_stats(self):
#         """重置统计量（在新的前向传播前调用）"""
#         self.mean = None
#         self.std = None



class RevIN2d(nn.Module):
    """
    2D RevIN - 用于 LOB 数据 (Batch, Channel, Time, Level)
    标准可逆实例归一化，修复梯度/数值/存储问题
    """
    def __init__(
        self, 
        num_channels: int, 
        eps: float = 1e-5, 
        affine: bool = True
    ):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            # 形状严格对齐输入 (B, C, T, L)
            self.affine_weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.affine_bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
    
    def forward(self, x: torch.Tensor, mode: str = 'norm') -> torch.Tensor:
        if mode == 'norm':
            # 归一化时 同步返回 统计量（不存实例属性，更安全）
            self.mean, self.std = self._compute_stats(x)
            return self._normalize(x, self.mean, self.std)
        elif mode == 'denorm':
            return self._denormalize(x, self.mean, self.std)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _compute_stats(self, x: torch.Tensor):
        """【核心修复】计算有偏标准差 + 阻断梯度"""
        # 输入: (B, C, T, L)，在 Time+Level 维度计算统计量
        mean = x.mean(dim=(2, 3), keepdim=True).detach()  # detach 切断梯度
        # 有偏方差 + eps → 开根号 = 标准深度学习标准差
        var = torch.var(x, dim=(2, 3), keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps).detach()         # detach 切断梯度
        return mean, std

    def _normalize(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        x_norm = (x - mean) / std
        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias
        return x_norm

    def _denormalize(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        x_denorm = x * std + mean
        return x_denorm
    

