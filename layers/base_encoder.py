# =============================================================================
# Base Encoder - 编码器基类
# Version: v2.0
# =============================================================================

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple


class BaseEncoder(nn.Module, ABC):
    """
    所有模态编码器的抽象基类。
    
    约束:
    1. 输出形状必须为 (Batch, Time_Downsampled, Feature_Dim)
    2. 必须实现 output_dim 属性
    3. 必须实现 downsample_ratio 属性
    """
    
    def __init__(self):
        super().__init__()
        
    @property
    @abstractmethod
    def output_dim(self) -> int:
        """返回编码后的特征维度。"""
        pass
    
    @property
    @abstractmethod
    def downsample_ratio(self) -> int:
        """返回时间维度的降采样倍率。"""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            x: 输入张量，形状取决于具体模态
            
        Returns:
            编码后的特征，形状 (B, T_out, D)
        """
        pass
    
    def get_output_length(self, input_length: int) -> int:
        """
        计算给定输入长度对应的输出长度。
        
        Args:
            input_length: 输入序列长度
            
        Returns:
            输出序列长度
        """
        return input_length // self.downsample_ratio
