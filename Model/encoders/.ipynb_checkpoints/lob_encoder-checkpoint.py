# =============================================================================
# LOB Encoder - 限价订单簿特征编码器
# Version: v2.1 - 使用 GroupNorm 替代 BatchNorm，避免信息泄漏
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from .base_encoder import BaseEncoder


def _get_num_groups(num_channels: int, max_groups: int = 8) -> int:
    """
    计算 GroupNorm 的 num_groups 参数
    确保 num_channels 能被 num_groups 整除
    """
    for g in [max_groups, 4, 2, 1]:
        if num_channels % g == 0:
            return g
    return 1


class CausalConv2d(nn.Module):
    """
    因果 2D 卷积。
    
    在 Time 维度上进行 Causal Padding，保证不泄露未来信息。
    
    Input: (Batch, Channel, Time, Level)
    Output: (Batch, Channel_out, Time_out, Level_out)
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: tuple = (3, 2),
        stride: tuple = (1, 1), 
        dilation: tuple = (1, 1), 
        bias: bool = True
    ):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        
        # Time 维度的 Causal Padding (只在左侧/过去填充)
        # Padding = (K - 1) * D
        self.time_pad = (kernel_size[0] - 1) * dilation[0]
        
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=(0, 0),  # 手动处理 padding
            dilation=dilation,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, L)
        
        # Causal Padding: 只在 Time 维度左侧填充
        # F.pad format: (left, right, top, bottom) for last 2 dims
        # x shape: (B, C, T, L) -> last dim is L, 2nd last is T
        x = F.pad(x, (0, 0, self.time_pad, 0))
        
        return self.conv(x)


class CausalDownsamplingBlock(nn.Module):
    """
    因果降采样模块。
    
    功能:
    1. 在 Time 维度进行可学习的下采样 (Stride > 1)，同时保证因果性
    2. 在 Level 维度进行特征聚合
    
    Input: (Batch, C_in, T, L_in)
    Output: (Batch, C_out, T_out, L_out)
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        time_stride: int = 2, 
        level_stride: int = 2,
        time_kernel: int = 3,
        level_kernel: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.time_stride = time_stride
        self.level_stride = level_stride
        
        # Level 维度压缩 (空间聚合)
        # 使用 GroupNorm 替代 BatchNorm，避免跨样本信息泄漏
        num_groups = 2
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 
                kernel_size=(1, level_kernel), 
                stride=(1, level_stride)
            ),
            nn.GroupNorm(num_groups, out_channels),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        )
        
        # Time 维度压缩 (时间聚合/降频) - 使用因果卷积
        self.time_conv = CausalConv2d(
            out_channels, 
            out_channels, 
            kernel_size=(time_kernel, 1), 
            stride=(time_stride, 1)
        )
        self.norm_time = nn.GroupNorm(num_groups, out_channels)
        # self.norm_time = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, L)
        x = self.spatial_conv(x)
        x = self.time_conv(x)
        x = self.norm_time(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class LOBEncoder(BaseEncoder):
    """
    LOB 特征编码器 (多层因果降采样网络)。
    
    将高频 LOB 数据降频并提取特征。
    
    Input: (Batch, Channels, Time, Levels)
           e.g., (B, 4, 3000, 10) -> 4通道: ask_p, bid_p, ask_v, bid_v
           
    Output: (Batch, Time_Downsampled, Feature_Dim)
            e.g., (B, 300, 32)
    
    Args:
        in_channels: 输入通道数 (e.g., 4 for price+volume)
        base_channels: 基础通道数
        num_layers: 降采样层数
        time_strides: 每层的时间步长
        level_strides: 每层的 level 步长
        dropout: Dropout 比率
    """
    
    def __init__(
        self, 
        in_channels: int = 4, 
        base_channels: int = 32, 
        num_layers: int = 3, 
        time_strides: List[int] = [2, 5, 1],
        level_strides: List[int] = [2, 2, 2],
        kernel_sizes:dict = {'time': 3, 'level': 2},
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_layers = num_layers
        self.time_strides = time_strides
        self.level_strides = level_strides
        
        # 计算总降采样倍率
        self._downsample_ratio = 1
        for ts in time_strides:
            self._downsample_ratio *= ts
            
        # 构建降采样层
        self.blocks = nn.ModuleList()
        current_channels = in_channels
        next_channels = base_channels
        ## 归一化
        self.norm = nn.GroupNorm(2,in_channels)
        for i in range(num_layers):
            t_stride = time_strides[i] if i < len(time_strides) else 1
            l_stride = level_strides[i] if i < len(level_strides) else 1
            
            block = CausalDownsamplingBlock(
                current_channels, 
                next_channels, 
                time_stride=t_stride,
                level_stride=l_stride,
                time_kernel=kernel_sizes['time'],
                level_kernel=kernel_sizes['level'],
                dropout=dropout
            )
            self.blocks.append(block)
            
            current_channels = next_channels
            
        self._output_channels = current_channels
        # 最终 Level 维度会被压缩，需要在 forward 中动态计算
        self._final_level_dim = None
        
    @property
    def output_dim(self) -> int:
        """编码后的特征维度 = channels * final_level_dim"""
        if self._final_level_dim is None:
            # 需要通过一次前向传播来确定
            return self._output_channels  # 默认假设 Level=1
        return self._output_channels * self._final_level_dim
    
    @property
    def downsample_ratio(self) -> int:
        return self._downsample_ratio
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, L)
        x = self.norm(x)
        for block in self.blocks:
            x = block(x)
            
        # x: (B, C_out, T_out, L_out)
        B, C, T, L = x.shape
        self._final_level_dim = L
        
        # Reshape to (B, T, C*L) for transformer input
        x = x.permute(0, 2, 1, 3)  # (B, T, C, L)
        x = x.reshape(B, T, C * L)  # (B, T, C*L)
        
        return x


class LOBEncoderV2(BaseEncoder):
    """
    改进的LOB编码器
    
    改进点：
    1. Level-aware卷积：显式建模不同档位间的关系
    2. 通道渐进增长：32→64→64
    3. 注意力增强：在最后加入Self-Attention
    4. 保留Level结构：不是简单flatten，而是用Attention聚合
    """
    
    def __init__(
        self, 
        in_channels: int = 4,
        base_channels: int = 32,
        d_model: int = 64,
        time_strides: List[int] = [5, 5, 2],
        num_levels: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.norm = nn.GroupNorm(2,in_channels) ## 归一化
        # 1. 初始特征提取 (不改变分辨率)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(2, base_channels),
            nn.GELU(),
        )
        
        # 2. Level-aware模块：捕获不同档位间的关系
        self.level_attention = LevelAttention(base_channels, num_levels)
        
        # 3. 时间降采样（逐层增加通道）
        self.time_blocks = nn.ModuleList()
        channels = [base_channels, base_channels * 2, base_channels * 2]  # 32→64→64
        
        for i, (in_c, out_c, stride) in enumerate(zip(
            [base_channels] + channels[:-1], channels, time_strides
        )):
            self.time_blocks.append(
                CausalDownsamplingBlockV2(in_c, out_c, time_stride=stride, dropout=dropout)
            )
        
        # 4. Level聚合（用Attention而非简单flatten）
        self.level_aggregator = LevelAggregator(channels[-1], d_model)
        
        self._output_dim = d_model
        
    def forward(self, x):
        # x: (B, 4, T, 10)
        x = self.norm(x)
        x = self.stem(x)                    # (B, 32, T, 10)
        x = self.level_attention(x)         # (B, 32, T, 10) - Level间关系增强
        
        for block in self.time_blocks:
            x = block(x)                    # 逐步降采样Time
        
        # x: (B, 64, T/50, L')
        x = self.level_aggregator(x)        # (B, T/50, d_model) - Attention聚合Level
        return x


class LevelAttention(nn.Module):
    """Level间注意力：让模型学习不同档位的重要性"""
    
    def __init__(self, channels, num_levels):
        super().__init__()
        # 每个Level生成一个注意力权重
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, num_levels)),  # 时间维度pool
            nn.Conv2d(channels, channels, kernel_size=(1, num_levels), groups=channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (B, C, T, L)
        attn = self.attention(x)  # (B, C, 1, 1) or (B, C, 1, L)
        return x * attn.expand_as(x) + x  # 残差连接


class LevelAggregator(nn.Module):
    """用注意力聚合Level维度，而非简单flatten"""
    
    def __init__(self, in_channels, d_model):
        super().__init__()
        self.query = nn.Linear(in_channels, d_model)
        self.key = nn.Linear(in_channels, d_model)
        self.value = nn.Linear(in_channels, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # x: (B, C, T, L)
        B, C, T, L = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, T, L, C)
        
        # 每个时间步，用Attention聚合L个Level
        q = self.query(x.mean(dim=2, keepdim=True))  # (B, T, 1, d)
        k = self.key(x)    # (B, T, L, d)
        v = self.value(x)  # (B, T, L, d)
        
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (d_model ** 0.5), dim=-1)
        out = torch.matmul(attn, v).squeeze(2)  # (B, T, d)
        
        return self.out_proj(out)