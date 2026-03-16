# =============================================================================
# Trade Encoder - 交易数据特征编码器
# Version: v2.1 - 使用 GroupNorm 替代 BatchNorm，避免信息泄漏
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from .base_encoder import BaseEncoder
import numpy as np
from ..revin import RevIN
def _get_num_groups(num_channels: int, max_groups: int = 8) -> int:
    """
    计算 GroupNorm 的 num_groups 参数
    确保 num_channels 能被 num_groups 整除
    """
    for g in [max_groups, 4, 2, 1]:
        if num_channels % g == 0:
            return g
    return 1


class CausalConv1d(nn.Module):
    """
    因果 1D 卷积。
    
    在 Time 维度上进行 Causal Padding，保证不泄露未来信息。
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3,
        stride: int = 1, 
        dilation: int = 1, 
        bias: bool = True
    ):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        
        # Causal Padding
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=0,  # 手动处理
            dilation=dilation,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = F.pad(x, (self.padding, 0))  # 左侧填充
        return self.conv(x)


class TradeEncoder(BaseEncoder):
    """
    Trade 数据特征编码器。
    
    处理聚合后的交易数据（如 100ms 窗口内的成交统计）。
    
    Input: (Batch, Features, Time)
           e.g., (B, 8, 3000) -> 8个特征: volume, vwap, num_trades, ofi, ...
           
    Output: (Batch, Time_Downsampled, Feature_Dim)
            e.g., (B, 300, 64)
    
    Args:
        in_features: 输入特征数
        hidden_channels: 隐藏层通道数列表
        kernel_size: 卷积核大小
        time_stride: 总的时间降采样倍率
        dropout: Dropout 比率
    """
    
    def __init__(
        self, 
        in_features: int = 8, 
        hidden_channels: List[int] = [16, 16, 16], 
        kernel_size: List[int] = [5, 2, 2],
        time_stride: List[int] = [5, 2, 2],
        dropout: float = 0.1,
        # use_revin: bool = True
    ):
        super().__init__()
        
        self.in_features = in_features
        self.hidden_channels = hidden_channels
        self._time_stride = np.prod(time_stride)
        # self.use_revin = use_revin
        # if self.use_revin:
        #     self.revin = RevIN(num_features=in_features-1, affine=True)
        # else:
        #     self.revin = None
        ## 类别变量的embedding
        self.has_trade_embedding = nn.Embedding(2, hidden_channels[0])
        ## 非类别变量的先特征归一化
        self.norm = nn.LayerNorm(in_features-1)
        ## 映射到hidden_channels[0]
        self.map_to_hidden = nn.Linear(in_features-1, hidden_channels[0])
        # 构建卷积层
        layers = []
        current_channels = hidden_channels[0]
        
        # 计算每层的 stride 分配
        # 例如 time_stride=10 可以分解为 [2, 5] 或 [5, 2]
        # num_conv_layers = len(hidden_channels)
        # strides = self._factorize_stride(time_stride, num_conv_layers)
        
        for i, out_channels in enumerate(hidden_channels):
            layers.append(
                CausalConv1d(
                    current_channels, 
                    out_channels, 
                    kernel_size=kernel_size[i],
                    stride=time_stride[i]
                )
            )
            # 使用 GroupNorm 替代 BatchNorm，避免跨样本信息泄漏
            # num_groups = _get_num_groups(out_channels)
            num_groups = 1 ## 避免信息泄漏
            layers.append(nn.GroupNorm(num_groups, out_channels))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_channels = out_channels
            
        self.encoder = nn.Sequential(*layers)
        self._output_dim = hidden_channels[-1] if hidden_channels else in_features
        
    def _factorize_stride(self, total_stride: int, num_layers: int) -> List[int]:
        """
        将总 stride 分解到各层。
        
        例如: 10 -> [2, 5] 或 [5, 2]
        """
        if num_layers == 0:
            return []
        if num_layers == 1:
            return [total_stride]
            
        # 简单策略: 尽量均匀分配
        strides = [1] * num_layers
        remaining = total_stride
        
        # 优先分配较大的因子到前面的层
        factors = []
        temp = remaining
        for p in [2, 3, 5, 7]:
            while temp % p == 0:
                factors.append(p)
                temp //= p
        if temp > 1:
            factors.append(temp)
            
        # 分配到各层
        for i, f in enumerate(factors):
            strides[i % num_layers] *= f
            
        return strides
        
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    @property
    def downsample_ratio(self) -> int:
        return self._time_stride
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # 输入: (B, F, T) = (B, 8, 3000)
        has_trade = x[:, 0, :].long()  # (B, T) → 正确
        x = x[:, 1:, :]                # (B, F-1, T) → (B,7,3000)
        
        # 转换为 (B, T, F-1) 适配 LayerNorm
        x = x.permute(0, 2, 1).contiguous()  # (B, 3000, 7)
        x = self.norm(x)
        x = self.map_to_hidden(x)            # (B, 3000, hidden_channels[0])
        

        embedding_out = self.has_trade_embedding(has_trade)  # (B, T, C)
        x = x + embedding_out 
        
        # 转换回 (B, C, T) 适配 CausalConv1d（必须！）
        x = x.permute(0, 2, 1).contiguous()  # (B, C, T)
        # ====================================================
        
        # 卷积编码
        x = self.encoder(x)  # (B, C_out, T_out)
        
        # 转换为 Transformer 要求的 (B, T, C)
        x = x.permute(0, 2, 1).contiguous()  # (B, T_out, C_out)
        
        return x
        return x


class TradeEncoderGRU(BaseEncoder):
    """
    基于 GRU 的 Trade 编码器 (可选替代方案)。
    
    适用于需要更强时序建模能力的场景。
    """
    
    def __init__(
        self, 
        in_features: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 2,
        time_stride: int = 10,
        dropout: float = 0.1,
        bidirectional: bool = False
    ):
        super().__init__()
        
        self.in_features = in_features
        self._time_stride = time_stride
        self._hidden_dim = hidden_dim
        self._bidirectional = bidirectional
        
        # 先用 Conv 进行降采样
        self.downsample = CausalConv1d(
            in_features, hidden_dim,
            kernel_size=time_stride,
            stride=time_stride
        )
        
        # GRU 进行时序建模
        self.gru = nn.GRU(
            hidden_dim, 
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self._output_dim = hidden_dim * (2 if bidirectional else 1)
        
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    @property
    def downsample_ratio(self) -> int:
        return self._time_stride
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, T)
        
        x = self.downsample(x)  # (B, hidden, T_out)
        x = x.permute(0, 2, 1)  # (B, T_out, hidden)
        
        x, _ = self.gru(x)  # (B, T_out, hidden * num_directions)
        
        return x
