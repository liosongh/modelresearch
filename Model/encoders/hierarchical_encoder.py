# =============================================================================
# Hierarchical Temporal Encoder - 层次化时间编码器
# Version: v1.0
#
# 在多个时间尺度上分别建模，然后融合：
# - Micro: tick-level动态
# - Meso: 秒级趋势
# - Macro: 分钟级模式
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from .base_encoder import BaseEncoder


class CausalConv2dBlock(nn.Module):
    """因果2D卷积块"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (3, 2),
        stride: Tuple[int, int] = (1, 1),
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.time_kernel = kernel_size[0]
        self.time_stride = stride[0]
        self.time_pad = kernel_size[0] - 1  # causal padding
        
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(0, 0)  # 手动处理时间维度padding
        )
        self.norm = nn.GroupNorm(min(4, out_channels), out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, L)
        # Causal padding on time dimension
        x = F.pad(x, (0, 0, self.time_pad, 0))
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class HierarchicalFusionAttention(nn.Module):
    """
    层次融合注意力
    
    让细粒度特征可以attend到粗粒度特征，获取全局上下文。
    """
    
    def __init__(
        self, 
        micro_dim: int, 
        meso_dim: int, 
        macro_dim: int, 
        d_model: int, 
        nhead: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # 投影到统一维度
        self.micro_proj = nn.Sequential(
            nn.Linear(micro_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.meso_proj = nn.Sequential(
            nn.Linear(meso_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.macro_proj = nn.Sequential(
            nn.Linear(macro_dim, d_model),
            nn.LayerNorm(d_model),
        )
        
        # Micro attends to Meso
        self.micro_meso_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.micro_meso_norm = nn.LayerNorm(d_model)
        
        # Micro attends to Macro  
        self.micro_macro_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.micro_macro_norm = nn.LayerNorm(d_model)
        
        # 融合门控
        self.gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, 3),
            nn.Softmax(dim=-1)
        )
        
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        micro: torch.Tensor, 
        meso: torch.Tensor, 
        macro: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            micro: (B, T_micro, D_micro)
            meso: (B, T_meso, D_meso) where T_meso < T_micro
            macro: (B, T_macro, D_macro) where T_macro < T_meso
            
        Returns:
            output: (B, T_micro, d_model)
        """
        # 投影
        micro_feat = self.micro_proj(micro)  # (B, T_micro, d_model)
        meso_feat = self.meso_proj(meso)     # (B, T_meso, d_model)
        macro_feat = self.macro_proj(macro)  # (B, T_macro, d_model)
        
        # Micro attends to Meso: 获取秒级上下文
        micro_meso, _ = self.micro_meso_attn(
            query=micro_feat,
            key=meso_feat,
            value=meso_feat
        )
        micro_meso = self.micro_meso_norm(micro_feat + self.dropout(micro_meso))
        
        # Micro attends to Macro: 获取分钟级上下文
        micro_macro, _ = self.micro_macro_attn(
            query=micro_feat,
            key=macro_feat,
            value=macro_feat
        )
        micro_macro = self.micro_macro_norm(micro_feat + self.dropout(micro_macro))
        
        # 门控融合
        # 将三个尺度的信息用门控加权
        combined_for_gate = torch.cat([micro_feat, micro_meso, micro_macro], dim=-1)
        gate_weights = self.gate(combined_for_gate)  # (B, T_micro, 3)
        
        # 加权组合
        stacked = torch.stack([micro_feat, micro_meso, micro_macro], dim=-1)  # (B, T, d, 3)
        output = (stacked * gate_weights.unsqueeze(2)).sum(dim=-1)  # (B, T, d)
        
        output = self.out_proj(output)
        
        return output


class HierarchicalTemporalEncoder(BaseEncoder):
    """
    层次化时间编码器
    
    将输入在多个时间尺度上分别建模，然后融合：
    - Micro (细粒度): 保留tick-level动态，stride较小
    - Meso (中粒度): 秒级趋势，中等stride
    - Macro (粗粒度): 分钟级模式，大stride
    
    这比单一尺度的下采样能捕获更丰富的时间结构。
    
    Args:
        in_channels: 输入通道数
        base_channels: 基础通道数
        d_model: 输出特征维度
        micro_stride: 细粒度的时间步长
        meso_stride: 中粒度的时间步长 (相对于micro)
        macro_stride: 粗粒度的时间步长 (相对于meso)
        num_levels: LOB档位数
        dropout: Dropout比率
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 32,
        d_model: int = 64,
        micro_stride: int = 2,
        meso_stride: int = 5,
        macro_stride: int = 5,
        num_levels: int = 10,
        nhead: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.micro_stride = micro_stride
        self.meso_stride = meso_stride
        self.macro_stride = macro_stride
        self._d_model = d_model
        
        # 计算总下采样率
        self._total_downsample = micro_stride
        
        # ============================================================
        # Micro-level encoder: 保留细节
        # ============================================================
        self.micro_encoder = nn.Sequential(
            CausalConv2dBlock(in_channels, base_channels, 
                              kernel_size=(3, 2), stride=(micro_stride, 2), dropout=dropout),
            CausalConv2dBlock(base_channels, base_channels,
                              kernel_size=(3, 2), stride=(1, 2), dropout=dropout),
        )
        # 计算micro输出的level维度 (假设从10 -> 5 -> 2)
        micro_level_out = max(1, num_levels // 4)
        micro_out_dim = base_channels * micro_level_out
        
        # ============================================================
        # Meso-level encoder: 秒级聚合
        # ============================================================
        meso_total_stride = micro_stride * meso_stride
        self.meso_encoder = nn.Sequential(
            CausalConv2dBlock(in_channels, base_channels,
                              kernel_size=(5, 2), stride=(meso_total_stride, 2), dropout=dropout),
            CausalConv2dBlock(base_channels, base_channels * 2,
                              kernel_size=(3, 2), stride=(1, 2), dropout=dropout),
            CausalConv2dBlock(base_channels * 2, base_channels * 2,
                              kernel_size=(3, 2), stride=(1, 2), dropout=dropout),
        )
        meso_level_out = max(1, num_levels // 8)
        meso_out_dim = base_channels * 2 * meso_level_out
        
        # ============================================================
        # Macro-level encoder: 分钟级模式
        # ============================================================
        macro_total_stride = meso_total_stride * macro_stride
        self.macro_encoder = nn.Sequential(
            CausalConv2dBlock(in_channels, base_channels,
                              kernel_size=(7, 2), stride=(macro_total_stride, 2), dropout=dropout),
            CausalConv2dBlock(base_channels, base_channels * 2,
                              kernel_size=(3, 2), stride=(1, 2), dropout=dropout),
            CausalConv2dBlock(base_channels * 2, base_channels * 4,
                              kernel_size=(3, 2), stride=(1, 2), dropout=dropout),
        )
        macro_level_out = max(1, num_levels // 8)
        macro_out_dim = base_channels * 4 * macro_level_out
        
        # ============================================================
        # 层次融合注意力
        # ============================================================
        self.hierarchy_fusion = HierarchicalFusionAttention(
            micro_dim=micro_out_dim,
            meso_dim=meso_out_dim,
            macro_dim=macro_out_dim,
            d_model=d_model,
            nhead=nhead,
            dropout=dropout
        )
        
    @property
    def output_dim(self) -> int:
        return self._d_model
    
    @property
    def downsample_ratio(self) -> int:
        return self._total_downsample
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, L) LOB数据
            
        Returns:
            output: (B, T_micro, d_model)
        """
        B = x.size(0)
        
        # 各层级编码
        micro_feat = self.micro_encoder(x)   # (B, C1, T/micro_stride, L1)
        meso_feat = self.meso_encoder(x)     # (B, C2, T/(micro*meso), L2)
        macro_feat = self.macro_encoder(x)   # (B, C3, T/(micro*meso*macro), L3)
        
        # Flatten Level维度
        micro_feat = micro_feat.permute(0, 2, 1, 3)  # (B, T_micro, C1, L1)
        micro_feat = micro_feat.reshape(B, micro_feat.size(1), -1)  # (B, T_micro, C1*L1)
        
        meso_feat = meso_feat.permute(0, 2, 1, 3)  # (B, T_meso, C2, L2)
        meso_feat = meso_feat.reshape(B, meso_feat.size(1), -1)  # (B, T_meso, C2*L2)
        
        macro_feat = macro_feat.permute(0, 2, 1, 3)  # (B, T_macro, C3, L3)
        macro_feat = macro_feat.reshape(B, macro_feat.size(1), -1)  # (B, T_macro, C3*L3)
        
        # 层次融合
        output = self.hierarchy_fusion(micro_feat, meso_feat, macro_feat)
        
        return output


class HierarchicalTemporalEncoderLite(BaseEncoder):
    """
    轻量级层次化时间编码器
    
    使用多尺度卷积并行提取特征，然后拼接融合。
    比完整版计算量更小。
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 32,
        d_model: int = 64,
        total_stride: int = 10,
        num_levels: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self._d_model = d_model
        self._total_stride = total_stride
        
        # 多尺度卷积分支
        # Branch 1: 小kernel，捕获局部细节
        self.branch1 = nn.Sequential(
            CausalConv2dBlock(in_channels, base_channels, 
                              kernel_size=(3, 2), stride=(total_stride, 2), dropout=dropout),
            CausalConv2dBlock(base_channels, base_channels,
                              kernel_size=(3, 2), stride=(1, 2), dropout=dropout),
        )
        
        # Branch 2: 中等kernel，捕获中等范围
        self.branch2 = nn.Sequential(
            CausalConv2dBlock(in_channels, base_channels,
                              kernel_size=(5, 2), stride=(total_stride, 2), dropout=dropout),
            CausalConv2dBlock(base_channels, base_channels,
                              kernel_size=(3, 2), stride=(1, 2), dropout=dropout),
        )
        
        # Branch 3: 大kernel，捕获全局模式
        self.branch3 = nn.Sequential(
            CausalConv2dBlock(in_channels, base_channels,
                              kernel_size=(9, 2), stride=(total_stride, 2), dropout=dropout),
            CausalConv2dBlock(base_channels, base_channels,
                              kernel_size=(3, 2), stride=(1, 2), dropout=dropout),
        )
        
        # 融合投影
        # 假设每个分支输出 (B, base_channels, T/stride, L/4)
        level_out = max(1, num_levels // 4)
        concat_dim = base_channels * 3 * level_out
        
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
    @property
    def output_dim(self) -> int:
        return self._d_model
    
    @property
    def downsample_ratio(self) -> int:
        return self._total_stride
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, L)
        Returns:
            output: (B, T/stride, d_model)
        """
        B = x.size(0)
        
        # 多尺度编码
        b1 = self.branch1(x)  # (B, C, T', L')
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        
        # 确保时间维度一致 (取最小)
        T_min = min(b1.size(2), b2.size(2), b3.size(2))
        b1 = b1[:, :, :T_min, :]
        b2 = b2[:, :, :T_min, :]
        b3 = b3[:, :, :T_min, :]
        
        # 拼接
        combined = torch.cat([b1, b2, b3], dim=1)  # (B, C*3, T', L')
        
        # Reshape
        combined = combined.permute(0, 2, 1, 3)  # (B, T', C*3, L')
        combined = combined.reshape(B, T_min, -1)  # (B, T', C*3*L')
        
        # 投影
        output = self.fusion(combined)  # (B, T', d_model)
        
        return output
