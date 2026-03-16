# =============================================================================
# Sparse-Aware Trade Encoder - 稀疏感知交易数据编码器
# Version: v1.0
#
# 核心设计思想：
# 1. 显式建模稀疏性 (has_trade, time_since_last)
# 2. 时间感知: 距上次交易越久，当前交易信息量越大
# 3. 事件vs非事件: 不同的处理路径
# 4. 特征分组处理: 稀疏/价格/量/微观结构特征分开处理
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from .base_encoder import BaseEncoder
import math


class CausalConv1d(nn.Module):
    """因果1D卷积"""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3,
        stride: int = 1, 
        dilation: int = 1, 
    ):
        super().__init__()
        
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=0, dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class CausalConvBlock(nn.Module):
    """因果卷积块"""
    
    def __init__(self, in_ch: int, out_ch: int, stride: int, dropout: float = 0.1):
        super().__init__()
        kernel = max(3, stride)
        self.pad = kernel - 1
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride)
        self.norm = nn.GroupNorm(1, out_ch)  # LayerNorm equivalent
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        x = self.conv(x)
        x = self.norm(x)
        x = F.gelu(x)
        return self.dropout(x)


class FeatureGroupEncoder(nn.Module):
    """
    特征分组编码器
    
    将Trade的12个特征分成4组，分别处理：
    - 稀疏特征: has_trade, log_time_since_last
    - 价格特征: rel_vwap, rel_price_last, rel_price_std
    - 量特征: log_total_volume, log_total_notional, log_trade_count, log_avg_trade_size
    - 微观结构特征: trade_imbalance, notional_imbalance, trade_count_ratio
    """
    
    def __init__(
        self,
        d_model: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # 特征索引 (根据trade_data_process.py中的顺序)
        # 0: has_trade, 1: log_time_since_last, 2: rel_vwap, 3: rel_price_last
        # 4: log_total_volume, 5: log_total_notional, 6: log_trade_count, 7: log_avg_trade_size
        # 8: trade_imbalance, 9: notional_imbalance, 10: trade_count_ratio, 11: rel_price_std
        
        self.sparse_idx = [0, 1]
        self.price_idx = [2, 3, 11]
        self.volume_idx = [4, 5, 6, 7]
        self.micro_idx = [8, 9, 10]
        
        # 分组编码器
        self.sparse_encoder = nn.Sequential(
            nn.Linear(2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
        )
        
        self.price_encoder = nn.Sequential(
            nn.Linear(3, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
        )
        
        self.volume_encoder = nn.Sequential(
            nn.Linear(4, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
        )
        
        self.micro_encoder = nn.Sequential(
            nn.Linear(3, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            x: (B, T, F) Trade特征
        Returns:
            sparse_feat, price_feat, volume_feat, micro_feat: 各组编码
        """
        sparse_feat = self.sparse_encoder(x[:, :, self.sparse_idx])
        price_feat = self.price_encoder(x[:, :, self.price_idx])
        volume_feat = self.volume_encoder(x[:, :, self.volume_idx])
        micro_feat = self.micro_encoder(x[:, :, self.micro_idx])
        
        return sparse_feat, price_feat, volume_feat, micro_feat


class SparseAwareGating(nn.Module):
    """
    稀疏感知门控机制
    
    核心：根据稀疏性调整特征权重
    - has_trade=0时，其他特征应该被抑制
    - time_since_last很长时，当前交易更重要
    """
    
    def __init__(self, d_model: int = 64):
        super().__init__()
        
        self.d_model = d_model
        
        # 基础门控: 根据稀疏特征决定信息通过量
        self.base_gate = nn.Sequential(
            nn.Linear(d_model // 4, d_model),  # 从sparse_feat
            nn.Sigmoid()
        )
        
        # 时间重要性: 距上次交易越久，信息越"新鲜"
        self.time_importance = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid()
        )
        
    def forward(
        self, 
        combined_feat: torch.Tensor,  # (B, T, d_model)
        sparse_feat: torch.Tensor,    # (B, T, d_model//4)
        has_trade: torch.Tensor,      # (B, T)
        log_time: torch.Tensor,       # (B, T)
    ) -> torch.Tensor:
        """
        应用稀疏感知门控
        """
        # 基础门控
        base_g = self.base_gate(sparse_feat)  # (B, T, d_model)
        
        # 时间重要性
        time_imp = self.time_importance(log_time.unsqueeze(-1))  # (B, T, d_model)
        
        # 综合门控
        has_trade_exp = has_trade.unsqueeze(-1)  # (B, T, 1)
        gate = base_g * (0.5 + 0.5 * time_imp * has_trade_exp)
        
        return combined_feat * gate


class DualPathProcessor(nn.Module):
    """
    双路径处理器
    
    - 有交易时: 全特征处理
    - 无交易时: 只有时间流逝信息
    """
    
    def __init__(self, d_model: int = 64, dropout: float = 0.1):
        super().__init__()
        
        # 有事件路径
        self.event_transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        
        # 无事件路径 (只传递稀疏信息)
        self.no_event_transform = nn.Sequential(
            nn.Linear(d_model // 4, d_model),  # 从sparse_feat
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        
    def forward(
        self,
        gated_feat: torch.Tensor,   # (B, T, d_model)
        sparse_feat: torch.Tensor,  # (B, T, d_model//4)
        has_trade: torch.Tensor,    # (B, T)
    ) -> torch.Tensor:
        """
        双路径处理
        """
        has_trade_exp = has_trade.unsqueeze(-1)  # (B, T, 1)
        
        # 有事件路径
        event_out = self.event_transform(gated_feat)
        
        # 无事件路径
        no_event_out = self.no_event_transform(sparse_feat)
        
        # 混合
        output = has_trade_exp * event_out + (1 - has_trade_exp) * no_event_out
        
        return output


class SparseAwareTradeEncoder(BaseEncoder):
    """
    稀疏感知Trade编码器
    
    核心思想:
    1. 显式建模稀疏性 (has_trade, time_since_last)
    2. 时间感知: 距上次交易越久，当前交易信息量越大
    3. 事件vs非事件: 不同的处理路径
    
    Input: (B, F, T) Trade特征，F=12
    Output: (B, T_out, d_model)
    
    你的12个特征顺序:
    0. has_trade (0/1)
    1. log_time_since_last
    2. rel_vwap
    3. rel_price_last
    4. log_total_volume
    5. log_total_notional
    6. log_trade_count
    7. log_avg_trade_size
    8. trade_imbalance
    9. notional_imbalance
    10. trade_count_ratio
    11. rel_price_std
    """
    
    def __init__(
        self,
        in_features: int = 12,
        d_model: int = 64,
        time_strides: List[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.d_model = d_model
        
        # 计算总降采样率
        if time_strides is not None:
            self._downsample_ratio = 1
            for s in time_strides:
                self._downsample_ratio *= s
        else:
            self._downsample_ratio = 1
        
        # ================================================================
        # Part 1: 特征分组编码
        # ================================================================
        self.feature_groups = FeatureGroupEncoder(d_model=d_model, dropout=dropout)
        
        # ================================================================
        # Part 2: 特征融合
        # ================================================================
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        
        # ================================================================
        # Part 3: 稀疏感知门控
        # ================================================================
        self.sparse_gating = SparseAwareGating(d_model=d_model)
        
        # ================================================================
        # Part 4: 双路径处理
        # ================================================================
        self.dual_path = DualPathProcessor(d_model=d_model, dropout=dropout)
        
        # # ================================================================
        # # Part 5: 时间降采样
        # # ================================================================
        # self.time_blocks = nn.ModuleList()
        # for stride in time_strides:
        #     self.time_blocks.append(CausalConvBlock(d_model, d_model, stride, dropout))
        
        self.output_norm = nn.LayerNorm(d_model)
        self._output_dim = d_model
        
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    @property
    def downsample_ratio(self) -> int:
        return self._downsample_ratio
        
    def forward(self, trade: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trade: (B, F, T) Trade特征
        Returns:
            encoded: (B, T_out, d_model)
        """
        B, F, T = trade.shape
        
        # 转为 (B, T, F)
        x = trade.permute(0, 2, 1)  # (B, T, 12)
        
        # 提取稀疏信息
        has_trade = x[:, :, 0]      # (B, T)
        log_time = x[:, :, 1]       # (B, T)
        
        # ================================================================
        # Step 1: 特征分组编码
        # ================================================================
        sparse_feat, price_feat, volume_feat, micro_feat = self.feature_groups(x)
        # 各 (B, T, d//4)
        
        # 拼接
        combined = torch.cat([sparse_feat, price_feat, volume_feat, micro_feat], dim=-1)
        # (B, T, d)
        
        # ================================================================
        # Step 2: 特征融合
        # ================================================================
        fused = self.feature_fusion(combined)  # (B, T, d)
        
        # ================================================================
        # Step 3: 稀疏感知门控
        # ================================================================
        gated = self.sparse_gating(fused, sparse_feat, has_trade, log_time)
        
        # ================================================================
        # Step 4: 双路径处理
        # ================================================================
        output = self.dual_path(gated, sparse_feat, has_trade)
        
        # # ================================================================
        # # Step 5: 时间降采样
        # # ================================================================
        # output = output.permute(0, 2, 1)  # (B, d, T)
        # for block in self.time_blocks:
        #     output = block(output)
        # output = output.permute(0, 2, 1)  # (B, T_out, d)
        
        return self.output_norm(output)


class ConditionalTradeEncoder(BaseEncoder):
    """
    条件Trade编码器
    
    核心思想: 同样的交易，在不同的盘口状态下有不同的"信息含量"
    
    例如：
    - 在流动性充足时的100手买入 vs 在薄盘时的100手买入
    - 顺着盘口不平衡方向的交易 vs 逆向交易
    
    Input:
        trade: (B, F, T) - Trade特征
        lob_context: (B, T, D_ctx) - LOB上下文摘要
        
    Output:
        encoded: (B, T_out, d_model) - 条件编码后的Trade特征
    """
    
    def __init__(
        self,
        in_features: int = 12,
        lob_context_dim: int = 32,
        d_model: int = 64,
        time_strides: List[int] = [5, 2, 2],
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.d_model = d_model
        self.lob_context_dim = lob_context_dim
        
        self._downsample_ratio = 1
        for s in time_strides:
            self._downsample_ratio *= s
        
        # ================================================================
        # Part 1: Trade基础编码器
        # ================================================================
        self.trade_encoder = SparseAwareTradeEncoder(
            in_features=in_features,
            d_model=d_model,
            time_strides=[1],  # 保持原始分辨率
            dropout=dropout,
        )
        
        # ================================================================
        # Part 2: LOB上下文编码
        # ================================================================
        self.context_proj = nn.Sequential(
            nn.Linear(lob_context_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        
        # ================================================================
        # Part 3: 条件门控
        # ================================================================
        # 信息含量门: 根据LOB状态调整Trade信号的"重要性"
        self.info_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        # 条件调制
        self.condition_modulation = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )
        
        # ================================================================
        # Part 4: 融合
        # ================================================================
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        
        # ================================================================
        # Part 5: 时间降采样
        # ================================================================
        self.time_blocks = nn.ModuleList()
        for stride in time_strides:
            self.time_blocks.append(CausalConvBlock(d_model, d_model, stride, dropout))
        
        self.output_norm = nn.LayerNorm(d_model)
        self._output_dim = d_model
        
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    @property
    def downsample_ratio(self) -> int:
        return self._downsample_ratio
        
    def forward(
        self, 
        trade: torch.Tensor,              # (B, F, T)
        lob_context: torch.Tensor,        # (B, T, D_ctx)
    ) -> torch.Tensor:
        """
        条件编码
        """
        B, F, T = trade.shape
        
        # ================================================================
        # Step 1: Trade基础编码
        # ================================================================
        trade_encoded = self.trade_encoder(trade)  # (B, T, d_model)
        
        # ================================================================
        # Step 2: LOB上下文投影
        # ================================================================
        lob_ctx = self.context_proj(lob_context)  # (B, T, d_model)
        
        # ================================================================
        # Step 3: 条件门控
        # ================================================================
        combined = torch.cat([trade_encoded, lob_ctx], dim=-1)
        info_gate = self.info_gate(combined)  # (B, T, d_model)
        modulation = self.condition_modulation(lob_ctx)  # (B, T, d_model)
        
        # 应用门控和调制
        gated_trade = trade_encoded * info_gate * (1 + modulation)
        
        # ================================================================
        # Step 4: 融合
        # ================================================================
        fused = self.fusion(torch.cat([gated_trade, lob_ctx], dim=-1))
        
        # ================================================================
        # Step 5: 时间降采样
        # ================================================================
        output = fused.permute(0, 2, 1)  # (B, d, T)
        for block in self.time_blocks:
            output = block(output)
        output = output.permute(0, 2, 1)  # (B, T_out, d)
        
        return self.output_norm(output)


class EventAwareDownsample(nn.Module):
    """
    事件感知降采样
    
    核心：保留事件信息，不被平均掉
    - 有交易的时刻在降采样时权重更高
    """
    
    def __init__(self, in_dim: int, out_dim: int, factor: int, dropout: float = 0.1):
        super().__init__()
        
        self.factor = factor
        
        # 事件注意力
        self.event_attention = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Tanh(),
            nn.Linear(in_dim, 1),
        )
        
        # 特征变换
        self.transform = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
    def forward(
        self, 
        x: torch.Tensor,           # (B, T, D)
        has_trade: torch.Tensor,   # (B, T)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        事件感知降采样
        
        Returns:
            x_ds: (B, T', D) 降采样后的特征
            has_trade_ds: (B, T') 降采样后的事件标记
        """
        B, T, D = x.shape
        
        # Padding
        pad_len = (self.factor - T % self.factor) % self.factor
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            has_trade = F.pad(has_trade, (0, pad_len))
        
        T_padded = x.size(1)
        T_out = T_padded // self.factor
        
        # Reshape to windows
        x = x.reshape(B, T_out, self.factor, D)
        has_trade_reshaped = has_trade.reshape(B, T_out, self.factor)
        
        # 注意力权重
        attn_scores = self.event_attention(x).squeeze(-1)  # (B, T_out, factor)
        
        # 事件加权: 有交易时attention score更高
        event_boost = has_trade_reshaped * 5.0
        attn_scores = attn_scores + event_boost
        
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, T_out, factor)
        
        # 加权聚合
        x_agg = (x * attn_weights.unsqueeze(-1)).sum(dim=2)  # (B, T_out, D)
        
        # 变换
        x_ds = self.transform(x_agg)
        
        # 降采样has_trade
        has_trade_ds = has_trade_reshaped.max(dim=-1)[0]  # (B, T_out)
        
        return x_ds, has_trade_ds


class SparseAwareTradeEncoderLite(BaseEncoder):
    """
    轻量版稀疏感知Trade编码器
    
    简化版本，直接处理，不分组
    """
    
    def __init__(
        self,
        in_features: int = 12,
        d_model: int = 64,
        time_strides: List[int] = [5, 2, 2],
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self._downsample_ratio = 1
        for s in time_strides:
            self._downsample_ratio *= s
        
        # 直接投影
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        
        # 稀疏门控
        self.sparse_gate = nn.Sequential(
            nn.Linear(2, d_model),  # has_trade + log_time
            nn.Sigmoid()
        )
        
        # 时间降采样
        self.time_blocks = nn.ModuleList()
        for stride in time_strides:
            self.time_blocks.append(CausalConvBlock(d_model, d_model, stride, dropout))
        
        self.output_norm = nn.LayerNorm(d_model)
        self._output_dim = d_model
        
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    @property
    def downsample_ratio(self) -> int:
        return self._downsample_ratio
        
    def forward(self, trade: torch.Tensor) -> torch.Tensor:
        B, F, T = trade.shape
        x = trade.permute(0, 2, 1)  # (B, T, F)
        
        # 稀疏特征
        sparse_feat = x[:, :, :2]  # has_trade, log_time
        
        # 投影
        feat = self.input_proj(x)
        
        # 稀疏门控
        gate = self.sparse_gate(sparse_feat)
        feat = feat * gate
        
        # 时间降采样
        feat = feat.permute(0, 2, 1)
        for block in self.time_blocks:
            feat = block(feat)
        feat = feat.permute(0, 2, 1)
        
        return self.output_norm(feat)
