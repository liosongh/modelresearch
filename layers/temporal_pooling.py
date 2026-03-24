# =============================================================================
# Temporal Pooling - 时序池化模块
# Version: v1.0
#
# 核心设计思想：
# 1. 在融合后、Transformer之前进行降频
# 2. 减少序列长度，降低Transformer计算量
# 3. 保留关键信息，特别是交易事件
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class AdaptiveTemporalPooling(nn.Module):
    """
    自适应时序池化
    
    使用可学习的query tokens来聚合时间序列
    可以将长序列压缩到任意目标长度
    
    优点：
    1. 自适应选择重要时间步
    2. 可学习的压缩模式
    3. 保留全局信息
    
    Input: (B, T, D) 长序列
    Output: (B, output_len, D) 压缩后的序列
    """
    
    def __init__(
        self,
        d_model: int = 64,
        output_len: int = 30,
        nhead: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.output_len = output_len
        
        # 可学习的query tokens
        self.query_tokens = nn.Parameter(torch.randn(1, output_len, d_model))
        
        # 位置编码
        self.query_pos = nn.Parameter(torch.randn(1, output_len, d_model))
        
        # 交叉注意力
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.query_tokens)
        nn.init.xavier_uniform_(self.query_pos)
        
    def forward(
        self, 
        x: torch.Tensor,
        has_trade: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) 长序列
            has_trade: (B, T) 可选，用于注意力偏置
        Returns:
            out: (B, output_len, D) 压缩后的序列
        """
        B, T, D = x.shape
        
        # 扩展query
        queries = self.query_tokens.expand(B, -1, -1) + self.query_pos
        
        # 交叉注意力
        attended, attn_weights = self.cross_attn(
            query=queries,
            key=x,
            value=x,
        )
        
        # 残差 + Norm
        out = self.norm1(queries + self.dropout(attended))
        
        # FFN
        out = self.norm2(out + self.dropout(self.ffn(out)))
        
        return out


class EventAwarePooling(nn.Module):
    """
    事件感知池化
    
    考虑交易发生的时刻，给予更高的注意力权重
    
    原理：
    - 有交易的时间步包含更多信息
    - 在池化时应该保留这些时间步的信息
    
    Input: (B, T, D)
    Output: (B, T/pool_factor, D)
    """
    
    def __init__(
        self,
        d_model: int = 64,
        pool_factor: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pool_factor = pool_factor
        
        # 注意力权重网络
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )
        
        # 输出变换
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
    def forward(
        self, 
        x: torch.Tensor,
        has_trade: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        事件感知池化
        
        Args:
            x: (B, T, D)
            has_trade: (B, T) 可选
        Returns:
            pooled: (B, T/pool_factor, D)
        """
        B, T, D = x.shape
        
        # Padding
        pad_len = (self.pool_factor - T % self.pool_factor) % self.pool_factor
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            if has_trade is not None:
                has_trade = F.pad(has_trade, (0, pad_len))
        
        T_padded = x.size(1)
        T_out = T_padded // self.pool_factor
        
        # Reshape to windows
        x = x.reshape(B, T_out, self.pool_factor, D)
        
        # 计算注意力分数
        attn_scores = self.attention(x).squeeze(-1)  # (B, T_out, pool_factor)
        
        # 事件加权
        if has_trade is not None:
            has_trade = has_trade.reshape(B, T_out, self.pool_factor)
            event_boost = has_trade * 3.0
            attn_scores = attn_scores + event_boost
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 加权聚合
        pooled = (x * attn_weights.unsqueeze(-1)).sum(dim=2)
        
        return self.output_proj(pooled)


class MultiScalePooling(nn.Module):
    """
    多尺度池化
    
    结合多种池化方式，保留不同粒度的信息
    
    Input: (B, T, D)
    Output: (B, T/pool_factor, D)
    """
    
    def __init__(
        self,
        d_model: int = 64,
        pool_factor: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.pool_factor = pool_factor
        
        # 卷积降采样
        self.conv_pool = nn.Conv1d(
            d_model, d_model,
            kernel_size=pool_factor,
            stride=pool_factor,
            padding=0
        )
        
        # 平均池化
        self.avg_pool = nn.AvgPool1d(pool_factor, stride=pool_factor)
        
        # 最大池化
        self.max_pool = nn.MaxPool1d(pool_factor, stride=pool_factor)
        
        # 融合
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
    def forward(
        self, 
        x: torch.Tensor,
        has_trade: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        多尺度池化
        """
        B, T, D = x.shape
        
        # Padding
        pad_len = (self.pool_factor - T % self.pool_factor) % self.pool_factor
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        
        x_t = x.permute(0, 2, 1)  # (B, D, T)
        
        # 三种池化
        conv_out = self.conv_pool(x_t).permute(0, 2, 1)
        avg_out = self.avg_pool(x_t).permute(0, 2, 1)
        max_out = self.max_pool(x_t).permute(0, 2, 1)
        
        # 融合
        combined = torch.cat([conv_out, avg_out, max_out], dim=-1)
        return self.fusion(combined)


class PerceiverPooling(nn.Module):
    """
    Perceiver风格池化
    
    使用固定数量的latent tokens，通过交叉注意力压缩序列
    可以将任意长度的序列压缩到固定长度
    
    参考：Perceiver IO (ICML 2022)
    
    Input: (B, T, D) 任意长度
    Output: (B, num_latents, D) 固定长度
    """
    
    def __init__(
        self,
        d_model: int = 64,
        num_latents: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_latents = num_latents
        
        # 可学习的latent tokens
        self.latents = nn.Parameter(torch.randn(1, num_latents, d_model))
        
        # 多层处理
        self.layers = nn.ModuleList([
            PerceiverLayer(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_norm = nn.LayerNorm(d_model)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.latents)
        
    def forward(
        self, 
        x: torch.Tensor,
        has_trade: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perceiver池化
        """
        B = x.size(0)
        
        latents = self.latents.expand(B, -1, -1)
        
        for layer in self.layers:
            latents = layer(latents, x)
        
        return self.output_norm(latents)


class PerceiverLayer(nn.Module):
    """Perceiver单层"""
    
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d_model)
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.self_norm = nn.LayerNorm(d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, latents: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        # Cross-attention
        cross_out, _ = self.cross_attn(latents, inputs, inputs)
        latents = self.cross_norm(latents + self.dropout(cross_out))
        
        # Self-attention
        self_out, _ = self.self_attn(latents, latents, latents)
        latents = self.self_norm(latents + self.dropout(self_out))
        
        # FFN
        latents = self.ffn_norm(latents + self.dropout(self.ffn(latents)))
        
        return latents


class CausalConvPooling(nn.Module):
    """
    因果卷积池化
    
    最简单高效的方式，使用因果卷积进行降采样
    
    Input: (B, T, D)
    Output: (B, T/pool_factor, D)
    """
    
    def __init__(
        self,
        d_model: int = 64,
        pool_factor: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.pool_factor = pool_factor
        kernel_size = max(3, pool_factor)
        self.pad = kernel_size - 1
        
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, stride=pool_factor)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor,
        has_trade: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        因果卷积池化
        """
        B, T, D = x.shape
        
        # Padding
        pad_len = (self.pool_factor - T % self.pool_factor) % self.pool_factor
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        
        x = x.permute(0, 2, 1)  # (B, D, T)
        x = F.pad(x, (self.pad, 0))  # 因果padding
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # (B, T', D)
        
        return self.dropout(F.gelu(self.norm(x)))


class HybridPooling(nn.Module):
    """
    混合池化 (推荐)
    
    结合事件感知和注意力池化的优点
    
    1. 首先用事件感知池化减少长度
    2. 然后用注意力精炼特征
    
    Input: (B, T, D)
    Output: (B, output_len, D)
    """
    
    def __init__(
        self,
        d_model: int = 64,
        pool_factor: int = 5,
        output_len: Optional[int] = None,
        nhead: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.pool_factor = pool_factor
        self.output_len = output_len
        
        # 第一阶段: 事件感知池化
        self.event_pool = EventAwarePooling(
            d_model=d_model,
            pool_factor=pool_factor,
            dropout=dropout,
        )
        
        # 第二阶段: 自注意力精炼 (可选)
        if output_len is not None:
            self.use_adaptive = True
            self.adaptive_pool = AdaptiveTemporalPooling(
                d_model=d_model,
                output_len=output_len,
                nhead=nhead,
                dropout=dropout,
            )
        else:
            self.use_adaptive = False
            self.adaptive_pool = None
            
    def forward(
        self, 
        x: torch.Tensor,
        has_trade: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        混合池化
        """
        # 第一阶段
        pooled = self.event_pool(x, has_trade)
        
        # 第二阶段 (可选)
        if self.use_adaptive:
            pooled = self.adaptive_pool(pooled)
        
        return pooled


def create_pooling(
    pool_type: str = 'event_aware',
    d_model: int = 64,
    pool_factor: int = 5,
    output_len: Optional[int] = None,
    nhead: int = 4,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    """
    池化模块工厂函数
    
    Args:
        pool_type: 池化类型
            - 'adaptive': 自适应注意力池化
            - 'event_aware': 事件感知池化
            - 'multi_scale': 多尺度池化
            - 'perceiver': Perceiver风格池化
            - 'conv': 因果卷积池化
            - 'hybrid': 混合池化
        d_model: 特征维度
        pool_factor: 池化因子
        output_len: 输出长度 (用于adaptive/perceiver)
        nhead: 注意力头数
        dropout: Dropout比率
    """
    
    if pool_type == 'adaptive':
        assert output_len is not None, "output_len required for adaptive pooling"
        return AdaptiveTemporalPooling(
            d_model=d_model,
            output_len=output_len,
            nhead=nhead,
            dropout=dropout,
        )
    elif pool_type == 'event_aware':
        return EventAwarePooling(
            d_model=d_model,
            pool_factor=pool_factor,
            dropout=dropout,
        )
    elif pool_type == 'multi_scale':
        return MultiScalePooling(
            d_model=d_model,
            pool_factor=pool_factor,
            dropout=dropout,
        )
    elif pool_type == 'perceiver':
        return PerceiverPooling(
            d_model=d_model,
            num_latents=output_len or 32,
            nhead=nhead,
            num_layers=kwargs.get('num_layers', 2),
            dropout=dropout,
        )
    elif pool_type == 'conv':
        return CausalConvPooling(
            d_model=d_model,
            pool_factor=pool_factor,
            dropout=dropout,
        )
    elif pool_type == 'hybrid':
        return HybridPooling(
            d_model=d_model,
            pool_factor=pool_factor,
            output_len=output_len,
            nhead=nhead,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown pool_type: {pool_type}")
