# =============================================================================
# Event-Driven Trade Encoder - 事件驱动的Trade编码器
# Version: v1.0
#
# 将Trade视为离散事件流而非时间序列，更好地处理稀疏性和事件本质
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from .base_encoder import BaseEncoder


class SparseEventLayer(nn.Module):
    """
    稀疏事件注意力层
    
    关键创新：双重注意力机制
    1. 全局注意力：所有时间步都参与
    2. 事件注意力：只关注有交易的时间步 (更强的信号)
    
    最后门控融合两种表示。
    """
    
    def __init__(
        self, 
        d_model: int, 
        nhead: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 全局注意力
        self.global_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.global_norm = nn.LayerNorm(d_model)
        
        # 事件注意力 (只attend有交易的位置)
        self.event_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.event_norm = nn.LayerNorm(d_model)
        
        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        event_mask: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: 输入特征 (B, T, d_model)
            event_mask: 事件掩码 (B, T), True表示有交易
            causal_mask: 因果掩码 (T, T)
        
        Returns:
            output: 输出特征 (B, T, d_model)
        """
        B, T, D = x.shape
        
        # 1. 全局自注意力 (因果)
        global_normed = self.global_norm(x)
        global_out, _ = self.global_attn(
            global_normed, global_normed, global_normed, 
            attn_mask=causal_mask
        )
        global_out = x + self.dropout(global_out)
        
        # 2. 事件注意力 (只attend有交易的位置)
        # 创建key_padding_mask: True表示被mask掉的位置
        key_padding_mask = ~event_mask  # 无交易的位置被mask
        
        # 检查是否至少有一些交易
        has_any_events = event_mask.any(dim=1)  # (B,)
        
        event_normed = self.event_norm(x)
        
        if has_any_events.all():
            # 所有batch都有交易
            event_out, _ = self.event_attn(
                query=event_normed,
                key=event_normed,
                value=event_normed,
                key_padding_mask=key_padding_mask,
                attn_mask=causal_mask
            )
            event_out = x + self.dropout(event_out)
        else:
            # 有些batch没有交易，需要特殊处理
            event_out = global_out.clone()
            
            for b in range(B):
                if has_any_events[b]:
                    # 单个样本的事件注意力
                    single_x = x[b:b+1]  # (1, T, D)
                    single_mask = key_padding_mask[b:b+1]  # (1, T)
                    
                    single_normed = self.event_norm(single_x)
                    single_out, _ = self.event_attn(
                        query=single_normed,
                        key=single_normed,
                        value=single_normed,
                        key_padding_mask=single_mask,
                        attn_mask=causal_mask
                    )
                    event_out[b] = single_x[0] + self.dropout(single_out[0])
        
        # 3. 门控融合
        combined = torch.cat([global_out, event_out], dim=-1)
        gate = self.gate(combined)
        fused = gate * global_out + (1 - gate) * event_out
        
        # 4. FFN
        ffn_normed = self.ffn_norm(fused)
        output = fused + self.dropout(self.ffn(ffn_normed))
        
        return output


class SparseEventTransformer(nn.Module):
    """
    稀疏事件Transformer
    
    堆叠多层SparseEventLayer进行事件序列建模。
    """
    
    def __init__(
        self, 
        d_model: int, 
        nhead: int, 
        num_layers: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            SparseEventLayer(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        
    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """生成因果掩码"""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
        
    def forward(
        self, 
        x: torch.Tensor, 
        event_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: 输入特征 (B, T, d_model)
            event_mask: 事件掩码 (B, T), True表示有交易
        
        Returns:
            output: 输出特征 (B, T, d_model)
        """
        T = x.size(1)
        causal_mask = self._generate_causal_mask(T, x.device)
        
        for layer in self.layers:
            x = layer(x, event_mask, causal_mask)
        
        x = self.final_norm(x)
        return x


class EventDrivenTradeEncoder(BaseEncoder):
    """
    事件驱动的Trade编码器
    
    核心思想：
    1. 将Trade视为离散事件序列，而非等间隔时间序列
    2. 显式建模事件发生与否 (has_trade)
    3. 使用稀疏注意力机制，只关注有交易的时间步
    
    相比原始的1D CNN，这种方法：
    - 更好地处理Trade数据的稀疏性
    - 事件间的依赖关系建模更准确
    - 避免"无交易"时间步稀释真实信号
    
    Args:
        in_features: 输入特征数
        d_model: 模型维度
        nhead: 注意力头数
        num_layers: Transformer层数
        time_stride: 时间下采样率 (与LOB对齐)
        dropout: Dropout比率
        has_trade_idx: has_trade特征的索引位置
    """
    
    def __init__(
        self,
        in_features: int = 12,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        time_stride: int = 10,
        dropout: float = 0.1,
        has_trade_idx: int = 0
    ):
        super().__init__()
        
        self.in_features = in_features
        self._d_model = d_model
        self._time_stride = time_stride
        self.has_trade_idx = has_trade_idx
        
        # 1. 事件特征嵌入 (处理所有Trade特征)
        self.event_embedding = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 2. 时间编码器 (处理事件间隔，假设第1个特征是log_time_since_last)
        self.time_encoder = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
        )
        
        # 3. 事件状态嵌入 (有/无交易)
        self.state_embedding = nn.Embedding(2, d_model)  # 0: 无交易, 1: 有交易
        
        # 4. 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 5. 事件级Transformer (稀疏注意力)
        self.event_transformer = SparseEventTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # 6. 下采样层 (与LOB时间对齐)
        if time_stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=time_stride, stride=time_stride),
                nn.GroupNorm(4, d_model),
                nn.GELU(),
            )
        else:
            self.downsample = nn.Identity()
        
    @property
    def output_dim(self) -> int:
        return self._d_model
        
    @property
    def downsample_ratio(self) -> int:
        return self._time_stride
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: Trade数据 (B, F, T)
               F[0] = has_trade (0/1)
               F[1] = log_time_since_last
               F[2:] = 其他特征
               
        Returns:
            output: 编码后的特征 (B, T/stride, d_model)
        """
        B, F, T = x.shape
        
        # 分离特征
        has_trade = x[:, self.has_trade_idx, :].long()  # (B, T)
        time_since_last = x[:, 1:2, :].permute(0, 2, 1)  # (B, T, 1)
        
        # 转置为 (B, T, F)
        trade_features = x.permute(0, 2, 1)  # (B, T, F)
        
        # 1. 嵌入事件特征
        event_feat = self.event_embedding(trade_features)  # (B, T, d_model)
        
        # 2. 嵌入时间间隔
        time_feat = self.time_encoder(time_since_last)  # (B, T, d_model)
        
        # 3. 嵌入事件状态
        state_feat = self.state_embedding(has_trade)  # (B, T, d_model)
        
        # 4. 融合
        combined = torch.cat([event_feat, time_feat, state_feat], dim=-1)
        fused = self.feature_fusion(combined)  # (B, T, d_model)
        
        # 5. 事件级Transformer (稀疏注意力)
        event_mask = (has_trade == 1)  # True表示有交易
        output = self.event_transformer(fused, event_mask)  # (B, T, d_model)
        
        # 6. 下采样对齐
        if self._time_stride > 1:
            output = output.permute(0, 2, 1)  # (B, d_model, T)
            output = self.downsample(output)   # (B, d_model, T/stride)
            output = output.permute(0, 2, 1)   # (B, T/stride, d_model)
        
        return output


class EventDrivenTradeEncoderLite(BaseEncoder):
    """
    轻量级事件驱动Trade编码器
    
    使用更简单的架构，减少计算量但保留事件驱动的核心思想。
    
    架构:
    1. 事件感知的卷积编码
    2. 事件门控机制
    3. 时间下采样
    """
    
    def __init__(
        self,
        in_features: int = 12,
        hidden_dim: int = 64,
        time_stride: int = 10,
        dropout: float = 0.1,
        has_trade_idx: int = 0
    ):
        super().__init__()
        
        self.in_features = in_features
        self._hidden_dim = hidden_dim
        self._time_stride = time_stride
        self.has_trade_idx = has_trade_idx
        
        # 特征编码
        self.feature_encoder = nn.Sequential(
            nn.Conv1d(in_features, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(4, hidden_dim),
            nn.GELU(),
        )
        
        # 事件门控
        self.event_gate = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 时间聚合
        self.temporal_pool = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=time_stride, stride=time_stride),
            nn.GroupNorm(4, hidden_dim),
            nn.GELU(),
        )
        
    @property
    def output_dim(self) -> int:
        return self._hidden_dim
        
    @property
    def downsample_ratio(self) -> int:
        return self._time_stride
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, F, T)
        Returns:
            output: (B, T/stride, hidden_dim)
        """
        B, F, T = x.shape
        
        # 提取has_trade
        has_trade = x[:, self.has_trade_idx:self.has_trade_idx+1, :]  # (B, 1, T)
        
        # 编码特征
        encoded = self.feature_encoder(x)  # (B, hidden_dim, T)
        
        # 事件门控：强化有交易时的信号
        gate = self.event_gate(has_trade)  # (B, hidden_dim, T)
        # 使用软门控：有交易时信号放大，无交易时不完全抑制
        gated = encoded * (0.5 + 0.5 * gate)
        
        # 时间下采样
        output = self.temporal_pool(gated)  # (B, hidden_dim, T/stride)
        output = output.permute(0, 2, 1)    # (B, T/stride, hidden_dim)
        
        return output
