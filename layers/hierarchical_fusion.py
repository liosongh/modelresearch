# =============================================================================
# Hierarchical Fusion - 分层融合模块
# Version: v1.0
#
# 核心设计思想：
# 1. 多尺度处理: 100ms, 500ms, 2s 不同时间尺度的交互
# 2. LOB和Trade使用不同的降采样策略
# 3. 在每个尺度进行融合，最后聚合多尺度信息
# 4. 事件驱动: Trade发生时的即时影响
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
import math


class TickLevelInteraction(nn.Module):
    """
    Tick级别交互 (100ms)
    
    核心：Trade发生时的即时影响
    - Trade发生 → 吃掉LOB深度 → LOB状态更新
    - LOB状态 → 影响Trade的"信息含量"解读
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        
        # Trade对LOB的即时冲击
        self.trade_impact = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),  # 有界影响
        )
        
        # LOB对Trade的条件调制
        self.lob_condition = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
        )
        
        # 影响门控
        self.impact_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        self.lob_update = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.trade_update = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
    def forward(
        self, 
        lob_feat: torch.Tensor,    # (B, T, D)
        trade_feat: torch.Tensor,  # (B, T, D)
        has_trade: torch.Tensor,   # (B, T)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tick级交互
        """
        B, T, D = lob_feat.shape
        has_trade_exp = has_trade.unsqueeze(-1)  # (B, T, 1)
        
        # 1. Trade对LOB的冲击 (只在有交易时)
        impact = self.trade_impact(trade_feat) * has_trade_exp
        
        # 2. 计算冲击强度
        gate = self.impact_gate(torch.cat([lob_feat, impact], dim=-1))
        
        # 3. 更新LOB
        lob_updated = self.lob_update(torch.cat([lob_feat, gate * impact], dim=-1))
        
        # 4. Trade以LOB为条件
        lob_cond = self.lob_condition(lob_feat)
        trade_updated = self.trade_update(torch.cat([trade_feat, lob_cond], dim=-1))
        
        return lob_updated, trade_updated


class CrossModalTransformerLayer(nn.Module):
    """
    跨模态Transformer层
    
    双向Cross-Attention让两个模态深度交互
    """
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        
        # LOB self-attention
        self.lob_self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.lob_self_norm = nn.LayerNorm(d_model)
        
        # LOB cross-attention (query Trade)
        self.lob_cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.lob_cross_norm = nn.LayerNorm(d_model)
        
        # Trade self-attention
        self.trade_self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.trade_self_norm = nn.LayerNorm(d_model)
        
        # Trade cross-attention (query LOB)
        self.trade_cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.trade_cross_norm = nn.LayerNorm(d_model)
        
        # FFN
        self.lob_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.lob_ffn_norm = nn.LayerNorm(d_model)
        
        self.trade_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.trade_ffn_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        lob: torch.Tensor, 
        trade: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        跨模态交互
        """
        # Self-attention
        lob_self, _ = self.lob_self_attn(lob, lob, lob, attn_mask=causal_mask)
        lob = self.lob_self_norm(lob + self.dropout(lob_self))
        
        trade_self, _ = self.trade_self_attn(trade, trade, trade, attn_mask=causal_mask)
        trade = self.trade_self_norm(trade + self.dropout(trade_self))
        
        # Cross-attention
        lob_cross, _ = self.lob_cross_attn(lob, trade, trade, attn_mask=causal_mask)
        lob = self.lob_cross_norm(lob + self.dropout(lob_cross))
        
        trade_cross, _ = self.trade_cross_attn(trade, lob, lob, attn_mask=causal_mask)
        trade = self.trade_cross_norm(trade + self.dropout(trade_cross))
        
        # FFN
        lob = self.lob_ffn_norm(lob + self.dropout(self.lob_ffn(lob)))
        trade = self.trade_ffn_norm(trade + self.dropout(self.trade_ffn(trade)))
        
        return lob, trade


class CrossModalTransformer(nn.Module):
    """
    跨模态Transformer
    
    多层双向Cross-Attention
    """
    
    def __init__(
        self, 
        d_model: int, 
        nhead: int, 
        num_layers: int = 2, 
        dropout: float = 0.1,
        use_causal_mask: bool = True,
    ):
        super().__init__()
        
        self.use_causal_mask = use_causal_mask
        
        self.layers = nn.ModuleList([
            CrossModalTransformerLayer(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])
        
    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """生成因果掩码"""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
        
    def forward(
        self, 
        lob_feat: torch.Tensor, 
        trade_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        跨模态Transformer处理
        """
        T = lob_feat.size(1)
        
        if self.use_causal_mask:
            causal_mask = self._generate_causal_mask(T, lob_feat.device)
        else:
            causal_mask = None
        
        for layer in self.layers:
            lob_feat, trade_feat = layer(lob_feat, trade_feat, causal_mask)
        
        return lob_feat, trade_feat


class TemporalDownsample(nn.Module):
    """时间降采样 (因果卷积)"""
    
    def __init__(self, factor: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        
        self.factor = factor
        kernel = max(3, factor)
        self.pad = kernel - 1
        
        self.conv = nn.Conv1d(d_model, d_model, kernel, stride=factor)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            x_ds: (B, T/factor, D)
        """
        x = x.permute(0, 2, 1)  # (B, D, T)
        
        # Padding
        T = x.size(2)
        pad_len = (self.factor - T % self.factor) % self.factor
        if pad_len > 0:
            x = F.pad(x, (pad_len, 0))  # 左侧padding保持因果性
        
        x = F.pad(x, (self.pad, 0))  # 因果padding for conv
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # (B, T/factor, D)
        
        return self.dropout(F.gelu(self.norm(x)))


class EventAwareTemporalDownsample(nn.Module):
    """
    事件感知时间降采样
    
    有交易的时刻在降采样时权重更高
    """
    
    def __init__(self, factor: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        
        self.factor = factor
        
        # 事件注意力
        self.event_attention = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )
        
        # 特征变换
        self.transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
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
        
        # 事件加权
        event_boost = has_trade_reshaped * 5.0
        attn_scores = attn_scores + event_boost
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 加权聚合
        x_agg = (x * attn_weights.unsqueeze(-1)).sum(dim=2)
        x_ds = self.transform(x_agg)
        
        # 降采样has_trade
        has_trade_ds = has_trade_reshaped.max(dim=-1)[0]
        
        return x_ds, has_trade_ds


class ScaleFusion(nn.Module):
    """
    单尺度融合模块
    
    在特定时间尺度融合LOB和Trade特征
    """
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        
        # Cross-attention融合
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        
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
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        lob: torch.Tensor, 
        trade: torch.Tensor,
        has_trade: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        融合LOB和Trade
        """
        # Cross-attention: LOB queries Trade
        fused, _ = self.cross_attn(lob, trade, trade)
        fused = self.norm1(lob + self.dropout(fused))
        
        # 门控组合
        gate = self.gate(torch.cat([lob, trade], dim=-1))
        combined = gate * lob + (1 - gate) * trade
        
        # FFN
        output = self.norm2(combined + self.dropout(self.ffn(combined)))
        
        return output


class MultiScaleAggregator(nn.Module):
    """
    多尺度聚合器
    
    自适应融合不同时间尺度的特征
    """
    
    def __init__(self, d_model: int, num_scales: int):
        super().__init__()
        
        self.num_scales = num_scales
        
        # 学习各尺度的重要性
        self.scale_importance = nn.Sequential(
            nn.Linear(d_model * 2, num_scales),
            nn.Softmax(dim=-1)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(d_model * num_scales, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        
    def forward(
        self, 
        scale_features: List[torch.Tensor],  # List of (B, T, D)
        lob_feat: torch.Tensor,              # (B, T, D) 用于计算重要性
        trade_feat: torch.Tensor,            # (B, T, D)
    ) -> torch.Tensor:
        """
        多尺度聚合
        """
        # 计算尺度重要性
        context = torch.cat([lob_feat, trade_feat], dim=-1)
        importance = self.scale_importance(context)  # (B, T, num_scales)
        
        # 加权
        weighted = sum(
            scale_features[i] * importance[:, :, i:i+1]
            for i in range(self.num_scales)
        )
        
        # 拼接并融合
        concat = torch.cat(scale_features, dim=-1)
        fused = self.fusion(concat)
        
        return fused + weighted


class HierarchicalFusion(nn.Module):
    """
    分层融合模块
    
    核心设计：
    1. 在原始时间分辨率(100ms)进行早期交互
    2. 降频后在粗粒度进行深度交互
    3. 多尺度信息聚合
    
    时间尺度设计：
    - Level 0: 100ms (原始) - tick级动态
    - Level 1: 500ms (5x) - 短期响应
    - Level 2: 2s (4x, 累计20x) - 中期趋势
    """
    
    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_cross_layers: int = 2,
        scales: List[int] = [1, 5, 4],  # 相对降采样倍率
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.scales = scales
        self.num_scales = len(scales)
        
        # 计算绝对降采样倍率
        self.absolute_scales = []
        cumulative = 1
        for s in scales:
            cumulative *= s
            self.absolute_scales.append(cumulative)
        
        # ================================================================
        # Stage 1: Tick级交互 (100ms)
        # ================================================================
        self.tick_interaction = TickLevelInteraction(d_model, dropout)
        
        # ================================================================
        # Stage 2: 各尺度的降采样器和融合器
        # ================================================================
        self.lob_downsamplers = nn.ModuleList()
        self.trade_downsamplers = nn.ModuleList()
        self.scale_fusers = nn.ModuleList()
        self.cross_modal_transformers = nn.ModuleList()
        
        for i, scale in enumerate(scales[1:]):  # 跳过scale=1
            # LOB降采样 (卷积)
            self.lob_downsamplers.append(
                TemporalDownsample(scale, d_model, dropout)
            )
            
            # Trade降采样 (事件感知)
            self.trade_downsamplers.append(
                EventAwareTemporalDownsample(scale, d_model, dropout)
            )
            
        # 每个尺度的融合器和跨模态Transformer
        for i in range(self.num_scales-1):
            self.scale_fusers.append(
                ScaleFusion(d_model, nhead, dropout)
            )
            self.cross_modal_transformers.append(
                CrossModalTransformer(d_model, nhead, num_cross_layers, dropout)
            )
        
        # ================================================================
        # Stage 3: 多尺度聚合
        # ================================================================
        self.scale_aggregator = MultiScaleAggregator(d_model, self.num_scales-1)
        
        self.output_norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        lob_feat: torch.Tensor,    # (B, T, D) @ 100ms
        trade_feat: torch.Tensor,  # (B, T, D) @ 100ms
        has_trade: torch.Tensor,   # (B, T)
    ) -> Dict[str, torch.Tensor]:
        """
        分层融合
        
        Returns:
            dict with:
            - 'fused': (B, T, D) 融合后的特征 (原始分辨率)
            - 'scale_features': 各尺度特征
        """
        B, T, D = lob_feat.shape
        
        # 存储各尺度特征
        lob_scales = [lob_feat]
        trade_scales = [trade_feat]
        has_trade_scales = [has_trade]
        fused_scales = []
        
        # ============================================================
        # Stage 1: Tick级交互
        # ============================================================
        lob_tick, trade_tick = self.tick_interaction(lob_feat, trade_feat, has_trade)
        lob_scales[0] = lob_tick
        trade_scales[0] = trade_tick
        
        # ============================================================
        # Stage 2: 逐级降采样
        # ============================================================
        for i, (lob_down, trade_down) in enumerate(
            zip(self.lob_downsamplers, self.trade_downsamplers)
        ):
            lob_prev = lob_scales[-1]
            trade_prev = trade_scales[-1]
            has_trade_prev = has_trade_scales[-1]
            
            # LOB降采样
            lob_ds = lob_down(lob_prev)
            lob_scales.append(lob_ds)
            
            # Trade降采样
            trade_ds, has_trade_ds = trade_down(trade_prev, has_trade_prev)
            trade_scales.append(trade_ds)
            has_trade_scales.append(has_trade_ds)
        
        # ============================================================
        # Stage 3: 每个尺度的跨模态交互和融合
        # ============================================================
        for i, (cross_transformer, fuser) in enumerate(
            zip(self.cross_modal_transformers, self.scale_fusers)
        ):
            lob_s = lob_scales[i+1]
            trade_s = trade_scales[i+1]
            has_trade_s = has_trade_scales[i+1]
            
            # 跨模态交互
            lob_s, trade_s = cross_transformer(lob_s, trade_s)
            
            # 融合
            fused_s = fuser(lob_s, trade_s, has_trade_s)
            fused_scales.append(fused_s)
        
        # ============================================================
        # Stage 4: 上采样到统一尺度并聚合
        # ============================================================
        T_finest = T
        fused_upsampled = []
        
        for i, fused in enumerate(fused_scales):
            if fused.size(1) < T_finest:
                # 上采样
                fused_up = F.interpolate(
                    fused.permute(0, 2, 1),
                    size=T_finest,
                    mode='linear',
                    align_corners=False
                ).permute(0, 2, 1)
            else:
                fused_up = fused
            fused_upsampled.append(fused_up)
        
        # 多尺度聚合
        final_fused = self.scale_aggregator(fused_upsampled, lob_tick, trade_tick)
        
        return {
            'fused': self.output_norm(final_fused),
            'scale_features': fused_scales,
            'lob_scales': lob_scales,
            'trade_scales': trade_scales,
        }


class HierarchicalFusionLite(nn.Module):
    """
    轻量版分层融合
    
    简化版本，只用两个尺度
    """
    
    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        scale_factor: int = 5,  # 降采样倍率
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.scale_factor = scale_factor
        
        # Tick交互
        self.tick_interaction = TickLevelInteraction(d_model, dropout)
        
        # 降采样
        self.lob_downsample = TemporalDownsample(scale_factor, d_model, dropout)
        self.trade_downsample = EventAwareTemporalDownsample(scale_factor, d_model, dropout)
        
        # 跨模态Transformer
        self.cross_transformer = CrossModalTransformer(d_model, nhead, 2, dropout)
        
        # 融合
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.output_norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        lob_feat: torch.Tensor,
        trade_feat: torch.Tensor,
        has_trade: torch.Tensor,
    ) -> torch.Tensor:
        """
        轻量融合
        """
        # Tick交互
        lob_tick, trade_tick = self.tick_interaction(lob_feat, trade_feat, has_trade)
        
        # 降采样
        lob_ds = self.lob_downsample(lob_tick)
        trade_ds, _ = self.trade_downsample(trade_tick, has_trade)
        
        # 跨模态交互
        lob_ds, trade_ds = self.cross_transformer(lob_ds, trade_ds)
        
        # 融合
        fused = self.fusion(torch.cat([lob_ds, trade_ds], dim=-1))
        
        return self.output_norm(fused)
