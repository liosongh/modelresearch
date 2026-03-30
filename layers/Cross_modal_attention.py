# =============================================================================
# Cross-Modal Attention - 跨模态注意力融合模块
# Version: v1.0
#
# 实现早期跨模态注意力，让LOB和Trade在编码过程中就互相"看到"对方
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class CrossModalAttentionBlock(nn.Module):
    """
    跨模态注意力模块
    
    让LOB特征可以query Trade信息，反之亦然。
    实现双向的特征增强。
    
    架构:
    1. Self-Attention: 各模态内部时序建模
    2. Cross-Attention: 跨模态信息交换
    3. FFN: 特征变换
    
    Args:
        d_model: 特征维度
        nhead: 注意力头数
        dropout: Dropout比率
        use_pre_norm: 是否使用Pre-LN (更稳定)
    """
    
    def __init__(
        self, 
        d_model: int, 
        nhead: int = 4, 
        dropout: float = 0.1,
        use_pre_norm: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.use_pre_norm = use_pre_norm
        
        # LOB Self-Attention
        self.lob_self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.lob_self_norm = nn.LayerNorm(d_model)
        
        # Trade Self-Attention  
        self.trade_self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.trade_self_norm = nn.LayerNorm(d_model)
        
        # LOB -> Trade Cross-Attention (LOB queries Trade)
        self.lob_cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.lob_cross_norm = nn.LayerNorm(d_model)
        
        # Trade -> LOB Cross-Attention (Trade queries LOB)
        self.trade_cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.trade_cross_norm = nn.LayerNorm(d_model)
        
        # FFN for LOB
        self.lob_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.lob_ffn_norm = nn.LayerNorm(d_model)
        
        # FFN for Trade
        self.trade_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.trade_ffn_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        lob_feat: torch.Tensor,     # (B, T, D)
        trade_feat: torch.Tensor,   # (B, T, D)
        causal_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        双向跨模态注意力
        
        Args:
            lob_feat: LOB特征序列 (B, T, D)
            trade_feat: Trade特征序列 (B, T, D)
            causal_mask: 因果掩码 (T, T), 防止看到未来
        
        Returns:
            lob_out: 增强后的LOB特征 (B, T, D)
            trade_out: 增强后的Trade特征 (B, T, D)
            attn_weights: 注意力权重字典 (用于可视化)
        """
        attn_weights = {}
        
        # ============================================================
        # Step 1: Self-Attention (各模态内部建模)
        # ============================================================
        if self.use_pre_norm:
            # Pre-LN: Norm -> Attn -> Add
            lob_normed = self.lob_self_norm(lob_feat)
            lob_self, lob_self_w = self.lob_self_attn(
                lob_normed, lob_normed, lob_normed, attn_mask=causal_mask
            )
            lob_feat = lob_feat + self.dropout(lob_self)
            
            trade_normed = self.trade_self_norm(trade_feat)
            trade_self, trade_self_w = self.trade_self_attn(
                trade_normed, trade_normed, trade_normed, attn_mask=causal_mask
            )
            trade_feat = trade_feat + self.dropout(trade_self)
        else:
            # Post-LN: Attn -> Add -> Norm
            lob_self, lob_self_w = self.lob_self_attn(
                lob_feat, lob_feat, lob_feat, attn_mask=causal_mask
            )
            lob_feat = self.lob_self_norm(lob_feat + self.dropout(lob_self))
            
            trade_self, trade_self_w = self.trade_self_attn(
                trade_feat, trade_feat, trade_feat, attn_mask=causal_mask
            )
            trade_feat = self.trade_self_norm(trade_feat + self.dropout(trade_self))
        
        attn_weights['lob_self'] = lob_self_w
        attn_weights['trade_self'] = trade_self_w
        
        # ============================================================
        # Step 2: Cross-Attention (跨模态信息交换)
        # ============================================================
        if self.use_pre_norm:
            # LOB queries Trade: "最近发生了什么交易？"
            lob_normed = self.lob_cross_norm(lob_feat)
            lob_cross, lob_cross_w = self.lob_cross_attn(
                query=lob_normed,
                key=trade_feat,
                value=trade_feat,
                attn_mask=causal_mask
            )
            lob_enhanced = lob_feat + self.dropout(lob_cross)
            
            # Trade queries LOB: "当前订单簿状态如何？"
            trade_normed = self.trade_cross_norm(trade_feat)
            trade_cross, trade_cross_w = self.trade_cross_attn(
                query=trade_normed,
                key=lob_feat,
                value=lob_feat,
                attn_mask=causal_mask
            )
            trade_enhanced = trade_feat + self.dropout(trade_cross)
        else:
            lob_cross, lob_cross_w = self.lob_cross_attn(
                query=lob_feat,
                key=trade_feat,
                value=trade_feat,
                attn_mask=causal_mask
            )
            lob_enhanced = self.lob_cross_norm(lob_feat + self.dropout(lob_cross))
            
            trade_cross, trade_cross_w = self.trade_cross_attn(
                query=trade_feat,
                key=lob_feat,
                value=lob_feat,
                attn_mask=causal_mask
            )
            trade_enhanced = self.trade_cross_norm(trade_feat + self.dropout(trade_cross))
        
        attn_weights['lob_cross'] = lob_cross_w
        attn_weights['trade_cross'] = trade_cross_w
        
        # ============================================================
        # Step 3: FFN
        # ============================================================
        if self.use_pre_norm:
            lob_out = lob_enhanced + self.dropout(self.lob_ffn(self.lob_ffn_norm(lob_enhanced)))
            trade_out = trade_enhanced + self.dropout(self.trade_ffn(self.trade_ffn_norm(trade_enhanced)))
        else:
            lob_out = self.lob_ffn_norm(lob_enhanced + self.dropout(self.lob_ffn(lob_enhanced)))
            trade_out = self.trade_ffn_norm(trade_enhanced + self.dropout(self.trade_ffn(trade_enhanced)))
        
        return lob_out, trade_out, attn_weights


class EarlyCrossModalEncoder(nn.Module):
    """
    早期跨模态编码器
    
    在CNN降采样后，立即进行跨模态信息交换，而不是等到最后才融合。
    
    架构:
    LOB CNN → Project → ┐
                        ├→ CrossModalAttn × N → Fusion → Output
    Trade CNN → Project → ┘
    
    Args:
        lob_input_dim: LOB编码器输出维度
        trade_input_dim: Trade编码器输出维度
        d_model: 统一的特征维度
        num_cross_layers: 跨模态注意力层数
        nhead: 注意力头数
        dropout: Dropout比率
        fusion_strategy: 融合策略 ('concat', 'gated', 'add')
    """
    
    def __init__(
        self,
        lob_input_dim: int,
        trade_input_dim: int,
        d_model: int = 64,
        num_cross_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,
        fusion_strategy: str = 'concat'
    ):
        super().__init__()
        
        self.d_model = d_model
        self.fusion_strategy = fusion_strategy
        
        # 投影到统一维度
        self.lob_proj = nn.Sequential(
            nn.Linear(lob_input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.trade_proj = nn.Sequential(
            nn.Linear(trade_input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 堆叠多层跨模态注意力
        self.cross_layers = nn.ModuleList([
            CrossModalAttentionBlock(d_model, nhead, dropout)
            for _ in range(num_cross_layers)
        ])
        
        # 融合层
        if fusion_strategy == 'concat':
            self.fusion = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )
        elif fusion_strategy == 'gated':
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
            self.fusion = nn.LayerNorm(d_model)
        elif fusion_strategy == 'add':
            self.fusion = nn.LayerNorm(d_model)
        else:
            raise ValueError(f"Unknown fusion_strategy: {fusion_strategy}")
        
        self._output_dim = d_model
        
    @property
    def output_dim(self) -> int:
        return self._output_dim
        
    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """生成因果掩码"""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
        
    def forward(
        self, 
        lob_encoded: torch.Tensor,   # (B, T, D_lob)
        trade_encoded: torch.Tensor, # (B, T, D_trade)
        return_intermediates: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        前向传播
        
        Args:
            lob_encoded: LOB编码器输出 (B, T, D_lob)
            trade_encoded: Trade编码器输出 (B, T, D_trade)
            return_intermediates: 是否返回中间结果
            
        Returns:
            fused: 融合后的特征 (B, T, d_model)
            info: 额外信息字典
        """
        # 1. 投影到统一维度
        lob_feat = self.lob_proj(lob_encoded)      # (B, T, d_model)
        trade_feat = self.trade_proj(trade_encoded) # (B, T, d_model)
        
        # 2. 生成因果掩码
        T = lob_feat.size(1)
        causal_mask = self._generate_causal_mask(T, lob_feat.device)
        
        # 3. 多层跨模态注意力
        all_attn_weights = []
        for cross_layer in self.cross_layers:
            lob_feat, trade_feat, weights = cross_layer(lob_feat, trade_feat, causal_mask)
            all_attn_weights.append(weights)
        
        # 4. 融合
        if self.fusion_strategy == 'concat':
            combined = torch.cat([lob_feat, trade_feat], dim=-1)  # (B, T, d_model*2)
            fused = self.fusion(combined)  # (B, T, d_model)
        elif self.fusion_strategy == 'gated':
            combined = torch.cat([lob_feat, trade_feat], dim=-1)
            gate = self.gate(combined)  # (B, T, d_model)
            fused = gate * lob_feat + (1 - gate) * trade_feat
            fused = self.fusion(fused)
        elif self.fusion_strategy == 'add':
            fused = self.fusion(lob_feat + trade_feat)
        
        info = {
            'lob_final': lob_feat,
            'trade_final': trade_feat,
        }
        
        if return_intermediates:
            info['attn_weights'] = all_attn_weights
        
        return fused, info


class CrossModalFusionWithCross(nn.Module):
    """
    结合跨模态特征和早期注意力的完整融合模块
    
    这个模块整合了:
    1. 跨模态特征 (from CrossModalFeatureBuilder)
    2. 早期跨模态注意力 (from EarlyCrossModalEncoder)
    
    Args:
        lob_input_dim: LOB编码器输出维度
        trade_input_dim: Trade编码器输出维度
        cross_feature_dim: 跨模态特征维度 (通常是6)
        d_model: 输出特征维度
        num_cross_layers: 跨模态注意力层数
        nhead: 注意力头数
        dropout: Dropout比率
    """
    
    def __init__(
        self,
        lob_input_dim: int,
        trade_input_dim: int,
        cross_feature_dim: int = 6,
        d_model: int = 64,
        num_cross_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # 跨模态特征编码器
        self.cross_feature_encoder = nn.Sequential(
            nn.Conv1d(cross_feature_dim, d_model // 2, kernel_size=3, padding=1),
            nn.GroupNorm(4, d_model // 2),
            nn.GELU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=1),
            nn.GroupNorm(4, d_model),
            nn.GELU(),
        )
        
        # 早期跨模态注意力
        self.early_cross_encoder = EarlyCrossModalEncoder(
            lob_input_dim=lob_input_dim,
            trade_input_dim=trade_input_dim,
            d_model=d_model,
            num_cross_layers=num_cross_layers,
            nhead=nhead,
            dropout=dropout,
            fusion_strategy='gated'
        )
        
        # 最终融合
        self.final_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self._output_dim = d_model
        
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    def forward(
        self,
        lob_encoded: torch.Tensor,    # (B, T, D_lob)
        trade_encoded: torch.Tensor,  # (B, T, D_trade)
        cross_features: torch.Tensor, # (B, 6, T_original)
        downsample_ratio: int = 1     # 如果cross_features需要下采样
    ) -> Tuple[torch.Tensor, Dict]:
        """
        前向传播
        
        Args:
            lob_encoded: LOB编码器输出 (B, T, D_lob)
            trade_encoded: Trade编码器输出 (B, T, D_trade)
            cross_features: 跨模态特征 (B, 6, T_original)
            downsample_ratio: 下采样比率
            
        Returns:
            fused: 融合特征 (B, T, d_model)
            info: 额外信息
        """
        B, T, _ = lob_encoded.shape
        
        # 1. 编码跨模态特征
        cross_encoded = self.cross_feature_encoder(cross_features)  # (B, d_model, T_original)
        
        # 2. 下采样跨模态特征以对齐
        if downsample_ratio > 1:
            cross_encoded = F.avg_pool1d(
                cross_encoded, 
                kernel_size=downsample_ratio, 
                stride=downsample_ratio
            )
        
        # 确保时间维度对齐
        if cross_encoded.size(2) != T:
            # 使用插值对齐
            cross_encoded = F.interpolate(cross_encoded, size=T, mode='linear', align_corners=False)
        
        cross_encoded = cross_encoded.permute(0, 2, 1)  # (B, T, d_model)
        
        # 3. 早期跨模态注意力
        modal_fused, modal_info = self.early_cross_encoder(
            lob_encoded, trade_encoded, return_intermediates=False
        )
        
        # 4. 最终融合
        combined = torch.cat([modal_fused, cross_encoded], dim=-1)  # (B, T, d_model*2)
        fused = self.final_fusion(combined)  # (B, T, d_model)
        
        info = {
            'modal_fused': modal_fused,
            'cross_encoded': cross_encoded,
            **modal_info
        }
        
        return fused, info
