# =============================================================================
# Multi-Modal Transformer V3 - 集成多模态Transformer
# Version: v3.0
#
# 核心设计思想：
# 1. 使用结构化LOB编码器和稀疏感知Trade编码器
# 2. 支持分层融合和事件驱动融合
# 3. 灵活的配置选项
# 4. 完整的Pipeline: 编码 → 融合 → Backbone → 输出
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

# 编码器
from .encoders.structured_lob_encoder import (
    StructuredLOBEncoder, 
    StructuredLOBEncoderLite,
    LOBContextSummarizer,
)
from .encoders.sparse_trade_encoder import (
    SparseAwareTradeEncoder,
    SparseAwareTradeEncoderLite,
    ConditionalTradeEncoder,
)
from .encoders.lob_encoder import LOBEncoder, LOBEncoderV2
from .encoders.trade_encoder import TradeEncoder

# 融合模块
from .fusion.hierarchical_fusion import (
    HierarchicalFusion,
    HierarchicalFusionLite,
    CrossModalTransformer,
    TickLevelInteraction,
)
from .fusion.event_driven_fusion import (
    EventDrivenFusion,
    EventDrivenFusionFast,
    IntegratedFusion,
)
from .fusion.feature_fusion import FeatureFusion

# Backbone
from .backbones.transformer import TransformerBackbone

# Pooling
from .pooling.temporal_pooling import create_pooling

# RevIN
from .revin import RevIN


class OutputHead(nn.Module):
    """
    输出头
    
    支持分类和回归任务
    """
    
    def __init__(
        self,
        d_model: int,
        num_classes: int = 3,
        task_type: str = 'classification',
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.task_type = task_type
        
        # 池化 (取最后一个时间步)
        self.pool = 'last'
        
        # 分类头
        if task_type == 'classification':
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes),
            )
        else:
            # 回归头
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            output: (B, num_classes) or (B, 1)
        """
        # 池化
        if self.pool == 'last':
            pooled = x[:, -1, :]  # (B, D)
        elif self.pool == 'mean':
            pooled = x.mean(dim=1)
        else:
            pooled = x[:, -1, :]
        
        return self.head(pooled)


class MultiModalTransformerV3(nn.Module):
    """
    多模态Transformer V3
    
    完整的Pipeline:
    1. LOB编码 (结构化/普通)
    2. Trade编码 (稀疏感知/条件/普通)
    3. 融合 (分层/事件驱动/简单)
    4. Transformer Backbone
    5. 输出头
    
    Args:
        # LOB配置
        lob_config: LOB编码器配置
        
        # Trade配置
        trade_config: Trade编码器配置
        
        # 融合配置
        fusion_config: 融合模块配置
        
        # Backbone配置
        backbone_config: Transformer配置
        
        # 输出配置
        output_config: 输出头配置
    """
    
    def __init__(
        self,
        # LOB配置
        lob_config: Optional[Dict] = None,
        
        # Trade配置
        trade_config: Optional[Dict] = None,
        
        # 融合配置
        fusion_config: Optional[Dict] = None,
        
        # 池化配置 (新增)
        pooling_config: Optional[Dict] = None,
        
        # Backbone配置
        backbone_config: Optional[Dict] = None,
        
        # 输出配置
        output_config: Optional[Dict] = None,
        
        # 通用配置
        d_model: int = 64,
        dropout: float = 0.1,
        use_revin: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.use_revin = use_revin
        
        # 默认配置
        lob_config = lob_config or {}
        trade_config = trade_config or {}
        fusion_config = fusion_config or {}
        pooling_config = pooling_config or {}
        backbone_config = backbone_config or {}
        output_config = output_config or {}
        
        # ================================================================
        # Part 1: LOB编码器
        # ================================================================
        lob_type = lob_config.get('type', 'structured')
        
        if lob_type == 'structured':
            self.lob_encoder = StructuredLOBEncoder(
                in_channels=lob_config.get('in_channels', 4),
                num_levels=lob_config.get('num_levels', 10),
                d_model=d_model,
                context_dim=lob_config.get('context_dim', 32),
                time_strides=lob_config.get('time_strides', None),
                nhead=lob_config.get('nhead', 4),
                use_level_transformer=lob_config.get('use_level_transformer', True),
                dropout=dropout,
            )
        elif lob_type == 'structured_lite':
            self.lob_encoder = StructuredLOBEncoderLite(
                in_channels=lob_config.get('in_channels', 4),
                num_levels=lob_config.get('num_levels', 10),
                d_model=d_model,
                time_strides=lob_config.get('time_strides', [5, 2, 2]),
                dropout=dropout,
            )
        elif lob_type == 'v2':
            self.lob_encoder = LOBEncoderV2(
                in_channels=lob_config.get('in_channels', 4),
                base_channels=lob_config.get('base_channels', 32),
                d_model=d_model,
                time_strides=lob_config.get('time_strides', [5, 5, 2]),
                num_levels=lob_config.get('num_levels', 10),
                dropout=dropout,
            )
        else:  # 'original'
            self.lob_encoder = LOBEncoder(
                in_channels=lob_config.get('in_channels', 4),
                base_channels=lob_config.get('base_channels', 32),
                num_layers=lob_config.get('num_layers', 3),
                time_strides=lob_config.get('time_strides', [2, 5, 1]),
                level_strides=lob_config.get('level_strides', [2, 2, 2]),
                dropout=dropout,
            )
        
        # LOB上下文摘要器 (用于条件Trade编码)
        self.lob_context = LOBContextSummarizer(
            in_channels=lob_config.get('in_channels', 4),
            num_levels=lob_config.get('num_levels', 10),
            context_dim=lob_config.get('context_dim', 32),
            use_learnable=True,
        )
        
        # ================================================================
        # Part 2: Trade编码器
        # ================================================================
        trade_type = trade_config.get('type', 'sparse_aware')
        
        if use_revin:
            # RevIN for continuous features
            self.trade_revin = RevIN(
                num_features=trade_config.get('in_features', 12) - 2,  # 排除has_trade
                affine=True
            )
        else:
            self.trade_revin = None
        
        if trade_type == 'sparse_aware':
            self.trade_encoder = SparseAwareTradeEncoder(
                in_features=trade_config.get('in_features', 12),
                d_model=d_model,
                time_strides=trade_config.get('time_strides', None),
                dropout=dropout,
            )
            self.use_conditional_trade = False
        elif trade_type == 'sparse_aware_lite':
            self.trade_encoder = SparseAwareTradeEncoderLite(
                in_features=trade_config.get('in_features', 12),
                d_model=d_model,
                time_strides=trade_config.get('time_strides', [5, 2, 2]),
                dropout=dropout,
            )
            self.use_conditional_trade = False
        elif trade_type == 'conditional':
            self.trade_encoder = ConditionalTradeEncoder(
                in_features=trade_config.get('in_features', 12),
                lob_context_dim=lob_config.get('context_dim', 32),
                d_model=d_model,
                time_strides=trade_config.get('time_strides', [5, 2, 2]),
                dropout=dropout,
            )
            self.use_conditional_trade = True
        elif trade_type == 'none':
            self.trade_encoder = None
            self.use_conditional_trade = False
        else:  # 'original'
            self.trade_encoder = TradeEncoder(
                in_features=trade_config.get('in_features', 12),
                hidden_channels=trade_config.get('hidden_channels', [16, 16, 16]),
                kernel_size=trade_config.get('kernel_size', [5, 2, 2]),
                time_stride=trade_config.get('time_stride', [5, 2, 2]),
                dropout=dropout,
            )
            self.use_conditional_trade = False
        
        # ================================================================
        # Part 3: 融合模块
        # ================================================================
        fusion_type = fusion_config.get('type', 'hierarchical')
        
        if fusion_type == 'hierarchical':
            self.fusion = HierarchicalFusion(
                d_model=d_model,
                nhead=fusion_config.get('nhead', 4),
                num_cross_layers=fusion_config.get('num_cross_layers', 2),
                scales=fusion_config.get('scales', [1, 5, 4]),
                dropout=dropout,
            )
            self.fusion_type = 'hierarchical'
        elif fusion_type == 'hierarchical_lite':
            self.fusion = HierarchicalFusionLite(
                d_model=d_model,
                nhead=fusion_config.get('nhead', 4),
                scale_factor=fusion_config.get('scale_factor', 5),
                dropout=dropout,
            )
            self.fusion_type = 'hierarchical_lite'
        elif fusion_type == 'event_driven':
            self.fusion = EventDrivenFusion(
                d_model=d_model,
                state_dim=fusion_config.get('state_dim', 32),
                nhead=fusion_config.get('nhead', 4),
                dropout=dropout,
            )
            self.fusion_type = 'event_driven'
        elif fusion_type == 'event_driven_fast':
            self.fusion = EventDrivenFusionFast(
                d_model=d_model,
                nhead=fusion_config.get('nhead', 4),
                dropout=dropout,
            )
            self.fusion_type = 'event_driven_fast'
        elif fusion_type == 'integrated':
            self.fusion = IntegratedFusion(
                d_model=d_model,
                state_dim=fusion_config.get('state_dim', 32),
                nhead=fusion_config.get('nhead', 4),
                dropout=dropout,
            )
            self.fusion_type = 'integrated'
        elif fusion_type == 'none':
            self.fusion = None
            self.fusion_type = 'none'
        else:  # 'simple' - 使用原有FeatureFusion
            self.fusion = FeatureFusion(
                strategy=fusion_config.get('strategy', 'concat'),
                lob_dim=self.lob_encoder.output_dim,
                trade_dim=self.trade_encoder.output_dim,
                d_model=d_model,
                dropout=dropout,
            )
            self.fusion_type = 'simple'
        
        # 特征对齐 (如果编码器输出维度不同)
        lob_out_dim = self.lob_encoder.output_dim
        # trade_out_dim = self.trade_encoder.output_dim
        
        if lob_out_dim != d_model:
            self.lob_proj = nn.Linear(lob_out_dim, d_model)
        else:
            self.lob_proj = nn.Identity()
            
        # if trade_out_dim != d_model:
        #     self.trade_proj = nn.Linear(trade_out_dim, d_model)
        # else:
        #     self.trade_proj = nn.Identity()
        
        # ================================================================
        # Part 4: 池化模块 (可选，用于减少序列长度)
        # ================================================================
        self.use_pooling = pooling_config.get('enabled', False)
        
        if self.use_pooling:
            pool_type = pooling_config.get('type', 'causal_conv')
            self.pooling = create_pooling(
                pool_type=pool_type,
                d_model=d_model,
                pool_factor=pooling_config.get('pool_factor', 5),
                output_len=pooling_config.get('output_len', None),
                nhead=pooling_config.get('nhead', 4),
                dropout=dropout,
                num_layers=pooling_config.get('num_layers', 2),
            )
        else:
            self.pooling = None
        
        # ================================================================
        # Part 5: Transformer Backbone
        # ================================================================
        self.backbone = TransformerBackbone(
            d_model=d_model,
            nhead=backbone_config.get('nhead', 4),
            num_layers=backbone_config.get('num_layers', 2),
            dim_feedforward=backbone_config.get('dim_feedforward', 256),
            dropout=backbone_config.get('dropout', dropout),
            positional_encoding=backbone_config.get('positional_encoding', 'sinusoidal'),
            max_seq_len=backbone_config.get('max_seq_len', 500),
        )
        
        # ================================================================
        # Part 6: 输出头
        # ================================================================
        self.output_head = OutputHead(
            d_model=d_model,
            num_classes=output_config.get('num_classes', 3),
            task_type=output_config.get('task_type', 'classification'),
            dropout=dropout,
        )
        
        # 记录配置
        self.config = {
            'lob': lob_config,
            'trade': trade_config,
            'fusion': fusion_config,
            'pooling': pooling_config,
            'backbone': backbone_config,
            'output': output_config,
            'd_model': d_model,
            'dropout': dropout,
            'use_revin': use_revin,
        }
        
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        return_intermediates: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播
        
        Args:
            inputs: dict with:
                - 'lob': (B, 4, T, 10) LOB数据
                - 'trade': (B, F, T) Trade数据
                - 'has_trade': (B, T) 可选，交易发生标记
            return_intermediates: 是否返回中间结果
            
        Returns:
            output: (B, num_classes) 分类概率
            或 dict with 各阶段输出
        """
        lob = inputs['lob']      # (B, 4, T, 10)
        trade = inputs['trade']  # (B, F, T)
        

        
        # B, _, T = trade.shape
        

        
        # ================================================================
        # Step 1: LOB编码
        # ================================================================
        lob_encoded = self.lob_encoder(lob)  # (B, T_out, D_lob)
        lob_encoded = self.lob_proj(lob_encoded)  # (B, T_out, d_model)

        # ================================================================
        # Step 2: Trade编码
        # ================================================================
        if trade is not None and self.trade_encoder is not None :
            # 推断has_trade
            has_trade = inputs.get('has_trade', None)
            if has_trade is None:
                has_trade = (trade[:, 0, :] > 0.5).float()
            # ================================================================
            # 2.1 RevIN for Trade
            # ================================================================
            if self.trade_revin is not None:
                binary_feat = trade[:, 0:2, :]
                continuous_feat = trade[:, 2:, :]
                continuous_feat = self.trade_revin(continuous_feat, mode='norm')
                trade = torch.cat([binary_feat, continuous_feat], dim=1)
            # 2.2 LOB上下文 (用于条件编码)
            lob_context = self.lob_context(lob)  # (B, T, context_dim)
            
            # 2.3 Trade编码
            if self.use_conditional_trade:
                trade_encoded = self.trade_encoder(trade, lob_context)
            else:
                trade_encoded = self.trade_encoder(trade)
            trade_encoded = self.trade_proj(trade_encoded)  # (B, T_out, d_model)
        else:
            trade_encoded = None
        
        # # 确保时间维度对齐
        # T_out = lob_encoded.size(1)
        # if trade_encoded.size(1) != T_out:
        #     trade_encoded = F.interpolate(
        #         trade_encoded.permute(0, 2, 1),
        #         size=T_out,
        #         mode='linear',
        #         align_corners=False
        #     ).permute(0, 2, 1)
        
        # # 降采样has_trade
        # if has_trade.size(1) != T_out:
        #     has_trade_down = F.max_pool1d(
        #         has_trade.unsqueeze(1),
        #         kernel_size=has_trade.size(1) // T_out,
        #         stride=has_trade.size(1) // T_out
        #     ).squeeze(1)
        #     # 确保长度对齐
        #     if has_trade_down.size(1) != T_out:
        #         has_trade_down = F.interpolate(
        #             has_trade_down.unsqueeze(1),
        #             size=T_out,
        #             mode='nearest'
        #         ).squeeze(1)
        # else:
        #     has_trade_down = has_trade
        
        # ================================================================
        # Step 3: 融合
        # ================================================================
        if self.fusion_type in ['hierarchical', 'event_driven']:
            fusion_output = self.fusion(lob_encoded, trade_encoded, has_trade)
            fused = fusion_output['fused']
        elif self.fusion_type in ['hierarchical_lite', 'event_driven_fast', 'integrated']:
            fused = self.fusion(lob_encoded, trade_encoded, has_trade)
        elif self.fusion_type == 'none':
            fused = lob_encoded
        else:  # simple
            fused = self.fusion(lob_encoded, trade_encoded)
        
        # # ================================================================
        # # Step 5: 池化 (可选，减少序列长度)
        # # ================================================================
        # if self.use_pooling and self.pooling is not None:
        #     # 降采样has_trade以匹配fused的长度
        #     T_fused = fused.size(1)
        #     if has_trade.size(1) != T_fused:
        #         # 使用max_pool保留事件信息
        #         pool_ratio = has_trade.size(1) // T_fused
        #         if pool_ratio > 1:
        #             has_trade_pooled = F.max_pool1d(
        #                 has_trade.unsqueeze(1),
        #                 kernel_size=pool_ratio,
        #                 stride=pool_ratio
        #             ).squeeze(1)
        #             # 确保长度对齐
        #             if has_trade_pooled.size(1) > T_fused:
        #                 has_trade_pooled = has_trade_pooled[:, :T_fused]
        #             elif has_trade_pooled.size(1) < T_fused:
        #                 has_trade_pooled = F.pad(has_trade_pooled, (0, T_fused - has_trade_pooled.size(1)))
        #         else:
        #             has_trade_pooled = has_trade[:, :T_fused]
        #     else:
        #         has_trade_pooled = has_trade
            
        #     fused = self.pooling(fused, has_trade_pooled)

        
        # ================================================================
        # Step 6: Transformer Backbone
        # ================================================================
        backbone_output = self.backbone(fused)  # (B, T_out, d_model)
        
        # ================================================================
        # Step 7: 输出
        # ================================================================
        output = self.output_head(backbone_output)
        
        # if return_intermediates:
        #     return {
        #         'output': output,
        #         'lob_encoded': lob_encoded,
        #         'trade_encoded': trade_encoded,
        #         'lob_context': lob_context,
        #         'fused': fused,
        #         'backbone_output': backbone_output,
        #         'has_trade': has_trade,
        #     }
        
        return output
    
    def get_encoder_output_dim(self) -> int:
        """获取编码器输出维度"""
        return self.d_model
    
    def get_downsample_ratio(self) -> int:
        """获取总降采样率"""
        return self.lob_encoder.downsample_ratio


def create_model_v3(
    preset: str = 'default',
    **kwargs
) -> MultiModalTransformerV3:
    """
    创建V3模型的工厂函数
    
    预设配置:
    - 'default': 默认配置
    - 'lite': 轻量配置
    - 'full': 完整配置
    - 'fast': 快速配置
    """
    
    if preset == 'default':
        config = {
            'lob_config': {
                'type': 'structured',
                'in_channels': 4,
                'num_levels': 10,
                'context_dim': 32,
                'time_strides': [5, 2, 2],
                'use_level_transformer': True,
            },
            'trade_config': {
                'type': 'sparse_aware',
                'in_features': 12,
                'time_strides': [5, 2, 2],
            },
            'fusion_config': {
                'type': 'hierarchical',
                'nhead': 4,
                'num_cross_layers': 2,
                'scales': [1, 5, 4],
            },
            'pooling_config': {
                'enabled': False,  # 默认不启用
                'type': 'event_aware',
                'pool_factor': 5,
            },
            'backbone_config': {
                'nhead': 4,
                'num_layers': 2,
                'dim_feedforward': 256,
            },
            'output_config': {
                'num_classes': 3,
                'task_type': 'classification',
            },
            'd_model': 64,
            'dropout': 0.1,
            'use_revin': True,
        }
    elif preset == 'lite':
        config = {
            'lob_config': {
                'type': 'structured_lite',
                'time_strides': [5, 2, 2],
            },
            'trade_config': {
                'type': 'sparse_aware_lite',
                'time_strides': [5, 2, 2],
            },
            'fusion_config': {
                'type': 'hierarchical_lite',
                'scale_factor': 5,
            },
            'pooling_config': {
                'enabled': False,
            },
            'backbone_config': {
                'num_layers': 1,
            },
            'd_model': 32,
            'dropout': 0.1,
        }
    elif preset == 'full':
        config = {
            'lob_config': {
                'type': 'structured',
                'context_dim': 64,
                'time_strides': [5, 2, 2],
                'use_level_transformer': True,
                'nhead': 8,
            },
            'trade_config': {
                'type': 'conditional',
                'time_strides': [5, 2, 2],
            },
            'fusion_config': {
                'type': 'hierarchical',
                'nhead': 8,
                'num_cross_layers': 3,
            },
            'pooling_config': {
                'enabled': True,  # full模式启用池化
                'type': 'hybrid',
                'pool_factor': 5,
                'output_len': 30,
            },
            'backbone_config': {
                'nhead': 8,
                'num_layers': 4,
                'dim_feedforward': 512,
            },
            'd_model': 128,
            'dropout': 0.1,
        }
    elif preset == 'fast':
        config = {
            'lob_config': {
                'type': 'original',
                'time_strides': [2, 5, 2],
            },
            'trade_config': {
                'type': 'original',
                'time_stride': [5, 2, 2],
            },
            'fusion_config': {
                'type': 'event_driven_fast',
            },
            'pooling_config': {
                'enabled': True,  # fast模式启用池化以加速
                'type': 'conv',
                'pool_factor': 5,
            },
            'backbone_config': {
                'num_layers': 1,
            },
            'd_model': 32,
        }
    else:
        config = {}
    
    # 覆盖用户提供的配置
    for key, value in kwargs.items():
        if key in config and isinstance(value, dict):
            config[key].update(value)
        else:
            config[key] = value
    
    return MultiModalTransformerV3(**config)
