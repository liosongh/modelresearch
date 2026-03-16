# =============================================================================
# Structured LOB Encoder - 结构化限价订单簿编码器
# Version: v1.0
#
# 核心设计思想：
# 1. 价格流和数量流分开处理（scale不同）
# 2. Level间用注意力建模（非简单卷积）
# 3. 构造有金融意义的手工特征
# 4. 保持LOB的结构信息
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from .base_encoder import BaseEncoder
import math


class LOBContextSummarizer(nn.Module):
    """
    从LOB数据中提取市场状态摘要
    
    提取的特征包括:
    1. 盘口特征 (best bid/ask, spread)
    2. 深度特征 (多档累计量)
    3. 不平衡特征 (bid/ask volume imbalance)
    4. 压力特征 (各档位的流动性分布)
    
    Input: LOB raw data (B, 4, T, 10)
           4 channels: ask_price, bid_price, ask_notional, bid_notional
           10 levels
    Output: Context summary (B, T, context_dim)
    """
    
    def __init__(
        self, 
        in_channels: int = 4,
        num_levels: int = 10,
        context_dim: int = 32,
        use_learnable: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_levels = num_levels
        self.context_dim = context_dim
        self.use_learnable = use_learnable
        
        # 通道索引 (根据你的数据格式)
        self.ASK_P, self.BID_P = 0, 1
        self.ASK_V, self.BID_V = 2, 3
        
        # 手工特征维度
        self.handcraft_dim = 15
        
        if use_learnable:
            # 可学习的特征提取
            # Level-wise attention: 学习每个档位的重要性
            self.level_attention = nn.Sequential(
                nn.Linear(in_channels, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )
            
            # 特征变换
            self.feature_transform = nn.Sequential(
                nn.Linear(in_channels * num_levels, context_dim),
                nn.LayerNorm(context_dim),
                nn.GELU(),
            )
            
            # 手工特征 + 学习特征融合
            self.fusion = nn.Sequential(
                nn.Linear(self.handcraft_dim + context_dim, context_dim),
                nn.LayerNorm(context_dim),
                nn.GELU(),
            )
        else:
            self.fusion = nn.Sequential(
                nn.Linear(self.handcraft_dim, context_dim),
                nn.LayerNorm(context_dim),
                nn.GELU(),
            )
    
    def compute_handcraft_features(self, lob: torch.Tensor) -> torch.Tensor:
        """
        计算基于领域知识的手工特征
        
        Args:
            lob: (B, 4, T, 10)
        Returns:
            features: (B, T, 15)
        """
        B, C, T, L = lob.shape
        
        ask_p = lob[:, self.ASK_P, :, :]  # (B, T, 10)
        bid_p = lob[:, self.BID_P, :, :]
        ask_v = lob[:, self.ASK_V, :, :]
        bid_v = lob[:, self.BID_V, :, :]
        
        features = []
        
        # 1. Mid price (中间价) - 归一化
        mid_price = (ask_p[:, :, 0] + bid_p[:, :, 0]) / 2  # (B, T)
        mid_price_norm = mid_price / (mid_price.mean(dim=1, keepdim=True) + 1e-9) - 1
        features.append(mid_price_norm.unsqueeze(-1))
        
        # 2. Spread (价差) - 基点
        spread = ask_p[:, :, 0] - bid_p[:, :, 0]  # (B, T)
        spread_bps = spread / (mid_price + 1e-9) * 10000
        features.append(spread_bps.unsqueeze(-1))
        
        # 3. Top-of-book imbalance (盘口不平衡)
        tob_imb = (bid_v[:, :, 0] - ask_v[:, :, 0]) / (bid_v[:, :, 0] + ask_v[:, :, 0] + 1e-9)
        features.append(tob_imb.unsqueeze(-1))
        
        # 4-6. Multi-level imbalance (多档不平衡)
        for levels in [3, 5, 10]:
            bid_depth = bid_v[:, :, :levels].sum(dim=-1)
            ask_depth = ask_v[:, :, :levels].sum(dim=-1)
            imb = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-9)
            features.append(imb.unsqueeze(-1))
        
        # 7-8. Depth (总深度) - 对数化
        total_bid = torch.log1p(bid_v.sum(dim=-1))
        total_ask = torch.log1p(ask_v.sum(dim=-1))
        features.append(total_bid.unsqueeze(-1))
        features.append(total_ask.unsqueeze(-1))
        
        # 9. Weighted price distance (加权价距) - Ask侧
        ask_weights = ask_v / (ask_v.sum(dim=-1, keepdim=True) + 1e-9)
        ask_price_dist = ((ask_p - ask_p[:, :, 0:1]) * ask_weights).sum(dim=-1)
        ask_price_dist_rel = ask_price_dist / (mid_price + 1e-9) * 10000
        features.append(ask_price_dist_rel.unsqueeze(-1))
        
        # 10. Weighted price distance - Bid侧
        bid_weights = bid_v / (bid_v.sum(dim=-1, keepdim=True) + 1e-9)
        bid_price_dist = ((bid_p[:, :, 0:1] - bid_p) * bid_weights).sum(dim=-1)
        bid_price_dist_rel = bid_price_dist / (mid_price + 1e-9) * 10000
        features.append(bid_price_dist_rel.unsqueeze(-1))
        
        # 11-12. Concentration (流动性集中度) - 第1档占比
        depth_all_bid = bid_v.sum(dim=-1)
        depth_all_ask = ask_v.sum(dim=-1)
        bid_conc = bid_v[:, :, 0] / (depth_all_bid + 1e-9)
        ask_conc = ask_v[:, :, 0] / (depth_all_ask + 1e-9)
        features.append(bid_conc.unsqueeze(-1))
        features.append(ask_conc.unsqueeze(-1))
        
        # 13. Price pressure (价格压力)
        pressure = (bid_v[:, :, 0] * bid_p[:, :, 0] - ask_v[:, :, 0] * ask_p[:, :, 0]) / \
                   (bid_v[:, :, 0] + ask_v[:, :, 0] + 1e-9)
        pressure_rel = pressure / (mid_price + 1e-9) * 10000
        features.append(pressure_rel.unsqueeze(-1))
        
        # 14-15. Depth derivative (深度变化) - 第1档vs第2档
        depth_deriv_bid = bid_v[:, :, 0] - bid_v[:, :, 1]
        depth_deriv_ask = ask_v[:, :, 0] - ask_v[:, :, 1]
        features.append((torch.log1p(torch.abs(depth_deriv_bid)) * torch.sign(depth_deriv_bid)).unsqueeze(-1))
        features.append((torch.log1p(torch.abs(depth_deriv_ask)) * torch.sign(depth_deriv_ask)).unsqueeze(-1))
        
        return torch.cat(features, dim=-1)  # (B, T, 15)
    
    def forward(self, lob: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lob: (B, 4, T, 10)
        Returns:
            context: (B, T, context_dim)
        """
        B, C, T, L = lob.shape
        
        # 手工特征
        handcraft = self.compute_handcraft_features(lob)  # (B, T, 15)
        
        if self.use_learnable:
            # 可学习特征
            lob_reshaped = lob.permute(0, 2, 3, 1)  # (B, T, 10, 4)
            
            # Level attention weights
            attn_scores = self.level_attention(lob_reshaped)  # (B, T, 10, 1)
            attn_weights = F.softmax(attn_scores, dim=2)      # (B, T, 10, 1)
            
            # Weighted features
            weighted = lob_reshaped * attn_weights  # (B, T, 10, 4)
            
            # Flatten and transform
            flat = weighted.reshape(B, T, -1)  # (B, T, 40)
            learned = self.feature_transform(flat)  # (B, T, context_dim)
            
            # 融合
            combined = torch.cat([handcraft, learned], dim=-1)
            context = self.fusion(combined)
        else:
            context = self.fusion(handcraft)
        
        return context  # (B, T, context_dim)


class CausalTimeBlock(nn.Module):
    """因果时间降采样块"""
    
    def __init__(self, in_dim: int, out_dim: int, stride: int, dropout: float = 0.1):
        super().__init__()
        
        kernel_size = max(3, stride)
        self.pad = kernel_size - 1
        
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size, stride=stride)
        self.norm = nn.GroupNorm(1, out_dim)  # LayerNorm equivalent
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = F.pad(x, (self.pad, 0))  # 因果padding
        x = self.conv(x)
        x = self.norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x


class LevelTransformerEncoder(nn.Module):
    """
    Level Transformer编码器
    将10档LOB视为10个token，用Transformer学习依赖关系
    """
    
    def __init__(
        self,
        in_features: int = 4,  # 每个level: ask_p, bid_p, ask_v, bid_v (已处理)
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Level位置编码
        self.level_embedding = nn.Embedding(20, d_model // 2)  # 10 bid + 10 ask
        
        # 边类型编码 (bid vs ask)
        self.side_embedding = nn.Embedding(2, d_model // 2)
        
        # 特征投影
        self.input_proj = nn.Linear(2, d_model)  # (rel_price, log_volume)
        
        # Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            # norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 聚合层 (用可学习query)
        self.agg_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.agg_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.agg_norm = nn.LayerNorm(d_model)
        
    def forward(self, rel_ask_p,rel_bid_p,log_ask_v,log_bid_v) -> torch.Tensor:
        """
        Args:
            lob: (B, 4, T, 10) - [ask_p, bid_p, ask_v, bid_v]
        Returns:
            encoded: (B, T, d_model)
        """
        B,  T, L = rel_ask_p.shape
        device = rel_ask_p.device
        
        # # 提取各通道
        # ask_p = lob[:, 0, :, :]  # (B, T, 10)
        # bid_p = lob[:, 1, :, :]
        # ask_v = lob[:, 2, :, :]
        # bid_v = lob[:, 3, :, :]
        
        # # 中间价
        # mid = (ask_p[:, :, 0:1] + bid_p[:, :, 0:1]) / 2  # (B, T, 1)
        
        # # 相对价格 (bps)
        # rel_ask_p = (ask_p - mid) / (mid + 1e-9) * 10000  # (B, T, 10)
        # rel_bid_p = (mid - bid_p) / (mid + 1e-9) * 10000
        
        # # 对数量
        # log_ask_v = torch.log1p(ask_v)  # (B, T, 10)
        # log_bid_v = torch.log1p(bid_v)
        
        # 构造tokens: bid levels (0-9) + ask levels (10-19)
        bid_tokens = torch.stack([rel_bid_p, log_bid_v], dim=-1)  # (B, T, 10, 2)
        ask_tokens = torch.stack([rel_ask_p, log_ask_v], dim=-1)  # (B, T, 10, 2)
        all_tokens = torch.cat([bid_tokens, ask_tokens], dim=2)   # (B, T, 20, 2)
        
        # 位置和边编码
        level_ids = torch.arange(L * 2, device=device)  # [0..19]
        level_emb = self.level_embedding(level_ids)     # (20, d//2)
        
        side_ids = torch.tensor([0]*L + [1]*L, device=device)
        side_emb = self.side_embedding(side_ids)        # (20, d//2)
        
        pos_emb = torch.cat([level_emb, side_emb], dim=-1)  # (20, d)
        
        # 批量处理
        BT = B * T
        tokens_flat = all_tokens.reshape(BT, L * 2, 2)  # (B*T, 20, 2)
        
        # 投影
        token_feat = self.input_proj(tokens_flat)  # (B*T, 20, d)
        token_feat = token_feat + pos_emb.unsqueeze(0)  # 广播位置编码
        
        # Transformer
        encoded = self.transformer(token_feat)  # (B*T, 20, d)
        
        # 聚合
        query = self.agg_query.expand(BT, -1, -1)  # (B*T, 1, d)
        aggregated, _ = self.agg_attn(query, encoded, encoded)  # (B*T, 1, d)
        aggregated = self.agg_norm(aggregated.squeeze(1))  # (B*T, d)
        
        return aggregated.reshape(B, T, -1)  # (B, T, d)


class StructuredLOBEncoder(BaseEncoder):
    """
    结构化LOB编码器
    
    核心思想：
    1. 价格流和数量流分开处理 (scale不同)
    2. Level间用注意力建模 (非简单卷积)
    3. 构造有金融意义的特征
    4. 保持LOB的结构信息
    
    Input: (B, 4, T, 10) - 4通道(ask_p, bid_p, ask_v, bid_v), 10档
    Output: (B, T_out, d_model)
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        num_levels: int = 10,
        d_model: int = 64,
        context_dim: int = 32,
        context_learnable: bool = True,
        time_strides: List[int] = None,  # 总共20x降采样
        nhead: int = 4,
        use_level_transformer: bool = True,
        dropout: float = 0.1,
        
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_levels = num_levels
        self.d_model = d_model
        self.use_level_transformer = use_level_transformer
        if time_strides is not None:
            # 计算总降采样率
            self._downsample_ratio = 1
            for s in time_strides:
                self._downsample_ratio *= s
        else:
            self._downsample_ratio = 1
        
        # ================================================================
        # Part 1: LOB上下文摘要 (手工特征)
        # ================================================================
        self.context_summarizer = LOBContextSummarizer(
            in_channels=in_channels,
            num_levels=num_levels,
            context_dim=context_dim,
            use_learnable=True,
        )
        
        # ================================================================
        # Part 2: 双流处理 - 价格流 & 数量流
        # ================================================================
        # 价格流: 捕获价格结构
        self.price_stream = nn.Sequential(
            nn.Linear(num_levels * 2, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
        )
        
        # 数量流: 捕获流动性分布
        self.volume_stream = nn.Sequential(
            nn.Linear(num_levels * 2, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
        )
        
        # ================================================================
        # Part 3: Level Transformer (可选)
        # ================================================================
        if use_level_transformer:
            self.level_transformer = LevelTransformerEncoder(
                in_features=2,
                d_model=d_model // 2,
                nhead=nhead // 2 if nhead >= 4 else 2,
                num_layers=2,
                dropout=dropout,
            )
            combined_dim = context_dim + d_model // 2 + d_model // 2 + d_model // 2
        else:
            self.level_transformer = None
            combined_dim = context_dim + d_model // 2 + d_model // 2
        
        # ================================================================
        # Part 4: 特征融合
        # ================================================================
        self.feature_fusion = nn.Sequential(
            nn.Linear(combined_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # # ================================================================
        # # Part 5: 时间降采样 (因果卷积)
        # # ================================================================
        # self.time_blocks = nn.ModuleList()
        # current_dim = d_model
        
        # for i, stride in enumerate(time_strides):
        #     out_dim = d_model
        #     self.time_blocks.append(
        #         CausalTimeBlock(current_dim, out_dim, stride, dropout)
        #     )
        #     current_dim = out_dim
        
        self.output_norm = nn.LayerNorm(d_model)
        self._output_dim = d_model
        
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    @property
    def downsample_ratio(self) -> int:
        return self._downsample_ratio
        
    def forward(self, lob: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lob: (B, 4, T, 10)
        Returns:
            encoded: (B, T_out, d_model)
        """
        B, C, T, L = lob.shape
        
        # ================================================================
        # Step 1: 上下文摘要 (手工特征)
        # ================================================================
        context = self.context_summarizer(lob)  # (B, T, context_dim)
        
        # ================================================================
        # Step 2: 双流处理
        # ================================================================
        ask_p = lob[:, 0, :, :]  # (B, T, 10)
        bid_p = lob[:, 1, :, :]
        ask_v = lob[:, 2, :, :]
        bid_v = lob[:, 3, :, :]
        
        # 中间价
        mid = (ask_p[:, :, 0:1] + bid_p[:, :, 0:1]) / 2
        
        # 价格流: 相对价格 (bps)
        rel_ask_p = (ask_p - mid) / (mid + 1e-9) * 10000
        rel_bid_p = (bid_p - mid) / (mid + 1e-9) * 10000
        price_input = torch.cat([rel_ask_p, rel_bid_p], dim=-1)  # (B, T, 20)
        price_feat = self.price_stream(price_input)  # (B, T, d//2)
        
        # 数量流: 对数化
        log_ask_v = torch.log1p(ask_v)
        log_bid_v = torch.log1p(bid_v)
        volume_input = torch.cat([log_ask_v, log_bid_v], dim=-1)  # (B, T, 20)
        volume_feat = self.volume_stream(volume_input)  # (B, T, d//2)
        
        # ================================================================
        # Step 3: Level Transformer (可选)
        # ================================================================
        if self.use_level_transformer:
            level_feat = self.level_transformer(rel_ask_p,rel_bid_p,log_ask_v,log_bid_v)  # (B, T, d//2)
            combined = torch.cat([context, price_feat, volume_feat, level_feat], dim=-1)
        else:
            combined = torch.cat([context, price_feat, volume_feat], dim=-1)
        
        # ================================================================
        # Step 4: 特征融合
        # ================================================================
        fused = self.feature_fusion(combined)  # (B, T, d_model)
        
        # # ================================================================
        # # Step 5: 时间降采样
        # # ================================================================
        # x = fused.permute(0, 2, 1)  # (B, d_model, T)
        # for block in self.time_blocks:
        #     x = block(x)
        # x = x.permute(0, 2, 1)  # (B, T_out, d_model)
        
        return self.output_norm(fused)


class StructuredLOBEncoderLite(BaseEncoder):
    """
    轻量版结构化LOB编码器
    
    简化版本，不使用Level Transformer，计算效率更高
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        num_levels: int = 10,
        d_model: int = 64,
        time_strides: List[int] = [5, 2, 2],
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self._downsample_ratio = 1
        for s in time_strides:
            self._downsample_ratio *= s
        
        # 手工特征
        self.context = LOBContextSummarizer(
            in_channels=in_channels,
            num_levels=num_levels,
            context_dim=d_model // 2,
            use_learnable=False,
        )
        
        # 简单特征提取
        self.feature_proj = nn.Sequential(
            nn.Linear(in_channels * num_levels, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        
        # 融合
        self.fusion = nn.Sequential(
            nn.Linear(d_model // 2 + d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 时间降采样
        self.time_blocks = nn.ModuleList()
        for stride in time_strides:
            self.time_blocks.append(CausalTimeBlock(d_model, d_model, stride, dropout))
        
        self.output_norm = nn.LayerNorm(d_model)
        self._output_dim = d_model
        
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    @property
    def downsample_ratio(self) -> int:
        return self._downsample_ratio
        
    def forward(self, lob: torch.Tensor) -> torch.Tensor:
        B, C, T, L = lob.shape
        
        # 上下文
        ctx = self.context(lob)  # (B, T, d//2)
        
        # 特征
        lob_flat = lob.permute(0, 2, 1, 3).reshape(B, T, -1)  # (B, T, 40)
        feat = self.feature_proj(lob_flat)  # (B, T, d)
        
        # 融合
        combined = torch.cat([ctx, feat], dim=-1)
        fused = self.fusion(combined)
        
        # 时间降采样
        x = fused.permute(0, 2, 1)
        for block in self.time_blocks:
            x = block(x)
        x = x.permute(0, 2, 1)
        
        return self.output_norm(x)
