# =============================================================================
# Cross-Modal Feature Builder - 跨模态特征构建器
# Version: v1.0
# 
# 领域知识驱动的跨模态特征构建，在编码器之前显式构建有金融意义的交叉特征
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class CrossModalFeatureBuilder(nn.Module):
    """
    领域知识驱动的跨模态特征构建器
    
    在编码器之前，显式构建LOB和Trade之间有金融意义的交叉特征。
    这些特征比让模型自己学习更高效、更可解释。
    
    构建的特征包括：
    1. vwap_book_interaction: VWAP偏离度 × 订单簿不平衡
    2. volume_impact: 成交量冲击比
    3. direction_consistency: 交易-订单簿一致性
    4. effective_trade_signal: 有效交易信号
    5. spread_sensitive_trade: 价差敏感成交
    6. depth_consumption: 深度消耗速度
    
    Args:
        num_levels: LOB档位数
        eps: 数值稳定性参数
    """
    
    def __init__(self, num_levels: int = 10, eps: float = 1e-9):
        super().__init__()
        self.num_levels = num_levels
        self.eps = eps
        
        # Level权重 (近档权重更高)
        weights = torch.softmax(torch.arange(num_levels, 0, -1, dtype=torch.float32), dim=0)
        self.register_buffer('level_weights', weights)
        
    def forward(
        self, 
        lob: torch.Tensor,      # (B, 4, T, L) - [ask_p, bid_p, ask_notional, bid_notional]
        trade: torch.Tensor,    # (B, F, T) - F=12
        trade_feature_indices: Optional[Dict[str, int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        构建跨模态特征
        
        Args:
            lob: LOB数据 (B, 4, T, L)
                 通道顺序: [ask_price, bid_price, ask_notional, bid_notional]
            trade: Trade数据 (B, F, T)
                 默认特征顺序: [has_trade, log_time_since_last, rel_vwap, ...]
            trade_feature_indices: Trade特征索引映射 (可选)
            
        Returns:
            dict包含:
            - 'lob': 原始LOB数据
            - 'lob_enhanced': 增强后的LOB特征
            - 'trade': 原始Trade数据
            - 'cross_features': 显式跨模态特征 (B, 6, T)
            - 'mid_price': 中间价 (B, T)
            - 'book_imbalance': 订单簿不平衡 (B, T)
        """
        B, C, T, L = lob.shape
        
        # 默认Trade特征索引
        if trade_feature_indices is None:
            trade_feature_indices = {
                'has_trade': 0,
                'log_time_since_last': 1,
                'rel_vwap': 2,
                'rel_price_last': 3,
                'log_total_volume': 4,
                'log_total_notional': 5,
                'log_trade_count': 6,
                'log_avg_trade_size': 7,
                'trade_imbalance': 8,
                'notional_imbalance': 9,
                'trade_count_ratio': 10,
                'rel_price_std': 11,
            }
        
        # ============================================================
        # 1. 从LOB提取关键信息
        # ============================================================
        lob_features = self._extract_lob_features(lob)
        
        # ============================================================
        # 2. 从Trade提取关键信息
        # ============================================================
        trade_features = self._extract_trade_features(trade, trade_feature_indices)
        
        # ============================================================
        # 3. 构建跨模态特征
        # ============================================================
        cross_features = self._build_cross_features(lob_features, trade_features)
        
        # ============================================================
        # 4. 构建增强的LOB特征
        # ============================================================
        lob_enhanced = self._enhance_lob(lob, lob_features)
        
        return {
            'lob': lob,
            'lob_enhanced': lob_enhanced,
            'trade': trade,
            'cross_features': cross_features,
            'mid_price': lob_features['mid_price'],
            'book_imbalance': lob_features['book_imbalance'],
        }
    
    def _extract_lob_features(self, lob: torch.Tensor) -> Dict[str, torch.Tensor]:
        """从LOB数据中提取关键特征"""
        B, C, T, L = lob.shape
        
        # 分离各通道
        ask_prices = lob[:, 0, :, :]      # (B, T, L)
        bid_prices = lob[:, 1, :, :]      # (B, T, L)
        ask_notional = lob[:, 2, :, :]    # (B, T, L)
        bid_notional = lob[:, 3, :, :]    # (B, T, L)
        
        # Mid Price & Spread
        mid_price = (ask_prices[:, :, 0] + bid_prices[:, :, 0]) / 2  # (B, T)
        spread = ask_prices[:, :, 0] - bid_prices[:, :, 0]          # (B, T)
        relative_spread = spread / (mid_price + self.eps)            # (B, T)
        
        # 订单簿不平衡 (多档加权)
        # 使用预计算的level权重
        weights = self.level_weights.view(1, 1, -1)  # (1, 1, L)
        weighted_ask_depth = (ask_notional * weights).sum(dim=-1)  # (B, T)
        weighted_bid_depth = (bid_notional * weights).sum(dim=-1)  # (B, T)
        book_imbalance = (weighted_bid_depth - weighted_ask_depth) / (
            weighted_bid_depth + weighted_ask_depth + self.eps
        )
        
        # 深度压力 (近档 vs 远档)
        near_depth = ask_notional[:, :, :3].sum(dim=-1) + bid_notional[:, :, :3].sum(dim=-1)
        far_depth = ask_notional[:, :, 3:].sum(dim=-1) + bid_notional[:, :, 3:].sum(dim=-1)
        depth_pressure = near_depth / (far_depth + self.eps)
        
        # 近档深度 (用于计算冲击)
        near_ask = ask_notional[:, :, :3].sum(dim=-1)  # (B, T)
        near_bid = bid_notional[:, :, :3].sum(dim=-1)  # (B, T)
        total_depth = weighted_ask_depth + weighted_bid_depth
        
        return {
            'mid_price': mid_price,
            'spread': spread,
            'relative_spread': relative_spread,
            'book_imbalance': book_imbalance,
            'depth_pressure': depth_pressure,
            'near_ask': near_ask,
            'near_bid': near_bid,
            'near_depth': near_depth,
            'total_depth': total_depth,
            'weighted_ask_depth': weighted_ask_depth,
            'weighted_bid_depth': weighted_bid_depth,
        }
    
    def _extract_trade_features(
        self, 
        trade: torch.Tensor, 
        indices: Dict[str, int]
    ) -> Dict[str, torch.Tensor]:
        """从Trade数据中提取关键特征"""
        return {
            'has_trade': trade[:, indices['has_trade'], :],           # (B, T)
            'rel_vwap': trade[:, indices['rel_vwap'], :],             # (B, T)
            'log_notional': trade[:, indices['log_total_notional'], :],   # (B, T)
            'trade_imbalance': trade[:, indices['trade_imbalance'], :], # (B, T)
        }
    
    def _build_cross_features(
        self, 
        lob_feat: Dict[str, torch.Tensor],
        trade_feat: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """构建跨模态特征"""
        
        # 特征1: VWAP偏离度 × 订单簿不平衡
        # 当成交价偏向某一侧且订单簿也不平衡时，趋势更可能延续
        vwap_book_interaction = trade_feat['rel_vwap'] * lob_feat['book_imbalance']
        
        # 特征2: 成交量冲击比
        # 成交量相对于盘口深度的比例 → 衡量市场冲击
        volume = torch.exp(trade_feat['log_notional'])  # 还原volume
        volume_impact = torch.log1p(volume / (lob_feat['total_depth'] + self.eps))
        
        # 特征3: 交易-订单簿一致性
        # 当Trade方向与LOB不平衡方向一致时，信号更强
        direction_consistency = trade_feat['trade_imbalance'] * lob_feat['book_imbalance']
        
        # 特征4: 有效交易信号
        # 只有当有交易发生时，Trade特征才有意义
        effective_trade_signal = trade_feat['has_trade'] * trade_feat['trade_imbalance']
        
        # 特征5: 价差敏感成交
        # 在spread较大时的成交可能更有信息含量
        spread_sensitive_trade = (
            lob_feat['relative_spread'] * 
            trade_feat['has_trade'] * 
            torch.abs(trade_feat['trade_imbalance'])
        )
        
        # 特征6: 深度消耗速度
        # 成交量与近档深度的比例
        buy_signal = torch.clamp(trade_feat['trade_imbalance'], min=0)
        sell_signal = torch.clamp(-trade_feat['trade_imbalance'], min=0)
        ask_consumption = buy_signal * volume / (lob_feat['near_ask'] + self.eps)
        bid_consumption = sell_signal * volume / (lob_feat['near_bid'] + self.eps)
        depth_consumption = torch.log1p(ask_consumption + bid_consumption)
        
        # Stack成特征张量
        cross_features = torch.stack([
            vwap_book_interaction,
            volume_impact,
            direction_consistency,
            effective_trade_signal,
            spread_sensitive_trade,
            depth_consumption,
        ], dim=1)  # (B, 6, T)
        
        return cross_features
    
    def _enhance_lob(
        self, 
        lob: torch.Tensor, 
        lob_feat: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """为LOB添加汇总特征作为额外通道"""
        B, C, T, L = lob.shape
        
        # 构建额外通道 (广播到每个Level)
        extra_channels = torch.stack([
            lob_feat['book_imbalance'],
            lob_feat['relative_spread'],
            lob_feat['depth_pressure'],
            torch.log1p(lob_feat['near_depth']),
        ], dim=1)  # (B, 4, T)
        
        # 扩展到Level维度
        extra_channels = extra_channels.unsqueeze(-1).expand(-1, -1, -1, L)  # (B, 4, T, L)
        
        # 拼接
        lob_enhanced = torch.cat([lob, extra_channels], dim=1)  # (B, 8, T, L)
        
        return lob_enhanced
    
    @property
    def num_cross_features(self) -> int:
        """跨模态特征数量"""
        return 6
    
    @property
    def num_lob_extra_channels(self) -> int:
        """LOB额外通道数"""
        return 4
