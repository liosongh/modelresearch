# =============================================================================
# Event-Driven Fusion - 事件驱动融合模块
# Version: v1.0
#
# 核心设计思想：
# 1. 维护一个"市场状态"，在Trade事件时更新
# 2. 无交易时状态衰减，有交易时状态更新
# 3. 状态影响LOB和Trade的特征解读
# 4. 更符合真实市场的动态
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict


class MarketStateManager(nn.Module):
    """
    市场状态管理器
    
    维护一个累积的市场状态，记录：
    - 近期的订单流方向
    - 流动性消耗情况
    - 信息冲击的影响
    
    这个状态在Trade发生时更新，无交易时衰减
    """
    
    def __init__(
        self,
        d_model: int = 64,
        state_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.state_dim = state_dim
        
        # 状态初始化 (可学习)
        self.init_state = nn.Parameter(torch.zeros(1, state_dim))
        
        # GRU风格的状态更新
        self.state_gru = nn.GRUCell(d_model, state_dim)
        
        # 更新门: 决定多少新信息融入状态
        self.update_gate = nn.Sequential(
            nn.Linear(d_model + state_dim, state_dim),
            nn.Sigmoid()
        )
        
        # 重置门: 决定是否重置状态
        self.reset_gate = nn.Sequential(
            nn.Linear(d_model + state_dim, state_dim),
            nn.Sigmoid()
        )
        
        # 衰减率 (无事件时)
        self.decay_rate = nn.Parameter(torch.tensor(0.95))
        
    def forward(
        self,
        trade_feat: torch.Tensor,  # (B, T, D)
        has_trade: torch.Tensor,   # (B, T)
    ) -> torch.Tensor:
        """
        管理市场状态
        
        Returns:
            state_seq: (B, T, state_dim) 状态序列
        """
        B, T, D = trade_feat.shape
        device = trade_feat.device
        
        # 初始化状态
        state = self.init_state.expand(B, -1).clone()
        
        states = []
        
        for t in range(T):
            trade_t = trade_feat[:, t, :]  # (B, D)
            event_t = has_trade[:, t:t+1]  # (B, 1)
            
            # 计算门控
            combined = torch.cat([trade_t, state], dim=-1)
            update_g = self.update_gate(combined)
            reset_g = self.reset_gate(combined)
            
            # GRU更新
            new_state = self.state_gru(trade_t, state * reset_g)
            
            # 条件更新
            decay = torch.sigmoid(self.decay_rate)
            state = torch.where(
                event_t.expand(-1, self.state_dim) > 0.5,
                update_g * new_state + (1 - update_g) * state,  # 有交易
                decay * state  # 无交易: 衰减
            )
            
            states.append(state)
        
        return torch.stack(states, dim=1)  # (B, T, state_dim)


class StateInfluenceModule(nn.Module):
    """
    状态影响模块
    
    将市场状态信息注入到LOB和Trade特征中
    """
    
    def __init__(self, d_model: int, state_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # 状态到LOB的映射
        self.state_to_lob = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Tanh(),
        )
        
        # 状态到Trade的映射
        self.state_to_trade = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Tanh(),
        )
        
        # LOB特征融合
        self.lob_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Trade特征融合
        self.trade_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
    def forward(
        self,
        lob_feat: torch.Tensor,   # (B, T, D)
        trade_feat: torch.Tensor, # (B, T, D)
        state: torch.Tensor,      # (B, T, state_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        状态影响特征
        """
        # 状态映射
        state_lob = self.state_to_lob(state)
        state_trade = self.state_to_trade(state)
        
        # 融合
        lob_enhanced = self.lob_fusion(torch.cat([lob_feat, state_lob], dim=-1))
        trade_enhanced = self.trade_fusion(torch.cat([trade_feat, state_trade], dim=-1))
        
        return lob_enhanced, trade_enhanced


class EventDrivenFusion(nn.Module):
    """
    事件驱动融合
    
    核心思想: 
    - 大多数时间(无交易): LOB主导，Trade信息衰减
    - 交易发生时: Trade信息注入，更新市场状态
    - 交易后: LOB反映新状态，信息逐渐被消化
    
    这更符合真实市场动态
    """
    
    def __init__(
        self,
        d_model: int = 64,
        state_dim: int = 32,
        nhead: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.state_dim = state_dim
        
        # ================================================================
        # Part 1: 市场状态管理
        # ================================================================
        self.state_manager = MarketStateManager(d_model, state_dim, dropout)
        
        # ================================================================
        # Part 2: 状态影响
        # ================================================================
        self.state_influence = StateInfluenceModule(d_model, state_dim, dropout)
        
        # ================================================================
        # Part 3: 跨模态注意力
        # ================================================================
        self.cross_attn_lob = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn_trade = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        self.lob_norm = nn.LayerNorm(d_model)
        self.trade_norm = nn.LayerNorm(d_model)
        
        # ================================================================
        # Part 4: 输出融合
        # ================================================================
        self.output_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.output_norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        lob_feat: torch.Tensor,    # (B, T, D)
        trade_feat: torch.Tensor,  # (B, T, D)
        has_trade: torch.Tensor,   # (B, T)
    ) -> Dict[str, torch.Tensor]:
        """
        事件驱动融合
        
        Returns:
            dict with:
            - 'fused': (B, T, D) 融合后的特征
            - 'lob_enhanced': (B, T, D) 增强后的LOB
            - 'trade_enhanced': (B, T, D) 增强后的Trade
            - 'market_state': (B, T, state_dim) 市场状态
        """
        # ============================================================
        # Step 1: 市场状态管理
        # ============================================================
        market_state = self.state_manager(trade_feat, has_trade)
        
        # ============================================================
        # Step 2: 状态影响特征
        # ============================================================
        lob_enhanced, trade_enhanced = self.state_influence(
            lob_feat, trade_feat, market_state
        )
        
        # ============================================================
        # Step 3: 跨模态注意力
        # ============================================================
        # LOB查询Trade
        lob_cross, _ = self.cross_attn_lob(lob_enhanced, trade_enhanced, trade_enhanced)
        lob_out = self.lob_norm(lob_enhanced + lob_cross)
        
        # Trade查询LOB
        trade_cross, _ = self.cross_attn_trade(trade_enhanced, lob_out, lob_out)
        trade_out = self.trade_norm(trade_enhanced + trade_cross)
        
        # ============================================================
        # Step 4: 输出融合
        # ============================================================
        fused = self.output_fusion(torch.cat([lob_out, trade_out], dim=-1))
        
        return {
            'fused': self.output_norm(fused),
            'lob_enhanced': lob_out,
            'trade_enhanced': trade_out,
            'market_state': market_state,
        }


class EventDrivenFusionFast(nn.Module):
    """
    快速版事件驱动融合
    
    使用并行计算近似状态传播，速度更快
    """
    
    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # 事件影响系数
        self.event_impact = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Tanh(),
        )
        
        # 衰减系数
        self.decay = nn.Parameter(torch.ones(d_model) * 0.9)
        
        # 跨模态注意力
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d_model)
        
        # 融合
        self.lob_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.trade_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.output_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        
    def forward(
        self,
        lob_feat: torch.Tensor,
        trade_feat: torch.Tensor,
        has_trade: torch.Tensor,
    ) -> torch.Tensor:
        """
        快速融合
        """
        B, T, D = lob_feat.shape
        
        # 事件影响
        trade_impact = self.event_impact(trade_feat) * has_trade.unsqueeze(-1)
        
        # 累积状态 (简化版)
        decay = torch.sigmoid(self.decay).unsqueeze(0).unsqueeze(0)
        
        state = torch.zeros_like(trade_impact[:, 0, :])
        states = []
        for t in range(T):
            state = decay.squeeze(0).squeeze(0) * state + trade_impact[:, t, :]
            states.append(state)
        
        state_seq = torch.stack(states, dim=1)
        
        # 融合
        lob_enhanced = self.lob_fusion(torch.cat([lob_feat, state_seq], dim=-1))
        trade_enhanced = self.trade_fusion(torch.cat([trade_feat, state_seq], dim=-1))
        
        # 跨模态
        fused, _ = self.cross_attn(lob_enhanced, trade_enhanced, trade_enhanced)
        fused = self.cross_norm(lob_enhanced + fused)
        
        # 输出
        output = self.output_fusion(torch.cat([fused, trade_enhanced], dim=-1))
        
        return output


class IntegratedFusion(nn.Module):
    """
    集成融合模块
    
    结合分层融合和事件驱动融合的优点
    """
    
    def __init__(
        self,
        d_model: int = 64,
        state_dim: int = 32,
        nhead: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # 事件驱动状态
        self.state_manager = MarketStateManager(d_model, state_dim, dropout)
        
        # 状态影响
        self.state_influence = StateInfluenceModule(d_model, state_dim, dropout)
        
        # 双向跨模态注意力
        self.lob_cross = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.trade_cross = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.lob_norm = nn.LayerNorm(d_model)
        self.trade_norm = nn.LayerNorm(d_model)
        
        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),  # lob + trade + state
            nn.Sigmoid()
        )
        
        # 输出
        self.output = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
    def forward(
        self,
        lob_feat: torch.Tensor,
        trade_feat: torch.Tensor,
        has_trade: torch.Tensor,
    ) -> torch.Tensor:
        """
        集成融合
        """
        # 状态管理
        state = self.state_manager(trade_feat, has_trade)
        
        # 状态影响
        lob_enhanced, trade_enhanced = self.state_influence(lob_feat, trade_feat, state)
        
        # 跨模态
        lob_cross, _ = self.lob_cross(lob_enhanced, trade_enhanced, trade_enhanced)
        lob_out = self.lob_norm(lob_enhanced + lob_cross)
        
        trade_cross, _ = self.trade_cross(trade_enhanced, lob_out, lob_out)
        trade_out = self.trade_norm(trade_enhanced + trade_cross)
        
        # 状态投影
        state_proj = nn.Linear(state.size(-1), self.lob_norm.normalized_shape[0], device=state.device)(state)
        
        # 门控融合
        gate = self.gate(torch.cat([lob_out, trade_out, state_proj], dim=-1))
        combined = gate * lob_out + (1 - gate) * trade_out
        
        # 输出
        output = self.output(torch.cat([combined, state_proj], dim=-1))
        
        return output
