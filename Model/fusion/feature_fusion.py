# =============================================================================
# Feature Fusion - 多模态特征融合模块
# Version: v2.0
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class FeatureFusion(nn.Module):
    """
    特征融合基类/工厂类。
    
    根据策略选择不同的融合方式。
    """
    
    def __init__(
        self, 
        input_dims: Dict[str, int],
        d_model: int,
        strategy: str = 'late_concat',
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        self.input_dims = input_dims
        self.d_model = d_model
        self.strategy = strategy
        self.modalities = list(input_dims.keys())
        
        # 根据策略选择融合层
        if strategy == 'late_concat':
            self.fusion = ConcatFusion(input_dims, d_model, use_layer_norm)
        elif strategy == 'gated':
            self.fusion = GatedFusion(input_dims, d_model, use_layer_norm)
        elif strategy == 'cross_attention':
            self.fusion = CrossAttentionFusion(input_dims, d_model, use_layer_norm)
        else:
            raise ValueError(f"Unknown fusion strategy: {strategy}")
            
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: 各模态特征字典 {'lob': (B, T, D1), 'trade': (B, T, D2), ...}
            
        Returns:
            fused: 融合后的特征 (B, T, d_model)
        """
        return self.fusion(features)


class ConcatFusion(nn.Module):
    """
    Late Concatenation Fusion (晚期拼接融合)。
    
    最简单直接的融合方式: 拼接所有模态特征后通过线性层映射。
    
    优点: 简单高效，不引入过多参数
    缺点: 模态间没有交互，仅依赖后续网络学习关系
    """
    
    def __init__(
        self, 
        input_dims: Dict[str, int],
        d_model: int,
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        total_dim = sum(input_dims.values())
        
        self.projector = nn.Linear(total_dim, d_model)
        # self.adapter_conv = nn.Conv1d(input_dims['lob'], d_model, 1)
        self.norm = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
        self.modality_order = sorted(input_dims.keys())  # 固定顺序
        
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 按固定顺序拼接，保证一致性
        concat_list = [features[k] for k in self.modality_order]
        
        # 验证时间维度一致
        T = concat_list[0].shape[1]
        for i, f in enumerate(concat_list):
            assert f.shape[1] == T, f"时间维度不一致: {self.modality_order[0]}={T}, {self.modality_order[i]}={f.shape[1]}"
        
        # Concatenate along feature dimension
        combined = torch.cat(concat_list, dim=-1)  # (B, T, sum_dims)
        # ## 为了和之前一样的model
        # combined = combined.permute(0, 2, 1)  # (B, sum_dims, T)
        # out = self.adapter_conv(combined)  # (B, T, d_model)
        # out = out.permute(0, 2, 1)  # (B, T, d_model)
        # Project and normalize
        out = self.projector(combined)  # (B, T, d_model)
        out = self.norm(out)
        
        return out


class GatedFusion(nn.Module):
    """
    Gated Fusion (门控融合)。
    
    使用可学习的门控机制动态调整各模态的贡献权重。
    
    优点: 模型可以学习哪个模态在当前时刻更重要
    缺点: 额外参数，可能过拟合
    """
    
    def __init__(
        self, 
        input_dims: Dict[str, int],
        d_model: int,
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        self.modalities = sorted(input_dims.keys())
        num_modalities = len(self.modalities)
        
        # 每个模态先投影到 d_model
        self.projectors = nn.ModuleDict({
            k: nn.Linear(v, d_model) for k, v in input_dims.items()
        })
        
        # 门控网络: 输入是拼接后的特征，输出是每个模态的权重
        total_dim = sum(input_dims.values())
        self.gate = nn.Sequential(
            nn.Linear(total_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_modalities),
            nn.Softmax(dim=-1)
        )
        
        self.norm = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
        
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 1. 投影各模态到相同维度
        projected = {k: self.projectors[k](features[k]) for k in self.modalities}
        
        # 2. 计算门控权重
        concat_raw = torch.cat([features[k] for k in self.modalities], dim=-1)
        weights = self.gate(concat_raw)  # (B, T, num_modalities)
        
        # 3. 加权融合
        stacked = torch.stack([projected[k] for k in self.modalities], dim=-1)  # (B, T, d_model, M)
        weights = weights.unsqueeze(2)  # (B, T, 1, M)
        
        out = (stacked * weights).sum(dim=-1)  # (B, T, d_model)
        out = self.norm(out)
        
        return out


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion (交叉注意力融合)。
    
    使用注意力机制让不同模态互相"看到"对方的信息。
    
    优点: 强大的模态间交互能力
    缺点: 计算量大，可能过拟合
    """
    
    def __init__(
        self, 
        input_dims: Dict[str, int],
        d_model: int,
        use_layer_norm: bool = True,
        nhead: int = 4
    ):
        super().__init__()
        
        self.modalities = sorted(input_dims.keys())
        
        # 投影到相同维度
        self.projectors = nn.ModuleDict({
            k: nn.Linear(v, d_model) for k, v in input_dims.items()
        })
        
        # 交叉注意力层 (每对模态之间)
        # 简化版: 使用一个共享的注意力层
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, batch_first=True
        )
        
        # 融合投影
        self.out_proj = nn.Linear(d_model * len(self.modalities), d_model)
        self.norm = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
        
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 1. 投影
        projected = {k: self.projectors[k](features[k]) for k in self.modalities}
        
        # 2. 交叉注意力
        # 简化实现: 将第一个模态作为 Query，其他作为 Key/Value
        # 完整实现应该是每对模态都做交叉注意力
        attended = []
        for query_key in self.modalities:
            query = projected[query_key]
            
            # 其他模态拼接作为 Key/Value
            others = [projected[k] for k in self.modalities if k != query_key]
            if others:
                kv = torch.cat(others, dim=1)  # (B, T*num_others, d_model)
            else:
                kv = query
                
            attn_out, _ = self.cross_attn(query, kv, kv)  # (B, T, d_model)
            attended.append(attn_out)
        
        # 3. 融合
        concat = torch.cat(attended, dim=-1)  # (B, T, d_model * M)
        out = self.out_proj(concat)  # (B, T, d_model)
        out = self.norm(out)
        
        return out
