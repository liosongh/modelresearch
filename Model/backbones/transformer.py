# =============================================================================
# Transformer Backbone - Transformer 主干网络
# Version: v2.0
# =============================================================================

import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    正弦余弦位置编码 (Sinusoidal Positional Encoding)。
    
    来自 "Attention Is All You Need" 论文。
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 预计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    可学习位置编码 (Learnable Positional Encoding)。
    
    对于金融时序数据，可学习的位置编码可能更有效，
    因为它可以学习到特定的时间模式（如开盘/收盘效应）。
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_len, d_model)
        
        # 初始化
        nn.init.normal_(self.pe.weight, mean=0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pe(positions)
        return self.dropout(x)


class TransformerBackbone(nn.Module):
    """
    Transformer Encoder 主干网络。
    
    用于对融合后的特征进行时序建模。
    
    Args:
        d_model: 模型维度
        nhead: 注意力头数
        num_layers: Encoder 层数
        dim_feedforward: FFN 隐藏层维度
        dropout: Dropout 比率
        activation: 激活函数
        positional_encoding: 位置编码类型 ('sinusoidal', 'learnable')
        max_seq_len: 最大序列长度
    """
    
    def __init__(
        self, 
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        #activation: str = 'gelu',
        positional_encoding: str = 'sinusoidal',
        max_seq_len: int = 500
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # 位置编码
        if positional_encoding == 'sinusoidal':
            self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        elif positional_encoding == 'learnable':
            self.pos_encoder = LearnablePositionalEncoding(d_model, max_seq_len, dropout)
        else:
            self.pos_encoder = nn.Identity()
            
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            # activation=activation,
            batch_first=True,  # 使用 (B, T, D) 格式
            # norm_first=True    # Pre-LN for better training stability
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            # enable_nested_tensor=False  # 兼容性
        )
        
        # # Final Layer Norm
        # self.final_norm = nn.LayerNorm(d_model)
        
    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """
        生成因果注意力掩码。
        
        确保每个位置只能看到之前的位置（包括自己）。
        """
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(
        self, 
        x: torch.Tensor, 
        causal: bool = True,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: 输入特征 (B, T, d_model)
            causal: 是否使用因果掩码
            src_key_padding_mask: Padding 掩码 (B, T)
            
        Returns:
            output: (B, T, d_model)
        """
        # 1. 位置编码
        x = self.pos_encoder(x)
        
        # 2. 因果掩码
        if causal:
            mask = self._generate_causal_mask(x.size(1), x.device)
        else:
            mask = None
            
        # 3. Transformer Encoding
        output = self.transformer_encoder(
            x, 
            mask=mask,
            # src_key_padding_mask=src_key_padding_mask
        )
        
        # 4. Final Norm
        # output = self.final_norm(output)
        
        return output
