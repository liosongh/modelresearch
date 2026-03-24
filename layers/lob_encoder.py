# =============================================================================
# LOB Encoder - 限价订单簿特征编码器
# Version: v3.0 - 多时间尺度因果编码（100ms -> 500ms -> 1000ms）
# =============================================================================

from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_encoder import BaseEncoder

class ChannelLayerNorm2d(nn.Module):
    """
    对每个 (t, l) 位置仅在通道维做 LayerNorm，避免跨时间统计泄漏。
    """

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)  # (B, T, L, C)
        x = F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2).contiguous()


class CausalConv2d(nn.Module):
    """
    因果 2D 卷积。

    仅在 Time 维度做左侧 padding，保证不泄露未来信息。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple = (3, 1),
        stride: tuple = (1, 1),
        dilation: tuple = (1, 1),
        bias: bool = True,
    ):
        super().__init__()
        self.time_pad = (kernel_size[0] - 1) * dilation[0]
        self.level_pad = ((kernel_size[1] - 1) * dilation[1]) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=(0, 0),
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.pad for 2D: (L_left, L_right, T_top, T_bottom)
        x = F.pad(x, (self.level_pad, self.level_pad, self.time_pad, 0))
        return self.conv(x)


class CausalDownsamplingBlock(nn.Module):
    """
    兼容旧接口的因果降采样块（保留导出符号）。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_stride: int = 2,
        level_stride: int = 2,
        time_kernel: int = 3,
        level_kernel: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            CausalConv2d(
                in_channels,
                out_channels,
                kernel_size=(time_kernel, level_kernel),
                stride=(time_stride, max(1, level_stride)),
                bias=False,
            ),
            ChannelLayerNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TimeAwareLevelAttention(nn.Module):
    """
    Time-aware Level Self-Attention：
    对每个时间步在 Level 维做自注意力，学习档位间关系。
    """

    def __init__(self, channels: int, num_heads: int = 4, attn_dropout: float = 0.0):
        super().__init__()
        self.channels = channels
        # 正确定义头数，确保属性存在
        self.num_heads = max(num_heads, 1)
        self.head_dim = channels // self.num_heads
        
        # 确保通道数可以被头数整除
        assert self.head_dim * self.num_heads == channels, \
            f"Channels {channels} must be divisible by num_heads {self.num_heads}"
        self.pre_norm = ChannelLayerNorm2d(channels)
        # 手动定义QKV投影层
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(attn_dropout)
        # self.attn = nn.MultiheadAttention(
        #     embed_dim=channels,
        #     num_heads=num_heads,
        #     dropout=attn_dropout,
        #     batch_first=True,
        #     # enable_flash_attention=False ## 关闭flash attention

        # )
        self.out_norm = ChannelLayerNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, L) -> (B*T, L, C)
        residual = x
        x = self.pre_norm(x)
        b, c, t, l =  x.shape
        batch_size = b * t
        x_seq = x.permute(0, 2, 3, 1).reshape(batch_size, l, c)
        # 1. 计算 QKV
        qkv = self.qkv(x_seq).chunk(3, dim=-1)
        q = qkv[0].view(batch_size, l, self.num_heads, self.head_dim).transpose(1, 2)
        k = qkv[1].view(batch_size, l, self.num_heads, self.head_dim).transpose(1, 2)
        v = qkv[2].view(batch_size, l, self.num_heads, self.head_dim).transpose(1, 2)
        # 2. 手动缩放点积注意力（无SDPA，纯矩阵运算，无CUDA内核限制）
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        # 3. 输出拼接
        out = (attn @ v).transpose(1, 2).reshape(batch_size, l, c)
        out = self.proj(out)

        # 恢复维度
        attn_out = out.reshape(b, t, l, c).permute(0, 3, 1, 2).contiguous()
        x = residual + attn_out
        return self.out_norm(x)


class ResBlock2D(nn.Module):
    """
    单尺度残差块：LevelAttention + CausalTemporalConv + FFN。
    提取时空的局部特征块
    level_attention: 时间 aware 的 level 自注意力，对每个时间步在 Level 维做自注意力，学习档位间关系。
    temporal: 时间序列的局部特征提取，对时间维度做卷积，学习时间维度上的关系。
    level_mix: 档位间的局部特征提取，对每个时间步在 Level 维做卷积，学习档位间局部特征。
    ffn: 前馈神经网络，对每个时间步在 Level 维做前馈神经网络，学习档位间全局特征。
    out_act: 输出激活函数，对每个时间步在 Level 维做输出激活函数，学习档位间关系。
    """

    def __init__(
        self,
        channels: int,
        time_kernel: int = 3,
        level_kernel: int = 3,
        attn_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        # self.level_attention = TimeAwareLevelAttention(
        #     channels, num_heads=attn_heads, attn_dropout=dropout
        # )
        ## 时间序列的局部特征提取，对时间维度做卷积，学习时间维度上的关系。 (B, C, T, L) -> (B, C, T, L)
        self.temporal = nn.Sequential(
            ChannelLayerNorm2d(channels),
            CausalConv2d(
            channels,
            channels,
            kernel_size=(time_kernel, 1),
            stride=(1, 1),
            bias=False,
        ),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
 
        ## 档位间的局部特征提取，对每个时间步在 Level 维做卷积，学习档位间局部特征。
        self.level_mix = nn.Sequential(
            ChannelLayerNorm2d(channels),
            nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, level_kernel),
            padding=(0, level_kernel // 2), ## 保持level维度长度不变
            bias=False,
            ),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        # ## 
        # self.ffn = nn.Sequential(
        #     ChannelLayerNorm2d(channels),
        #     nn.GELU(),
        #     nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        #     nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        # )
        # self.out_act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        # x = self.level_attention(x)
        ## 时间序列的局部特征提取，对时间维度做卷积，学习时间维度上的关系。
        x = self.temporal(x) ## (B, C, T, L) -> (B, C, T, L)
        # ## 残差连接
        # x = x + residual
        # residual = x
        ## 档位间的局部特征提取，对每个时间步在 Level 维做卷积，学习档位间局部特征。
        x = self.level_mix(x) ## (B, C, T, L) -> (B, C, T, L)
        ## 残差连接
        x = x + residual
        return x


class ResStage(nn.Module):
    """同一时间尺度上的多层残差堆叠。"""

    def __init__(
        self,
        channels: int,
        depth: int,
        time_kernel: int = 3,
        level_kernel: int = 3,
        attn_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ResBlock2D(
                    channels=channels,
                    time_kernel=time_kernel,
                    level_kernel=level_kernel,
                    attn_heads=attn_heads,
                    dropout=dropout,
                )
                for _ in range(max(1, depth))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class ScaleTransition(nn.Module):
    """
    跨尺度过渡：因果时间聚合+降采样，并可选 level 维降采样。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_kernel: int,
        time_stride: int,
        level_kernel: int = 3,
        level_stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.temporal_down = nn.Sequential(
            ChannelLayerNorm2d(in_channels),
            CausalConv2d(
            in_channels,
            out_channels,
            kernel_size=(time_stride, 1),
            stride=(time_stride, 1),
            bias=False,
        ),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        # self.level_down = nn.Conv2d(
        #     out_channels,
        #     out_channels,
        #     kernel_size=(1, level_kernel),
        #     stride=(1, max(1, level_stride)),
        #     padding=(0, level_kernel // 2),
        #     bias=False,
        # )
        # self.norm = ChannelLayerNorm2d(out_channels)
        # self.act = nn.GELU()
        # self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_down(x)
        return x


class LevelAggregator(nn.Module):
    """用注意力聚合 Level 维度，输出 (B, T, d_model)。"""

    def __init__(self, in_channels: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.scale = d_model ** -0.5
        self.query = nn.Linear(in_channels, d_model)
        self.key = nn.Linear(in_channels, d_model)
        self.value = nn.Linear(in_channels, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, L) -> (B, T, L, C)
        x = x.permute(0, 2, 3, 1)
        q = self.query(x.mean(dim=2, keepdim=True))  # (B, T, 1, d)
        k = self.key(x)  # (B, T, L, d)
        v = self.value(x)  # (B, T, L, d)

        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale, dim=-1)
        out = torch.matmul(attn, v).squeeze(2)  # (B, T, d)
        return self.out_proj(out)

# class LevelAggregator(nn.Module):
#     """
#     优化版：注意力聚合空间维度，输出 (B, T, d_model)
#     修复：静态Query、无残差、无归一化、无位置编码
#     """
#     def __init__(self, in_channels: int, d_model: int, max_spatial_len: int = 100):
#         super().__init__()
#         self.d_model = d_model
#         self.scale = d_model ** -0.5
        
#         # 1. 标准QKV投影（动态Query，修复静态问题）
#         self.q_proj = nn.Linear(in_channels, d_model)
#         self.k_proj = nn.Linear(in_channels, d_model)
#         self.v_proj = nn.Linear(in_channels, d_model)
        
#         # 2. 空间位置编码（可选但强烈推荐，保留空间结构）
#         self.spatial_pos_emb = nn.Parameter(torch.randn(1, 1, max_spatial_len, d_model))
        
#         # 3. 输出层 + 归一化 + 残差
#         self.out_proj = nn.Linear(d_model, d_model)
#         self.norm = nn.LayerNorm(d_model)  # 稳定训练
#         self.dropout = nn.Dropout(0.1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: (B, C, T, L) -> (B, T, L, C)
#         B, C, T, L = x.shape
#         x = x.permute(0, 2, 3, 1)  # (B, T, L, C)
        
#         # 动态QKV（修复静态Query问题）
#         q = self.q_proj(x)   # (B, T, L, d)  用全特征生成Q
#         k = self.k_proj(x)   # (B, T, L, d)
#         v = self.v_proj(x)   # (B, T, L, d)
        
#         # 加入空间位置编码
#         q = q + self.spatial_pos_emb[:, :, :L]
#         k = k + self.spatial_pos_emb[:, :, :L]
        
#         # 注意力：聚合空间维度 L → 1
#         # 对Q做全局平均，得到时间维度的查询 (B, T, 1, d)
#         q_global = q.mean(dim=2, keepdim=True)  # 保留你的核心设计，但Q是动态的
#         attn = torch.softmax(torch.matmul(q_global, k.transpose(-1, -2)) * self.scale, dim=-1)
#         attn = self.dropout(attn)
        
#         # 聚合输出
#         out = torch.matmul(attn, v).squeeze(2)  # (B, T, d)
#         out = self.out_proj(out)
        
#         # 残差连接 + 归一化（核心优化）
#         # 残差：输入x在时间维度的均值，匹配维度
#         residual = x.mean(dim=2)  # (B, T, C) → 若C=d_model可直接用
#         if C != self.d_model:
#             residual = nn.Linear(C, self.d_model).to(x.device)(residual)
#         out = self.norm(out + residual)
        
#         return out


class LOBEncoder(BaseEncoder):
    """
    多尺度 LOB 编码器：
    - Stage100ms: LevelAttention + ResStage
    - Stage500ms: Causal(k=stride_1, s=stride_1) + LevelAttention + ResStage
    - Stage1000ms: Causal(k=stride_2, s=stride_2) + LevelAttention + ResStage
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 32,
        pre_group: int = 2,
        post_group: int = 8,
        time_strides: Optional[List[int]] = None,
        time_kernels: Optional[List[int]] = None,
        level_strides: Optional[List[int]] = None,
        level_kernels: Optional[List[int]] = None,
        kernel_sizes: Optional[dict] = None,
        dropout: float = 0.1,
        stage_depths: Optional[List[int]] = None,
        stage_channels: Optional[List[int]] = None,
        attn_heads: int = 4,
        return_multi_scale: bool = False,
        scale_output_dim: Optional[int] = None,
    ):
        super().__init__()

        time_strides = time_strides or [5, 2]
        self._time_strides = [max(1, int(s)) for s in time_strides]
        if len(self._time_strides) == 0:
            raise ValueError("time_strides 不能为空。")
        num_transitions = len(self._time_strides)
        num_stages = num_transitions + 1

        level_strides = level_strides or [1] * num_transitions
        level_strides = [max(1, int(s)) for s in level_strides]
        if len(level_strides) < num_transitions:
            level_strides = level_strides + [level_strides[-1]] * (num_transitions - len(level_strides))
        self._level_strides = level_strides[:num_transitions]

        if isinstance(time_kernels, Sequence) and not isinstance(time_kernels, (str, bytes)):
            self._time_kernels = [int(k) for k in time_kernels]
            if len(self._time_kernels) < num_transitions:
                pad_value = self._time_kernels[-1] if self._time_kernels else 3
                self._time_kernels = self._time_kernels + [pad_value] * (num_transitions - len(self._time_kernels))
            self._time_kernels = self._time_kernels[:num_transitions]
        elif isinstance(time_kernels, int):
            self._time_kernels = [int(time_kernels)] * num_transitions
        else:
            # 默认与时间步长一致：例如 5 -> 500ms，2 -> 1000ms
            self._time_kernels = self._time_strides.copy()

        if isinstance(level_kernels, Sequence) and not isinstance(level_kernels, (str, bytes)):
            self._level_kernels = [int(k) for k in level_kernels]
            if len(self._level_kernels) < num_transitions:
                pad_value = self._level_kernels[-1] if self._level_kernels else 3
                self._level_kernels = self._level_kernels + [pad_value] * (num_transitions - len(self._level_kernels))
            self._level_kernels = self._level_kernels[:num_transitions]
        elif isinstance(level_kernels, int):
            self._level_kernels = [int(level_kernels)] * num_transitions
        else:
            self._level_kernels = [1] * num_transitions

 
        if stage_channels is None:
            stage_channels = [base_channels] + [base_channels * 2] * num_transitions
        stage_channels = [int(c) for c in stage_channels]
        if len(stage_channels) < num_stages:
            raise ValueError(f"stage_channels 长度至少为 {num_stages}。")
        stage_channels = stage_channels[:num_stages]

        if stage_depths is None:
            stage_depths = [1] * num_stages
        stage_depths = [int(d) for d in stage_depths]
        if len(stage_depths) < num_stages:
            raise ValueError(f"stage_depths 长度至少为 {num_stages}。")
        stage_depths = stage_depths[:num_stages]

        self.return_multi_scale = return_multi_scale
        self.scale_output_dim = int(scale_output_dim or base_channels)
        self._downsample_ratio = 1
        for stride in self._time_strides:
            self._downsample_ratio *= stride
        self._output_dim = self.scale_output_dim


        ## 映射通道数
        # self.input_norm = nn.GroupNorm(2,in_channels) ## 如果之前已经做了归一化的处理，就可以先不归一化
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stage_channels[0], kernel_size=1, bias=False),
            ChannelLayerNorm2d(stage_channels[0]),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        ) ## 第一次要做postnorm处理，后面不需要
        ## 多尺度特征提取
        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()
        # self.scale_aggregators = nn.ModuleList()

        for idx in range(num_stages):
            self.stages.append(
                ResStage(
                    channels=stage_channels[idx],
                    depth=stage_depths[idx],
                    time_kernel=kernel_sizes.get("time"), ## 在同一时间尺度下的特征提取
                    level_kernel=kernel_sizes.get("level"), ## 在同一时间尺度下的特征提取
                    attn_heads=attn_heads, ## 目前不需要这个参数
                    dropout=dropout,
                )
            )
            if idx == num_stages-1:
                self.scale_aggregators = nn.Sequential(
                    ChannelLayerNorm2d(stage_channels[idx]),
                    LevelAggregator(in_channels=stage_channels[idx], d_model=self.scale_output_dim),
                    nn.GELU(),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                )
            if idx < num_transitions:
                self.transitions.append(
                    ScaleTransition(
                        in_channels=stage_channels[idx],
                        out_channels=stage_channels[idx + 1],
                        time_kernel=self._time_kernels[idx],
                        time_stride=self._time_strides[idx],
                        level_kernel=self._level_kernels[idx],
                        level_stride=self._level_strides[idx],
                        dropout=dropout,
                    )
                )

        # 便于外部读取每个尺度，默认按 100ms 基准累计命名
        self.scale_names = []
        cumulative = 1
        self.scale_names.append(f"{100 * cumulative}ms")
        for stride in self._time_strides:
            cumulative *= stride
            self.scale_names.append(f"{100 * cumulative}ms")

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def downsample_ratio(self) -> int:
        return self._downsample_ratio

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        # x: (B, C, T, L)
        # x = self.input_norm(x)
        ## 使用1x1卷积将输入通道数转换为stage_channels[0]
        x = self.stem(x) ## (B, C, T, L) -> (B, stage_channels[0], T, L)

        # scale_features = []
        ## stage1: 使用ResStage对输入特征进行处理
        x = self.stages[0](x) ## (B, stage_channels[0], T, L) -> (B, stage_channels[0], T, L)
        # scale_features.append(self.scale_aggregators[0](x))

        for idx, transition in enumerate(self.transitions):
            x = transition(x) ## (B, stage_channels[idx], T, L) -> (B, stage_channels[idx + 1], T, L)
            x = self.stages[idx + 1](x) ## (B, stage_channels[idx + 1], T, L) -> (B, stage_channels[idx + 1], T_ds, L)
            # scale_features.append(self.scale_aggregators[idx + 1](x)) ## (B, stage_channels[idx + 1], T_ds, L) -> (B, scale_output_dim, T_ds, L)
        final_feature = self.scale_aggregators(x)
        # final_feature = scale_features[-1]
        # if self.return_multi_scale:
        #     scale_dict = {name: feat for name, feat in zip(self.scale_names, scale_features)}
        #     return {
        #         "final": final_feature,
        #         "scale_dict": scale_dict,
        #         "scale_list": scale_features,
        #     }
        return final_feature
