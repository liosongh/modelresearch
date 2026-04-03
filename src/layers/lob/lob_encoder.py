# =============================================================================
# LOB Encoder - 限价订单簿特征编码器
# Version: v3.0 - 多时间尺度因果编码（100ms -> 500ms -> 1000ms）
# =============================================================================

from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
# from layers.Base_encoder import BaseEncoder

# class ChannelLayerNorm2d(nn.Module):
#     """
#     对每个 (t, l) 位置仅在通道维做 LayerNorm，避免跨时间统计泄漏。
#     """

#     def __init__(self, channels: int, eps: float = 1e-5):
#         super().__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(channels))
#         self.bias = nn.Parameter(torch.zeros(channels))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x.permute(0, 2, 3, 1)  # (B, T, L, C)
#         x = F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)
#         return x.permute(0, 3, 1, 2).contiguous()


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
        ## 改为replication pad2d
        x = F.pad(x, (self.level_pad, self.level_pad, self.time_pad, 0), mode='replicate')
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
            ## 保持时序维度长度不变
            # nn.ReplicationPad2d((0, 0, 0, 1)),
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
        ## 
        self.ffn = nn.Sequential(
            ChannelLayerNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
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
        ## 整合信息
        x = self.ffn(x)
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
            kernel_size=(time_kernel, 1),
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
    def __init__(self, in_channels: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # 将输入映射到 d_model 空间
        self.feat_proj = nn.Linear(in_channels, d_model)
        
        # 定义一个可学习的 Query 向量，用于代表“理想的聚合特征”
        # 形状为 (1, 1, 1, d_model)
        self.query = nn.Parameter(torch.randn(1, 1, 1, d_model))
        
        self.scale = d_model ** -0.5
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, L) -> (B, T, L, C)
        B, C, T, L = x.shape
        x = x.permute(0, 2, 3, 1) 
        
        # 映射特征维度
        feat = self.feat_proj(x)  # (B, T, L, d_model)
        
        # 准备 Query: 扩展到每个 Batch 和 Time step
        # q: (B, T, 1, d_model)
        q = self.query.expand(B, T, 1, -1)
        
        # 计算注意力权重: (B, T, 1, d_model) * (B, T, d_model, L) -> (B, T, 1, L)
        attn = torch.softmax(
            torch.matmul(q, feat.transpose(-1, -2)) * self.scale, 
            dim=-1
        )
        
        # 聚合: (B, T, 1, L) * (B, T, L, d_model) -> (B, T, 1, d_model)
        out = torch.matmul(attn, feat).squeeze(2)  # (B, T, d_model)
        
        return self.out_proj(out)



class LOBEncoder(nn.Module):
    """
    多尺度 LOB 编码器：
    - Stage100ms: LevelAttention + ResStage
    - Stage500ms: Causal(k=stride_1, s=stride_1) + LevelAttention + ResStage
    - Stage1000ms: Causal(k=stride_2, s=stride_2) + LevelAttention + ResStage
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 32,
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
        output_dim: Optional[int] = None,
        # revin: bool = False,
    ):
        super().__init__()

        num_transitions = len(time_strides) ## 降频后的多尺度的层数
        num_stages = num_transitions + 1  ## 单尺度下的特征提取器层数，加上最开始的尺度
        self.return_multi_scale = return_multi_scale
        self._output_dim = int(output_dim or base_channels)
        self._downsample_ratio = 1
        for stride in time_strides:
            self._downsample_ratio *= stride



        ## 映射通道数
        # self.input_norm = nn.GroupNorm(in_channels,in_channels) ## 如果之前已经做了归一化的处理，就可以先不归一化
        ## 对于20level的ask和bid 做一个交互，变为10level
        self.stem = nn.Sequential(
            
            nn.Conv2d(
            in_channels,
            stage_channels[0],
            kernel_size=(1, 2),
            stride = (1, 2),
            # padding=(0,  // 2), ## 保持level维度长度不变
            bias=False,
            ),
            ChannelLayerNorm2d(stage_channels[0]),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        ) ## 第一次要做postnorm处理，后面不需要
        ## 多尺度特征提取
        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()
        # self.scale_aggregators = nn.ModuleList()

        ## 第一层的提取


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
            if idx == num_transitions:
                self.scale_aggregators = nn.Sequential(
                    ChannelLayerNorm2d(stage_channels[idx]),
                    LevelAggregator(in_channels=stage_channels[idx], d_model=self._output_dim),
                    nn.GELU(),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                )
            if idx < num_transitions:
                self.transitions.append(
                    ScaleTransition(
                        in_channels=stage_channels[idx],
                        out_channels=stage_channels[idx + 1],
                        time_kernel=time_kernels[idx],
                        time_stride=time_strides[idx],
                        level_kernel=level_kernels[idx],
                        level_stride=level_strides[idx],
                        dropout=dropout,
                    )
                )

        # # 便于外部读取每个尺度，默认按 100ms 基准累计命名
        # self.scale_names = []
        # cumulative = 1
        # self.scale_names.append(f"{100 * cumulative}ms")
        # for stride in time_strides:
        #     cumulative *= stride
        #     self.scale_names.append(f"{100 * cumulative}ms")

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
            # scale_features.append(self.scale_aggregators[idx + 1](x)) ## (B, stage_channels[idx + 1], T_ds, L) -> (B, output_dim, T_ds, L)
        final_feature = self.scale_aggregators(x) # (B, T, d_model)
        # final_feature = scale_features[-1]
        # if self.return_multi_scale:
        #     scale_dict = {name: feat for name, feat in zip(self.scale_names, scale_features)}
        #     return {
        #         "final": final_feature,
        #         "scale_dict": scale_dict,
        #         "scale_list": scale_features,
        #     }
        return final_feature
