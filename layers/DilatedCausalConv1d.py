import torch
import torch.nn as nn
from layers.deeplob_encoder import Deeplob_encoder

# ===================== 1. 定义 膨胀因果卷积1D 基础模块 =====================
class DilatedCausalConv1d(nn.Module):
    """
    膨胀因果卷积层（核心：保证因果性，不使用未来信息）
    :param in_channels: 输入通道数
    :param out_channels: 输出通道数
    :param kernel_size: 卷积核大小
    :param dilation: 膨胀率
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        # 因果卷积关键：padding = (kernel_size - 1) * dilation
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding
        )
        # 激活函数+归一化（和原编码器保持一致）
        self.act = nn.LeakyReLU(negative_slope=0.01)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        # 裁剪右侧padding，保证因果性（只保留历史信息）
        x = x[:, :, :-self.padding] if self.padding > 0 else x
        x = self.act(x)
        x = self.bn(x)
        return x

# ===================== 2. 完整预测模型：编码器 + 膨胀因果卷积头 =====================
class LOB_DilatedConv_Predictor(nn.Module):
    def __init__(self, num_classes: int = 1):
        """
        :param num_classes: 预测输出维度（1=回归，3=三分类，自定义）
        """
        super().__init__()
        # 1. 特征编码器（你的原有模型）
        self.encoder = Deeplob_encoder()
        self.encoder_dim = self.encoder.output_dim  # 192

        # 2. 多层膨胀因果卷积（dilation指数增长，扩大感受野）
        self.dilated_conv_layers = nn.Sequential(
            DilatedCausalConv1d(in_channels=self.encoder_dim, out_channels=128, kernel_size=3, dilation=1),
            DilatedCausalConv1d(in_channels=128, out_channels=64, kernel_size=3, dilation=2),
            DilatedCausalConv1d(in_channels=64, out_channels=32, kernel_size=3, dilation=4),
            DilatedCausalConv1d(in_channels=32, out_channels=16, kernel_size=3, dilation=8),
        )

        # 3. 预测头（自适应池化 + 全连接层）
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)  # 压缩时序维度为1
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        """
        输入x形状：[Batch, Channel=1, Height, Width] (LOB标准输入)
        输出：[Batch, num_classes] (预测结果)
        """
        # ========== 步骤1：编码器提取特征 ==========
        feat = self.encoder(x)  # 输出: [B, T, 192]

        # ========== 步骤2：维度转换（适配Conv1d） ==========
        # Conv1d要求输入: [Batch, Channels, Length] → 转置 [B, 192, T]
        feat = feat.permute(0, 2, 1)

        # ========== 步骤3：膨胀因果卷积提取时序特征 ==========
        feat = self.dilated_conv_layers(feat)  # [B, 16, T]

        # ========== 步骤4：池化 + 预测 ==========
        feat = self.adaptive_pool(feat).squeeze(-1)  # [B, 16]
        out = self.fc(feat)  # [B, num_classes]

        return out