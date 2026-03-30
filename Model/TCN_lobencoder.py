from layers.tcn import TemporalConvNet
from layers.lob_encoder import LOBEncoder
import torch
import torch.nn as nn
from torchinfo import summary
import yaml
from layers.RevIN import RevIN2d
# ===================== 2. 完整预测模型：编码器 + 膨胀因果卷积头 =====================
class LOB_TCN(nn.Module):
    def __init__(self, model_config,num_classes: int = 1):
        """
        :param num_classes: 预测输出维度（1=回归，3=三分类，自定义）
        """
        super().__init__()
        in_channels = model_config['lob_encoder']['in_channels']
        self.revin = None
        revin = model_config['revin']
        if revin:
            self.revin  = RevIN2d(in_channels)
        # 1. 特征编码器（你的原有模型）
        self.encoder = LOBEncoder(**model_config['lob_encoder'])
        self.encoder_dim = self.encoder.output_dim  # 192
        num_channels = model_config['tcn']['num_channels']
        self.fusion = nn.Linear(self.encoder_dim, num_channels[0])
        self.tcn = TemporalConvNet(num_channels[0], num_channels[1:], kernel_size=2, dropout=0.2)

        # 3. 预测头（自适应池化 + 全连接层）
        # self.adaptive_pool = nn.AdaptiveAvgPool1d(1)  # 压缩时序维度为1
        # self.last_step = 
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, inputs):
        """
        输入x形状：[Batch, Channel=1, Height, Width] (LOB标准输入)
        输出：[Batch, num_classes] (预测结果)
        """
        x = inputs['lob']
        if self.revin is not None:
            x = self.revin(x)
        # ========== 步骤1：编码器提取特征 ==========
        feat = self.encoder(x)  # 输出: [B, T, 192]

        feat = self.fusion(feat)
        # ========== 步骤2：维度转换（适配Conv1d） ==========
        # Conv1d要求输入: [Batch, Channels, Length] → 转置 [B, 192, T]
        feat = feat.permute(0, 2, 1)

        # ========== 步骤3：膨胀因果卷积提取时序特征 ==========
        feat = self.tcn(feat)  # [B, 16, T]

        # ========== 步骤4：池化 + 预测 ==========
        # feat = self.adaptive_pool(feat).squeeze(-1)  # [B, 16]
        last_step = feat[:, :, -1]  # (batch, num_channels[-1])
        out = self.fc(last_step)  # [B, num_classes]

        return out


    @classmethod
    def from_config(cls, config_path: str):
        """
        从配置文件创建模型。
        
        Args:
            config_path: YAML 配置文件路径
            
        Returns:
            model: MultiModalTransformer 实例
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        return cls(
            config,
            num_classes = 3
  
        )