from ..layers.tcn import TemporalConvNet
from ..layers.lob.lob_encoder import LOBEncoder
from ..layers.lob.level_wise_encoder import LevelWiseEncoder
from ..layers.lob.deeplob_encoder import Deeplob_encoder_simple,Deeplob_encoder
import torch
import torch.nn as nn
# from torchinfo import summary
import yaml
from ..layers.RevIN import RevIN2d

lob_encoder_dict = {
    'lob_encoder': LOBEncoder,
    'level_wise_encoder': LevelWiseEncoder,
    'Deeplob_encoder_simple': Deeplob_encoder_simple,
    'DeepLOB_encoder': Deeplob_encoder
}
# ===================== 2. 完整预测模型：编码器 + 膨胀因果卷积头 =====================
class LOB_TCN(nn.Module):
    def __init__(self, model_config,num_classes: int = 1):
        """
        :param num_classes: 预测输出维度（1=回归，3=三分类，自定义）
        """
        super().__init__()
        params = model_config['lob_encoder']['params']
        lob_encoder_name = model_config['lob_encoder']['encoder_name']
        in_channels = params['in_channels']
        self.revin = None
        revin = model_config['revin']
        if revin:
            self.revin  = RevIN2d(in_channels)
        # 1. 特征编码器（你的原有模型）
        encoder_class = lob_encoder_dict[lob_encoder_name]
        self.encoder = encoder_class(**params)
        self.encoder_dim = params['output_dim']
        # 2. 维度融合
        num_channels = model_config['time_module']['params']['num_channels']
        self.fusion = nn.Linear(self.encoder_dim, num_channels[0]) if self.encoder_dim != num_channels[0] else nn.Identity()

        # 3. TCN 时序提取
        self.tcn = TemporalConvNet(num_channels[0], num_channels[1:], kernel_size=2, dropout=0.2)

        # 4. 预测头（自适应池化 + 全连接层）
        # self.adaptive_pool = nn.AdaptiveAvgPool1d(1)  # 压缩时序维度为1
        # self.last_step = 
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, inputs):
        """
        输入x形状：[Batch, t, c, l] (LOB标准输入)
        输出：[Batch, num_classes] (预测结果)
        """
        x = inputs['lob']
        x = x.permute(0, 2, 1, 3) # (B, c, t, l)
        if self.revin is not None:
            x = self.revin(x)
        # ========== 步骤1：编码器提取特征 ==========
        feat = self.encoder(x)  # 输出: [B, T, d_model]

        feat = self.fusion(feat) # (B, T, d_model)
        # ========== 步骤2：维度转换（适配Conv1d） ==========
        # Conv1d要求输入: [Batch, Channels, Length] → 转置 [B, d_model, T]
        feat = feat.permute(0, 2, 1)

        # ========== 步骤3：膨胀因果卷积提取时序特征 ==========
        feat = self.tcn(feat)  # [B, d_model, T]

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