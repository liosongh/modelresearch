# =============================================================================
# Multi-Modal Transformer - 多模态 Transformer 主模型
# Version: v2.1 - 集成 RevIN，使用 GroupNorm 替代 BatchNorm
# =============================================================================

import torch
import torch.nn as nn
from typing import Any, Dict, Optional
import yaml

from layers.lob_encoder import LOBEncoder
from layers.deeplob_encoder import Deeplob_encoder
from layers.trade_encoder import TradeEncoder
from layers.feature_fusion import FeatureFusion
from layers.transformer import TransformerBackbone
from layers.RevIN import RevIN, RevIN2d

import time
class MultiModalTransformer(nn.Module):
    """
    多模态 Transformer 模型。
    
    架构:
    1. 多塔编码器 (LOB Tower, Trade Tower, ...)
    2. 特征融合层
    3. Transformer 主干网络
    4. 输出头 (分类 + 回归)
    
    Input:
        inputs: Dict[str, Tensor]
            - 'lob': (B, C, T, L) LOB 数据
            - 'trade': (B, F, T) Trade 数据
            - ...
            
    Output:
        logits: (B, num_classes) 分类输出
        regression: (B,) 回归输出 (可选)
    """
    
    def __init__(
        self, 
        lob_config: dict,
        trade_config: Optional[dict] = None,
        fusion_config: dict = None,
        transformer_config: dict = None,
        output_config: dict = None,
        use_revin: bool = True  # 是否使用 RevIN 归一化
    ):
        super().__init__()

        encoder_dims = {}
        self.use_revin = use_revin
        self.lob_multi_scale_outputs = None
        
        self.modalities = []
        # --- 1. LOB 编码器 ---
        # self.lob_encoder = LOBEncoder(**lob_config)
        self.lob_encoder = Deeplob_encoder()
        self.modalities.append('lob')
        # 直接使用编码器声明的输出维度，兼容多尺度聚合后的 scale_output_dim
        encoder_dims['lob'] = self.lob_encoder.output_dim
        
        # --- 2. Trade 编码器 (可选) ---
        self.trade_encoder = None
        self.trade_revin = None
        if trade_config is not None:
            self.trade_encoder = TradeEncoder(**trade_config)
            self.modalities.append('trade')
            encoder_dims['trade'] = self.trade_encoder.output_dim
            
            if use_revin:
                # Trade: (B, F, T) -> 对每个特征在 Time 维度归一化
                # 注意：has_trade 是第0个特征（二元特征），不应参与 RevIN
                total_features = trade_config.get('in_features', 8)
                # 只对连续特征进行归一化 (排除 has_trade)
                self.trade_revin = RevIN(num_features=total_features - 1, affine=True)
            
        # --- 3. 特征融合层 ---
        self.d_model = fusion_config['d_model']
        # 融合层会在第一次前向传播时根据实际维度初始化
        # self._fusion_initialized = False
        self._fusion_config = fusion_config
        self._encoder_dims = encoder_dims

        # actual_dims = {k: v.shape[-1] for k, v in encoder_outputs.items()}
        
        self.fusion = FeatureFusion(
            input_dims=encoder_dims, ## 融合前的模态维度
            d_model=self._fusion_config['d_model'], ## 融合后的维度
            strategy=self._fusion_config.get('strategy', 'late_concat'),
            use_layer_norm=self._fusion_config.get('use_layer_norm', True)
        )
        # ### 为了和之前一样的model
        # self.adapter_conv = nn.Conv1d(encoder_dims['lob'], self.d_model, 1)
        # --- 4. Transformer 主干 ---
        self.Backbone = TransformerBackbone(**transformer_config)
        
        # --- 5. 输出头 ---
        self.num_classes = output_config.get('num_classes', 3)
        self.return_regression = output_config.get('return_regression', True)
        self.classifier = nn.Linear(self.d_model, self.num_classes)
        if self.return_regression:
            self.regressor = nn.Linear(self.d_model, 1)
        else:
            self.regressor = None
            
    def _init_fusion(self, encoder_outputs: Dict[str, torch.Tensor]):
        """
        根据实际编码器输出维度初始化融合层。
        """
        actual_dims = {k: v.shape[-1] for k, v in encoder_outputs.items()}
        
        self.fusion = FeatureFusion(
            input_dims=actual_dims,
            d_model=self._fusion_config['d_model'],
            strategy=self._fusion_config.get('strategy', 'late_concat'),
            use_layer_norm=self._fusion_config.get('use_layer_norm', True)
        ).to(next(self.parameters()).device)
        
        self._fusion_initialized = True
            
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        return_features: bool = False
    ) -> Any:
        """
        前向传播。
        
        Args:
            inputs: 多模态输入字典
            
        Returns:
            logits: (B, num_classes)
            regression: (B,) 或 None
        """
        ## 计算forward的时间各个模块的时间
        
        
        
        # 0. RevIN 归一化 (每个样本独立，避免信息泄漏)
        normalized_inputs = {}
        # normalized_inputs_start_time = time.perf_counter()

        normalized_inputs['lob'] = inputs['lob']
        # if self.lob_encoder is not None and 'lob' in inputs:
        #     # lob_data = inputs['lob'].permute(0, 1, 3, 2).contiguous()
        #     # if self.use_revin and self.lob_revin is not None:
        #     #     lob_data = self.lob_revin(lob_data, mode='norm')
        #     normalized_inputs['lob'] = inputs['lob']
            
        if self.trade_encoder is not None and 'trade' in inputs:
            trade_data = inputs['trade'].permute(0, 1, 2).contiguous()
            if self.use_revin and self.trade_revin is not None:
                # Split-Norm-Concat: 分离二元特征和连续特征
                # 假设 has_trade 是第 0 个特征
                binary_feat = trade_data[:, 0:1, :]      # (B, 1, T)
                continuous_feat = trade_data[:, 1:, :]   # (B, F-1, T)
                
                # 只对连续特征归一化
                continuous_feat = self.trade_revin(continuous_feat, mode='norm')
                
                # 拼接回去
                trade_data = torch.cat([binary_feat, continuous_feat], dim=1)
                
            normalized_inputs['trade'] = trade_data
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        # normalized_inputs_time = time.perf_counter() - normalized_inputs_start_time
        # encoder_start_time = time.perf_counter()
        # 1. 编码各模态
        encoded = {}
        # LOB
        if 'lob' in normalized_inputs:
            lob_encoded = self.lob_encoder(normalized_inputs['lob'])
            # 兼容 LOBEncoder 的多尺度输出:
            # - Tensor: 单尺度输出 (B, T, D)
            # - Dict: {"final", "scale_dict", "scale_list"}
            if isinstance(lob_encoded, dict):
                encoded['lob'] = lob_encoded['final']
                self.lob_multi_scale_outputs = lob_encoded.get('scale_dict')
            else:
                encoded['lob'] = lob_encoded  # (B, T_ds, D_lob)
                self.lob_multi_scale_outputs = None
            
        # Trade
        if 'trade' in normalized_inputs:
            encoded['trade'] = self.trade_encoder(normalized_inputs['trade'])  # (B, T_ds, D_trade)
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        # encoder_time = time.perf_counter() - encoder_start_time
        # fusion_start_time = time.perf_counter()
        # # 2. 时间维度对齐检查
        # if len(encoded) > 1:
        #     time_lens = [v.shape[1] for v in encoded.values()]
        #     if len(set(time_lens)) > 1:
        #         raise ValueError(f"时间维度不一致: {time_lens}")
        #         # # 时间维度不一致，需要对齐
        #         # min_t = min(time_lens)
        #         # encoded = {k: v[:, :min_t, :] for k, v in encoded.items()}
                
        # # 3. 特征融合
        # if not self._fusion_initialized:
        #     self._init_fusion(encoded)
        # self.adapter_conv()
        
        fused = self.fusion(encoded)  # (B, T, d_model)
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        # fusion_time = time.perf_counter() - fusion_start_time
        # transformer_start_time = time.perf_counter()
        # 4. Transformer
        output = self.Backbone(fused, causal=True)  # (B, T, d_model)
        output = output[:, -1, :] ## (B, d_model)
        # 6. 输出头
        logits = self.classifier(output) # (B, num_classes)
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        # transformer_time = time.perf_counter() - transformer_start_time

        regression = None
        if self.return_regression and self.regressor is not None:
            regression = self.regressor(output).squeeze(-1)  # (B,)
        # print(f"normalized_inputs_time: {normalized_inputs_time:.4f}")
        # print(f"encoder_time: {encoder_time:.4f}")
        # print(f"fusion_time: {fusion_time:.4f}")
        # print(f"transformer_time: {transformer_time:.4f}")
        if return_features:
            return {
                "logits": logits,
                "encoded": encoded,
                "fused": fused,
                "transformer_output": output,
                "pooled": output,
                "lob_multi_scale": self.lob_multi_scale_outputs,
            }
        return logits
    
    @classmethod
    def from_config(cls, config_path: str) -> 'MultiModalTransformer':
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
            lob_config=config.get('lob_encoder', {}),
            trade_config=config.get('trade_encoder'),
            fusion_config=config.get('fusion', {}),
            transformer_config=config.get('transformer', {}),
            output_config=config.get('output_head', {})
        )


