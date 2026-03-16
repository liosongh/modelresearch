# Multi-Modal LOB & Trade Transformer

## 项目概述

这是一个多模态高频量化预测模型，支持 **LOB (限价订单簿) + Trade (逐笔成交) + X** 的灵活融合。

### 核心特性

- 🏗️ **模块化架构**: 每个模态使用独立的 Encoder，易于扩展
- 🔗 **灵活融合**: 支持 Late Concat / Gated / Cross-Attention 多种融合策略
- ⚡ **高效训练**: 支持 AMP 混合精度、梯度累积
- 📊 **全面评估**: 关注 Up/Down 类别的 F1-Score，避免 Stationary 类陷阱

---

## 目录结构

```
Trade_LOB_MultiModal/
├── Configs/                # 配置文件
│   ├── model_config.yaml   # 模型参数
│   └── train_config.yaml   # 训练参数
├── Data_Pipeline/          # 数据工程
│   ├── preprocessors/      # 归一化、清洗
│   ├── generators/         # 标签生成
│   └── dataset.py          # 多模态 Dataset
├── Model/                  # 模型定义
│   ├── encoders/           # LOB/Trade 编码器
│   ├── fusion/             # 融合层
│   ├── backbones/          # Transformer 主干
│   └── multi_modal_transformer.py  # 主模型
├── Train/                  # 训练模块
│   ├── trainer.py          # 训练器
│   └── train_utils.py      # 工具函数
├── Utils/                  # 通用工具
│   ├── metrics.py          # 评估指标
│   ├── losses.py           # 损失函数
│   └── config_loader.py    # 配置加载
└── run_train.py            # 训练入口
```

---

## 快速开始

### 1. 环境要求

```bash
Python >= 3.8
PyTorch >= 2.0
numpy
scikit-learn
pyyaml
tqdm
tensorboard (optional)
```

### 2. 数据准备

LOB 数据格式: `(N, C, L)` 或 `(N, D)`
- N: 时间步数
- C: 通道数 (e.g., 4 = ask_p, bid_p, ask_v, bid_v)
- L: Level 数 (e.g., 10)

Trade 数据格式: `(N, F)` (可选)
- F: 特征数 (e.g., volume, vwap, num_trades, ofi)

### 3. 训练

```bash
# 仅使用 LOB 数据
python run_train.py --lob_data path/to/lob.npy

# LOB + Trade
python run_train.py \
    --lob_data path/to/lob.npy \
    --trade_data path/to/trade.npy \
    --epochs 100 \
    --batch_size 256 \
    --lr 0.001
```

---

## 模型架构

```
                    ┌─────────────────┐
                    │   LOB Data      │
                    │ (B, C, T, L)    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  LOB Encoder    │
                    │ (Causal CNN)    │
                    └────────┬────────┘
                             │
                             │  (B, T_ds, D_lob)
                             │
    ┌─────────────────┐      │      ┌─────────────────┐
    │  Trade Data     │      │      │  Other Data     │
    │ (B, F, T)       │      │      │ (Optional)      │
    └────────┬────────┘      │      └────────┬────────┘
             │               │               │
    ┌────────▼────────┐      │      ┌────────▼────────┐
    │ Trade Encoder   │      │      │ Other Encoder   │
    │ (Causal 1D CNN) │      │      │ (Custom)        │
    └────────┬────────┘      │      └────────┬────────┘
             │               │               │
             │  (B, T_ds, D_trade)           │
             │               │               │
             └───────────────┼───────────────┘
                             │
                    ┌────────▼────────┐
                    │  Feature Fusion │
                    │ (Concat/Gated)  │
                    └────────┬────────┘
                             │
                             │  (B, T_ds, D_model)
                             │
                    ┌────────▼────────┐
                    │   Transformer   │
                    │   Backbone      │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Output Head    │
                    │ (Cls + Reg)     │
                    └─────────────────┘
```

---

## 配置说明

### model_config.yaml

```yaml
lob_encoder:
  in_channels: 4        # LOB 通道数
  base_channels: 32     # 编码器基础通道
  time_strides: [2, 5, 1]  # 时间降采样 (总 10x)

trade_encoder:
  in_features: 8        # Trade 特征数
  hidden_channels: [32, 64]
  time_stride: 10       # 需与 LOB 对齐

fusion:
  strategy: "late_concat"  # 融合策略
  d_model: 128

transformer:
  d_model: 128
  nhead: 4
  num_layers: 3
```

### train_config.yaml

```yaml
data:
  history_T: 3000       # 历史窗口 (100ms * 3000 = 5min)
  sample_stride: 50     # 采样步长

training:
  epochs: 100
  use_amp: true
  early_stopping:
    enabled: true
    patience: 15
    monitor: "val_f1_updown"
```

---

## 扩展指南

### 添加新的数据模态

1. 在 `Model/encoders/` 下创建新的 Encoder，继承 `BaseEncoder`
2. 在 `MultiModalTransformer.__init__()` 中注册新 Encoder
3. 在 `Configs/model_config.yaml` 中添加新模态配置

```python
# 示例: 添加 Order Flow 编码器
class OrderFlowEncoder(BaseEncoder):
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    @property
    def downsample_ratio(self) -> int:
        return self._time_stride
    
    def forward(self, x):
        # 实现编码逻辑
        pass
```

---

## 开发规范 (SOP)

详细开发规范请参考 [README_DEV_SOP.md](README_DEV_SOP.md)

---

## License

MIT License
