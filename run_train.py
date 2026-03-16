#!/usr/bin/env python
# =============================================================================
# Run Training - 训练运行脚本
# Version: v2.0
# =============================================================================

import argparse
import numpy as np
import torch
from pathlib import Path
import polars as pl
# 模块导入

from Model import MultiModalTransformer

from Train.trainer import Trainer
from Train.train_utils import (
    setup_optimizer,
    setup_scheduler,
    setup_loss_functions,
    set_seed
)

from Utils.config_loader import load_config, merge_configs
# 设置随机种子
set_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

print("加载配置...")
model_config_path = '/root/lio/Trade_LOB_MultiModal/Configs/model_config.yaml'
train_config_path = '/root/lio/Trade_LOB_MultiModal/Configs/train_config.yaml'
model_config = load_config(model_config_path)
train_config = load_config(train_config_path)
## 把当前配置保存，方便后续更新迭代对比
import yaml
model_version = model_config.get('model_version', 'multi_modal_model_1')
print(f"model_version: {model_version}")
config_save_path = f"./checkpoints/{model_version}"
config_save_path = Path(config_save_path)
# 确保保存目录存在
config_save_path.mkdir(exist_ok=True)
## 保存model的config
model_config_save_path = f"{config_save_path}/model_config.yaml"
train_config_save_path = f"{config_save_path}/train_config.yaml"

with open(model_config_save_path, 'w') as f:
    yaml.dump(model_config, f, indent=4, sort_keys=False, allow_unicode=True)
with open(train_config_save_path, 'w') as f:
    yaml.dump(train_config, f, indent=4, sort_keys=False, allow_unicode=True)

## 读取
lob_data = np.load(train_config.get('data', {}).get('lob_path', ''))
trade_data = np.load(train_config.get('data', {}).get('trade_path', ''))
labels_ret = np.load(train_config.get('data', {}).get('label_path', ''))


print(f"trade_data_agg.shape: {trade_data.shape}")
print(f"lob_data.shape: {lob_data.shape}")
print(f"labels_ret.shape: {labels_ret.shape}")
## 标签处理
alpha= 0.001
labels_class = np.ones_like(labels_ret, dtype=np.int8)
# 3. 向量化赋值：涨→2，跌→0
# 涨：labels_ret > alpha
labels_class[labels_ret > alpha] = 2
# 跌：labels_ret < -alpha
labels_class[labels_ret < -alpha] = 0



## 创建数据集
from Data_Pipeline.dataset import create_dataloaders
 # 创建 DataLoader
print("创建 DataLoader...")
all_config = '/root/lio/Trade_LOB_MultiModal/Configs/experiment_config.yaml'
with open(all_config, 'r') as f:
    all_config = yaml.safe_load(f)
## 两类数据
data_dict = {'lob': lob_data, 'trade': trade_data}
train_loader, val_loader = create_dataloaders(
    data_dict=data_dict,
    labels=labels_class,
    returns=labels_ret,
    config=all_config,
    device=device
)

print(f"训练集大小: {len(train_loader.dataset)}")
print(f"验证集大小: {len(val_loader.dataset)}")


# 创建模型
print("创建模型...")

from Model import MultiModalTransformer
# 根据数据情况调整配置
lob_config = model_config.get('lob_encoder', {})
trade_config = model_config.get('trade_encoder') if 'trade' in data_dict else None
fusion_config = model_config.get('fusion', {})
transformer_config = model_config.get('transformer', {})
output_config = model_config.get('output_head', {})

model = MultiModalTransformer(
    lob_config=lob_config,
    trade_config=trade_config,
    fusion_config=fusion_config,
    transformer_config=transformer_config,
    output_config=output_config,
    use_revin=False
)

# 设置优化器和调度器
optimizer = setup_optimizer(model, train_config.get('optimizer', {}))
scheduler = setup_scheduler(optimizer, train_config.get('scheduler', {}))

# 设置损失函数
loss_fn = setup_loss_functions(train_config.get('loss', {}), device=device)


# 创建训练器
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    config=train_config,
    variant_name=model_version,
    device=device,
    seed=42
)

# 开始训练
print("=" * 50)
history = trainer.fit()

print("=" * 50)
print("训练完成!")
print(f"最佳验证 Loss: {min(history['train_loss']):.4f}")
