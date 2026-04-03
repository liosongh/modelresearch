# =============================================================================
# Multi-Modal Dataset for LOB + Trade Data
# Version: v2.0
# =============================================================================

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, Optional, Tuple, List
import yaml
import polars as pl
from src.data.datapipeline.transform.lob_data_trans import transform_lob_data


dataset_config_path = '/root/lio/modelresearch/Configs/experiment_config.yaml'
model_config_path = '/root/lio/modelresearch/Configs/model_config.yaml'
with open(dataset_config_path, 'r') as f:
    dataset_config = yaml.safe_load(f)

with open(model_config_path, 'r') as f:
    model_config = yaml.safe_load(f)

history_T = dataset_config.get('data', {}).get('history_T', 3000)
sample_stride = dataset_config.get('data', {}).get('sample_stride', 50)


batch_size = dataset_config.get('dataloader', {}).get('batch_size', 256)
num_workers = dataset_config.get('dataloader', {}).get('num_workers', 4)


split_date = '2025-11-30'
from datetime import datetime, timezone
dt_utc = datetime.strptime(split_date, "%Y-%m-%d").replace(
tzinfo=timezone.utc, hour=0, minute=0, second=0, microsecond=0
)
split_ts_ms = int(dt_utc.timestamp() * 1000)  # 转毫秒

lob_encoder_name = model_config.get('lob_encoder', {}).get('encoder_name', 'lob_encoder')
data_dir_dict = dataset_config.get('data_dir_dict', {})
labels_dir = dataset_config.get('labels_dir', {})

class MultiModalDataset(Dataset):
    """
    多模态数据集，支持 LOB + Trade + 其他模态的灵活组合。
    
    设计原则:
    1. 数据字典化: 不同模态通过 key 区分，便于扩展
    2. 时间对齐: 所有模态必须在输入前对齐到相同的时间戳
    3. 第一维度必须是时间步数
    
    Args:
        data_dict: 模态数据字典, e.g., {'lob': (T, C, L), 'trade': T, F)}
        labels: 标签数组 (N,)
        augment: 是否启用数据增强 (随机偏移)
    """
    
    def __init__(
        self,
        data_dict: Dict[str, np.ndarray],
        labels: np.ndarray,
        augment: bool = True
    ):
        self.T = history_T
        self.stride = sample_stride
        self.augment = augment

        self.modalities = list(data_dict.keys())
        self.total_len = labels.shape[0]
        self.num_classes = labels.shape[1]
        self.labels = labels
        
        self.data: Dict[str, torch.Tensor] = data_dict

        
        self.valid_indices = np.arange(0, self.total_len - self.T + 1, self.stride)
        
    def __len__(self) -> int:
        return self.valid_indices.shape[0]
    
    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        base_idx = self.valid_indices[index]
        start = base_idx
        end = start + self.T
        seq_x = {k: torch.from_numpy(v[start:end, :]).float() for k, v in self.data.items()}
        
        label_idx = end - 1
        seq_y = torch.from_numpy(self.labels[label_idx]).float()
        return seq_x, seq_y



def create_ETHUSDT_dataset(split: str = 'train'):
    data_dict = {k: pl.read_parquet(v) for k, v in data_dir_dict.items()}
    labels= pl.read_parquet(labels_dir)
    print('读取数据')
    for k,v in data_dict.items():
        print(k,v.shape)
    print('读取标签')
    print(labels.shape)

    if split == 'train':
        data_dict = {k: v.filter(pl.col('time_bucket') < split_ts_ms) for k, v in data_dict.items()}
        labels = labels.filter(pl.col('time_bucket') < split_ts_ms)

    elif split == 'val':
        data_dict = {k: v.filter(pl.col('time_bucket') >= split_ts_ms) for k, v in data_dict.items()}
        labels = labels.filter(pl.col('time_bucket') >= split_ts_ms)


    elif split != 'test':
        raise ValueError(f"Invalid split: {split}")

    data_dict['lob'] = transform_lob_data(data_dict['lob'],lob_encoder_name)
    labels = labels.drop(['time_bucket']).to_numpy()

    print('转换数据')
    for k,v in data_dict.items():
        print(k,v.shape)
    print('转换标签')
    print(labels.shape)
    
    return MultiModalDataset(data_dict, labels, augment = False)



    
