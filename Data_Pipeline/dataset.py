# =============================================================================
# Multi-Modal Dataset for LOB + Trade Data
# Version: v2.0
# =============================================================================

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Optional, Tuple, List


class MultiModalDataset(Dataset):
    """
    多模态数据集，支持 LOB + Trade + 其他模态的灵活组合。
    
    设计原则:
    1. 数据字典化: 不同模态通过 key 区分，便于扩展
    2. 归一化前置: 归一化在数据预处理阶段完成，Dataset 不做计算
    3. 时间对齐: 所有模态必须在输入前对齐到相同的时间戳
    
    Args:
        data_dict: 模态数据字典, e.g., {'lob': (N, C, L), 'trade': (N, F)}
        labels: 分类标签 (N,)
        returns: 回归目标 (N,)
        T: 历史窗口长度
        stride: 采样步长 (训练时可设大以减少样本重叠)
        device: 数据存放设备
        augment: 是否启用数据增强 (随机偏移)
    """
    
    def __init__(
        self,
        data_dict: Dict[str, np.ndarray],
        labels: np.ndarray,
        returns: np.ndarray,
        T: int = 3000,
        stride: int = 1,
        device: str = 'cpu',
        augment: bool = True
    ):
        self.T = T
        self.stride = stride
        self.device = device
        self.augment = augment
        self.modalities = list(data_dict.keys())
        self.total_len = labels.shape[0]
        # 存储各模态数据
        self.data: Dict[str, torch.Tensor] = {}
        self.data_shapes: Dict[str, Tuple] = {}
        
        for key, arr in data_dict.items():
            tensor = torch.as_tensor(arr, dtype=torch.float32)
            # # 保持数据在 CPU，训练时按需搬运（或直接放 GPU 如果显存够）
            # if device != 'cpu':
            #     tensor = tensor.to(device).contiguous()
            # else:
            #     tensor = tensor.contiguous()
            # 🔴 提前完成所有维度变换（只做1次，而非每个样本重复做）
            if key == 'lob':
                # (N, C, L) → (C, N, L) （后续切片直接取T长度）
                tensor = tensor.permute(1, 0, 2).cuda().contiguous()
            elif key == 'trade':
                # (N, F) → (F, N)
                tensor = tensor.transpose(0, 1).cuda().contiguous()
            else:
                # (N,) → (1, N)
                tensor = tensor.unsqueeze(0).cuda().contiguous()
            self.data[key] = tensor
            self.data_shapes[key] = tensor.shape
            
        # 获取总时间长度（假设所有模态第一维是时间）
        # first_key = self.modalities[0]
        # self.total_len = self.data[first_key].shape[1]
        
        # # 验证所有模态时间长度一致
        # for key in self.modalities:
        #     assert self.data[key].shape[1] == self.total_len, \
        #         f"时间维度不一致: {first_key}={self.total_len}, {key}={self.data[key].shape[1]}"
        
        # # 处理标签 (对齐到窗口末端)
        # # 标签从 T-1 开始有效（需要 T 个历史点）
        # valid_labels = labels[T-1:self.total_len]
        # valid_returns = returns[T-1:self.total_len]
        
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.returns = torch.as_tensor(returns, dtype=torch.float32)


        # 2. 开启共享内存，彻底关闭多进程 CoW（最关键！不占显存）
        for k in self.data:
            self.data[k] = self.data[k].share_memory_()
        self.labels = self.labels.share_memory_()
        self.returns = self.returns.share_memory_()
        
        # if device != 'cpu':
        # self.labels = self.labels.to(device).contiguous()
        # self.returns = self.returns.to(device).contiguous()
        
        # 计算有效索引
        self.valid_indices = np.arange(0, self.total_len - T + 1, stride)
        # self.valid_indices = torch.as_tensor(self.valid_indices, dtype=torch.long)
        
    def __len__(self) -> int:
        return self.valid_indices.shape[0]
    
    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        base_idx = self.valid_indices[index]
        
        # # 数据增强: 随机偏移 (仅训练时)
        # if self.augment and self.stride > 1:
        #     max_offset = min(self.stride, self.total_len - self.T - base_idx)
        #     offset = torch.randint(0, max(1, max_offset), (1,)).item() if max_offset > 0 else 0
        # else:
        #     offset = 0
            
        start = base_idx
        end = start + self.T
        
        # 构建样本字典
        # sample = {}
        # for key in self.modalities:
        #     if key == 'lob':
        #         # (C, N, L) → (C, T, L) （直接切片，无运算）
        #         sample[key] = self.data[key][:, :, start:end]
        #     elif key == 'trade':
        #         # (F, N) → (F, T)
        #         sample[key] = self.data[key][:, start:end]
        #     else:
        #         # (1, N) → (1, T)
        #         sample[key] = self.data[key][:, start:end]

        sample = {
            "lob": self.data["lob"][:, start:end, :],
            "trade": self.data["trade"][:, start:end],
        }
            
        # 标签索引对应窗口末端
        label_idx = end - 1  # 对应 labels[T-1 + start] 在原始数据中
        
        return sample, self.labels[label_idx], self.returns[label_idx]


def multimodal_collate_fn(batch: List) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    多模态数据的 collate 函数。
    
    职责: 仅负责 stack，不做任何归一化计算。
    归一化应在预处理阶段完成。
    
    Returns:
        inputs: Dict[str, Tensor], 每个模态 stack 后的批量数据
        labels: Tensor (B,)
        returns: Tensor (B,)
    """
    batch_samples, batch_labels, batch_returns = zip(*batch)
    # batch_labels = torch.stack(batch_labels, dim=0)
    # batch_returns = torch.stack(batch_returns, dim=0)
    # 获取所有模态的 key
    modalities = batch_samples[0].keys()

    combined_inputs = {key: torch.stack([s[key] for s in batch_samples], dim=0) for key in modalities}

    # # 🔥 一次性搬一个 batch 到 GPU（比样本级搬运快10倍）
    # combined_inputs = {k: v.cuda(non_blocking=True) for k, v in combined_inputs.items()}
    # batch_labels = batch_labels.cuda(non_blocking=True)
    # batch_returns = batch_returns.cuda(non_blocking=True)
    return combined_inputs, batch_labels, batch_returns


def create_dataloaders(
    data_dict: Dict[str, np.ndarray],
    labels: np.ndarray,
    returns: np.ndarray,
    config: dict,
    device: str = 'cpu'
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证 DataLoader。
    
    Args:
        data_dict: 多模态数据字典
        labels: 标签数组
        returns: 收益率数组
        config: 包含 data 和 dataloader 配置的字典
        device: 设备
        
    Returns:
        train_loader, val_loader
    """
    data_cfg = config.get('data', {})
    loader_cfg = config.get('dataloader', {})
    
    history_T = data_cfg.get('history_T', 3000)
    sample_stride = data_cfg.get('sample_stride', 50)
    split_ratio = data_cfg.get('split_ratio', 0.7)
    
    batch_size = loader_cfg.get('batch_size', 256)
    num_workers = loader_cfg.get('num_workers', 4)
    pin_memory = loader_cfg.get('pin_memory', True)
    drop_last = loader_cfg.get('drop_last', True)
    prefetch_factor = loader_cfg.get('prefetch_factor', 4)
    
    # 计算划分点
    total_len = labels.shape[0]
    split_idx = int(total_len * split_ratio)
    
    # 划分数据
    train_dict, val_dict = {}, {}
    for key, arr in data_dict.items():
        train_dict[key] = arr[:split_idx]
        val_dict[key] = arr[split_idx:]
    
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]
    train_returns, val_returns = returns[:split_idx], returns[split_idx:]
    
    # 创建 Dataset
    train_dataset = MultiModalDataset(
        train_dict, train_labels, train_returns,
        T=history_T, stride=sample_stride, device=device, augment=True
    )
    
    val_dataset = MultiModalDataset(
        val_dict, val_labels, val_returns,
        T=history_T, stride=sample_stride, device=device, augment=False
    )
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers ,  # GPU 上数据时 worker=0
        # num_workers=0,
        # collate_fn=multimodal_collate_fn,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        drop_last=drop_last
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        # num_workers=num_workers if device == 'cpu' else 0,
        num_workers=num_workers,
        # collate_fn=multimodal_collate_fn,
        # pin_memory=pin_memory and device == 'cpu',
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader


def create_dataloaders_for_test(
    data_dict: Dict[str, np.ndarray],
    labels: np.ndarray,
    returns: np.ndarray,
    config: dict,
    device: str = 'cpu'
) -> Tuple[DataLoader, DataLoader]:
    """
    创建测试 DataLoader。
    """
    # data_cfg = config.get('data', {})
    # loader_cfg = config.get('dataloader', {})
    
    history_T = config.get('history_T', 3000)
    batch_size = config.get('batch_size', 256)
    num_workers = config.get('num_workers', 4)
    pin_memory = config.get('pin_memory', True)
    stride = config.get('stride', 1)
    
    test_dataset = MultiModalDataset(
        data_dict, labels, returns,
        T=history_T, stride=stride, device=device, augment=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers if device == 'cpu' else 0,
        # collate_fn=multimodal_collate_fn,
        pin_memory=pin_memory and device == 'cpu',
        drop_last=False
    )
    
    return test_loader