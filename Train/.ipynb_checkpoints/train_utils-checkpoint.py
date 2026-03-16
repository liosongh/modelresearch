# =============================================================================
# Training Utilities - 训练工具函数
# Version: v2.0
# =============================================================================

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    CosineAnnealingLR,
    StepLR,
    OneCycleLR
)
from typing import Dict, Optional, Any
import numpy as np

import sys
sys.path.append('..')
from Utils.losses import FocalLoss, DualLoss


def setup_optimizer(model: nn.Module, config: Dict) -> torch.optim.Optimizer:
    """
    根据配置创建优化器。
    
    Args:
        model: 模型
        config: 优化器配置
        
    Returns:
        optimizer: 优化器实例
    """
    opt_type = config.get('type', 'AdamW')
    lr = config.get('lr', 1e-3)
    weight_decay = config.get('weight_decay', 0.01)
    betas = tuple(config.get('betas', [0.9, 0.999]))
    eps = config.get('eps', 1e-8)
    
    if opt_type == 'AdamW':
        optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )
    elif opt_type == 'Adam':
        optimizer = Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )
    elif opt_type == 'SGD':
        momentum = config.get('momentum', 0.9)
        optimizer = SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum
        )
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")
        
    return optimizer


def setup_scheduler(
    optimizer: torch.optim.Optimizer, 
    config: Dict,
    num_training_steps: Optional[int] = None
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    根据配置创建学习率调度器。
    
    Args:
        optimizer: 优化器
        config: 调度器配置
        num_training_steps: 总训练步数 (用于 OneCycleLR)
        
    Returns:
        scheduler: 调度器实例或 None
    """
    sched_type = config.get('type', 'CosineAnnealingWarmRestarts')
    
    if sched_type == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.get('T_0', 10),
            T_mult=config.get('T_mult', 2),
            eta_min=config.get('eta_min', 1e-6)
        )
    elif sched_type == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.get('T_max', 100),
            eta_min=config.get('eta_min', 1e-6)
        )
    elif sched_type == 'StepLR':
        scheduler = StepLR(
            optimizer,
            step_size=config.get('step_size', 30),
            gamma=config.get('gamma', 0.1)
        )
    elif sched_type == 'OneCycleLR':
        if num_training_steps is None:
            raise ValueError("OneCycleLR requires num_training_steps")
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.get('max_lr', 1e-3),
            total_steps=num_training_steps,
            pct_start=config.get('pct_start', 0.3)
        )
    elif sched_type == 'none' or sched_type is None:
        return None
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")
        
    return scheduler


def setup_loss_functions(config: Dict, device: str = 'cuda') -> DualLoss:
    """
    根据配置创建损失函数。
    
    Args:
        config: 损失函数配置
        device: 设备
        
    Returns:
        loss_fn: DualLoss 实例
    """
    cls_config = config.get('classification', {})
    reg_config = config.get('regression', {})
    loss_weight = config.get('loss_weight', 0.8)
    
    # 分类损失
    cls_type = cls_config.get('type', 'FocalLoss')
    if cls_type == 'FocalLoss':
        class_weights = cls_config.get('class_weights')
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        cls_loss_fn = FocalLoss(
            alpha=cls_config.get('alpha', 1.0),
            gamma=cls_config.get('gamma', 2.0),
            class_weights=class_weights
        )
    elif cls_type == 'CrossEntropy':
        cls_loss_fn = nn.CrossEntropyLoss()
    else:
        cls_loss_fn = FocalLoss()
        
    # 回归损失
    reg_type = reg_config.get('type', 'HuberLoss')
    if reg_type == 'HuberLoss':
        reg_loss_fn = nn.HuberLoss(delta=reg_config.get('delta', 1.0))
    elif reg_type == 'MSELoss':
        reg_loss_fn = nn.MSELoss()
    elif reg_type == 'SmoothL1Loss':
        reg_loss_fn = nn.SmoothL1Loss()
    else:
        reg_loss_fn = nn.HuberLoss()
        
    return DualLoss(
        cls_loss_fn=cls_loss_fn,
        reg_loss_fn=reg_loss_fn,
        alpha=loss_weight
    )


class EarlyStopping:
    """
    Early Stopping 回调。
    
    当监控指标不再改善时提前停止训练。
    
    Args:
        patience: 容忍的 epoch 数
        min_delta: 最小改善量
        mode: 'min' (监控 loss) 或 'max' (监控 accuracy/f1)
        restore_best: 是否恢复最佳权重
    """
    
    def __init__(
        self, 
        patience: int = 10, 
        min_delta: float = 0.001,
        mode: str = 'max',
        restore_best: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state_dict = None
        
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        检查是否应该停止。
        
        Args:
            score: 当前指标值
            model: 模型
            
        Returns:
            should_stop: 是否应该停止
        """
        if self.best_score is None:
            self.best_score = score
            if self.restore_best:
                self.best_state_dict = model.state_dict().copy()
            return False
            
        # 检查是否有改善
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
            
        if improved:
            self.best_score = score
            self.counter = 0
            if self.restore_best:
                self.best_state_dict = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop
    
    def restore(self, model: nn.Module) -> None:
        """恢复最佳权重。"""
        if self.restore_best and self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)


def set_seed(seed: int = 42) -> None:
    """设置随机种子以保证可复现性。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
