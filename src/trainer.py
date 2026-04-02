# =============================================================================
# Trainer - 训练器模块
# Version: v2.0
# =============================================================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Callable
import time
import json

import sys
sys.path.append('..')
from Utils.metrics import compute_metrics, compute_updown_metrics, get_confusion_matrix
from Utils.losses import DualLoss
from .train_utils import EarlyStopping, set_seed


class Trainer: 
    def __init__(
        self,
        # model: nn.Module,
        # train_loader: DataLoader,
        # val_loader: DataLoader,
        loss_fn: DualLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        config: Dict = None,
        # device: str = 'cuda',
        variant_name: str = None,
        seed: int = None
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.model = model.to(self.device)
        # self.train_loader = train_loader
        # self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        
        # 配置
        # self.config = config or {}
        training_cfg = config
        
        self.epochs = training_cfg.get('epochs', 100)
        self.use_amp = training_cfg.get('use_amp', True)
        self.grad_accumulation_steps = training_cfg.get('grad_accumulation_steps', 1)
        self.max_grad_norm = training_cfg.get('gradient_clip', 1.0)

        
        # Early Stopping
        es_cfg = training_cfg.get('early_stopping', {})
        self.min_delta = es_cfg.get('min_delta', 0.001)
        if es_cfg.get('enabled', True):
            self.monitor = es_cfg.get('monitor', 'val_f1_updown')
            self.mode = es_cfg.get('mode', 'max')
            self.early_stopping = EarlyStopping(
                patience=es_cfg.get('patience', 15),
                min_delta=es_cfg.get('min_delta', 0.001),
                mode=self.mode
            )
            
        else:
            self.early_stopping = None
            
        # Checkpointing
        output_cfg = training_cfg.get('output', {})
        self.save_dir = Path(output_cfg.get('checkpoint_dir', './checkpoints'))
        seed_dir = f"seed_{seed}"
        if variant_name is not None and seed is not None:
            self.model_save_dir = self.save_dir / variant_name/ seed_dir
        else:
            self.model_save_dir = self.save_dir
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # AMP Scaler
        if self.use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None
            
        # Logging
        
        self.log_interval = output_cfg.get('log_interval', 10)
        # TensorBoard
        self.writer = None
        if output_cfg.get('use_tensorboard', False):
            from torch.utils.tensorboard import SummaryWriter
            log_dir = Path(output_cfg.get('log_dir', './logs'))
    
            self.writer = SummaryWriter(log_dir/variant_name/ seed_dir)


        
    def train_epoch(self) -> Tuple[float, float]:
        """
        训练一个 epoch。
        
        Returns:
            avg_loss: 平均损失
            accuracy: 准确率
        """
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        pbar = tqdm(self.train_loader, desc='Training', leave=False)

        for batch_idx, (inputs, labels) in enumerate(pbar):


            # 移动数据到设备
            # if isinstance(inputs, dict):
            inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
 
            labels = labels.to(self.device, non_blocking=True)

            # 前向传播
            with autocast(device_type='cuda', enabled=self.use_amp):
                logits = self.model(inputs)
                # cls_pred = output['backbone_output']
                # reg_pred = None
                losses = self.loss_fn(logits, labels)
                # loss = losses['total']
            # 反向传播
            if self.use_amp:
                self.scaler.scale(losses).backward()
            else:
                losses.backward()


            # 梯度累积
            if (batch_idx + 1) % self.grad_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    
            self.optimizer.zero_grad(set_to_none=True)
            # 记录
            running_loss += losses.item() * labels.size(0)
            ## 每100个batch记录一次
            if batch_idx % 100 == 0:   
                pbar.set_postfix({'loss': f'{losses.item():.8f}'})

            
        pbar.close()

        
        avg_loss = running_loss / len(self.train_loader.dataset)
        # accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        验证模型。
        
        Returns:
            metrics: 验证指标字典
        """
        self.model.eval()
        running_loss = 0.0

        
        pbar = tqdm(self.val_loader, desc='Validation', leave=False)
        
        for inputs, labels in pbar:
            # 移动数据
            if isinstance(inputs, dict):
                inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
            else:
                inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            # returns = returns.to(self.device, non_blocking=True)
            
            # 前向传播
            with autocast(device_type='cuda', enabled=self.use_amp):
                logits = self.model(inputs)

                losses = self.loss_fn(logits, labels)
                
            running_loss += losses.item() * labels.size(0)
            
            # preds = torch.argmax(cls_pred, dim=1)
            # all_preds.extend(preds.cpu().numpy())
            # all_labels.extend(labels.cpu().numpy())
            
        pbar.close()
        
        # # 计算指标
        # all_preds = np.array(all_preds)
        # all_labels = np.array(all_labels)
        
        avg_loss = running_loss / len(self.val_loader.dataset)
        # metrics = compute_metrics(all_labels, all_preds)
        # updown_metrics = compute_updown_metrics(all_labels, all_preds)
        
        # metrics['loss'] = avg_loss
        # # metrics.update(updown_metrics)
        # metrics['confusion_matrix'] = get_confusion_matrix(all_labels, all_preds,normalize=False)
        
        return avg_loss
    
    def fit(self,model,train_loader,val_loader) -> Dict:
        """
        完整训练流程。
        
        Returns:
            history: 训练历史
        """
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader


        print(f"开始训练，共 {self.epochs} 个 epoch")
        print(f"设备: {self.device}")
        print(f"AMP: {self.use_amp}")
        print(f"梯度累积步数: {self.grad_accumulation_steps}")
        print("-" * 50)
        # if self.mode == 'min':
        #     best_monitor = float('inf')

        # else:
        #     best_monitor = float('-inf')
        train_history = []
        # best_val_metrics = None
        # best_epoch = None
        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss = self.validate()
            
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()
                
            # 记录历史
            train_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                # 'train_acc': train_acc,
                'val_metrics': val_loss,
                # 'train_perf': self.last_train_perf
            })

            
            # 打印
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch}/{self.epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.8f}")
            print(f"  Val Loss: {val_loss:.8f}")
            
            # TensorBoard
            if self.writer is not None:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                
            # # 保存最佳模型
            # if self.mode == 'min':
            #     if val_metrics < best_monitor - self.min_delta:
            #         best_monitor = val_metrics
            #         best_val_metrics = val_metrics
            #         best_epoch = epoch
            #         self.save_checkpoint(f'best_model.pt', epoch, val_metrics)
            #         print(f"  [NEW BEST] Loss: {best_monitor:.4f}")
            # elif self.mode == 'max':
            #     if val_metrics > best_monitor + self.min_delta:
            #         best_monitor = val_metrics
            #         best_val_metrics = val_metrics
            #         best_epoch = epoch
            #         self.save_checkpoint(f'best_model.pt', epoch, val_metrics)
            #         print(f"  [NEW BEST] Loss: {best_monitor:.4f}")

                
                
            # Early Stopping
            if self.early_stopping is not None:
                if self.early_stopping(val_loss, self.model):
                    print(f"\nEarly stopping at epoch {epoch}")
                    self.early_stopping.restore(self.model)
                    break
                    
            # print()
            
        # 保存最终模型
        self.save_checkpoint('final_model.pt', epoch, val_loss)
        
        if self.writer is not None:
            self.writer.close()
            
        return {'train_history': train_history, 
                # 'best_val_metrics': best_val_metrics, 
                # 'best_epoch': best_epoch
                }
    
    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict) -> None:
        """保存检查点。"""
        model_to_save = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        torch.save(checkpoint, self.model_save_dir / filename)
        
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """加载检查点。"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        if 'history' in checkpoint:
            self.history = checkpoint['history']
            
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
