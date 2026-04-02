# =============================================================================
# Losses - 损失函数模块
# Version: v2.0
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: 类别权重，可以是标量或张量
        gamma: 聚焦参数，gamma > 0 降低易分样本的权重
        reduction: 'none' | 'mean' | 'sum'
        label_smoothing: 标签平滑系数
    """
    
    def __init__(
        self, 
        alpha: float = 1.0, 
        gamma: float = 2.0, 
        reduction: str = 'mean',
        label_smoothing: float = 0.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: 模型输出 logits (B, C)
            targets: 真实标签 (B,)
            
        Returns:
            loss: 标量损失
        """
        # Cross Entropy Loss (without reduction)
        ce_loss = F.cross_entropy(
            inputs, targets, 
            reduction='none',
            label_smoothing=self.label_smoothing,
            weight=self.class_weights
        )
        
        # Focal weight
        pt = torch.exp(-ce_loss)  # p_t
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class DualLoss(nn.Module):
    """
    双任务损失函数 (分类 + 回归)。
    
    Total Loss = alpha * Classification Loss + (1 - alpha) * Regression Loss
    
    Args:
        cls_loss_fn: 分类损失函数
        reg_loss_fn: 回归损失函数
        alpha: 分类损失权重
    """
    
    def __init__(
        self,
        cls_loss_fn: Optional[nn.Module] = None,
        reg_loss_fn: Optional[nn.Module] = None,
        alpha: float = 0.8
    ):
        super().__init__()
        
        self.cls_loss_fn = cls_loss_fn or FocalLoss()
        self.reg_loss_fn = reg_loss_fn or nn.HuberLoss()
        self.alpha = alpha
        
    def forward(
        self,
        cls_pred: torch.Tensor,
        cls_target: torch.Tensor,
        reg_pred: Optional[torch.Tensor] = None,
        reg_target: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Args:
            cls_pred: 分类预测 (B, C)
            cls_target: 分类标签 (B,)
            reg_pred: 回归预测 (B,) 可选
            reg_target: 回归标签 (B,) 可选
            
        Returns:
            losses: {'total', 'cls', 'reg'}
        """
        # 分类损失
        cls_loss = self.cls_loss_fn(cls_pred, cls_target)
        
        # 回归损失
        if reg_pred is not None and reg_target is not None:
            reg_loss = self.reg_loss_fn(reg_pred, reg_target)
            total_loss = self.alpha * cls_loss + (1 - self.alpha) * reg_loss
        else:
            reg_loss = torch.tensor(0.0, device=cls_pred.device)
            total_loss = cls_loss
            
        # # 总损失
        # total_loss = self.alpha * cls_loss + (1 - self.alpha) * reg_loss
        
        return {
            'total': total_loss,
            'cls': cls_loss,
            'reg': reg_loss
        }


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss.
    
    对于类别 c，软标签为:
    y_c = (1 - eps) if c == target else eps / (num_classes - 1)
    """
    
    def __init__(self, eps: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = inputs.size(-1)
        
        # 转换为 one-hot
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # 平滑标签
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.eps / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1 - self.eps)
            
        # KL Divergence
        loss = (-true_dist * log_probs).sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# =============================================================================
# 对比学习损失函数
# =============================================================================

class CrossModalContrastiveLoss(nn.Module):
    """
    跨模态对比学习损失
    
    核心思想：
    - 正样本：同一时刻的 (LOB, Trade) 对
    - 负样本：不同时刻的 (LOB, Trade) 对
    
    让模型学习到：同一市场状态下，LOB和Trade的表征应该相似。
    
    Args:
        temperature: 温度参数，控制分布的锐度
        reduction: 'mean' | 'sum' | 'none'
    """
    
    def __init__(self, temperature: float = 0.07, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        
    def forward(
        self, 
        lob_feat: torch.Tensor,    # (B, T, D)
        trade_feat: torch.Tensor,  # (B, T, D)
        event_mask: Optional[torch.Tensor] = None  # (B, T) 可选，只在有交易的位置计算
    ) -> torch.Tensor:
        """
        计算InfoNCE损失
        
        对于每个时间步t：
        - 正样本: (lob[t], trade[t])
        - 负样本: (lob[t], trade[t']) where t' != t
        
        Args:
            lob_feat: LOB特征 (B, T, D)
            trade_feat: Trade特征 (B, T, D)
            event_mask: 事件掩码 (B, T), True表示有交易
            
        Returns:
            loss: 对比学习损失
        """
        B, T, D = lob_feat.shape
        
        # 如果指定了event_mask，只在有交易的位置计算损失
        if event_mask is not None:
            # 找出有交易的位置
            valid_positions = event_mask.nonzero(as_tuple=True)
            if len(valid_positions[0]) == 0:
                return torch.tensor(0.0, device=lob_feat.device, requires_grad=True)
            
            lob_feat_flat = lob_feat[valid_positions]    # (N_valid, D)
            trade_feat_flat = trade_feat[valid_positions] # (N_valid, D)
        else:
            # Reshape to (B*T, D)
            lob_feat_flat = lob_feat.reshape(-1, D)
            trade_feat_flat = trade_feat.reshape(-1, D)
        
        N = lob_feat_flat.size(0)
        
        if N < 2:
            return torch.tensor(0.0, device=lob_feat.device, requires_grad=True)
        
        # L2 normalize
        lob_feat_flat = F.normalize(lob_feat_flat, dim=-1)
        trade_feat_flat = F.normalize(trade_feat_flat, dim=-1)
        
        # 计算相似度矩阵
        sim_matrix = torch.mm(lob_feat_flat, trade_feat_flat.t()) / self.temperature  # (N, N)
        
        # 正样本在对角线上
        labels = torch.arange(N, device=lob_feat.device)
        
        # InfoNCE Loss (双向)
        loss_lob2trade = F.cross_entropy(sim_matrix, labels, reduction=self.reduction)
        loss_trade2lob = F.cross_entropy(sim_matrix.t(), labels, reduction=self.reduction)
        
        loss = (loss_lob2trade + loss_trade2lob) / 2
        
        return loss


class TemporalContrastiveLoss(nn.Module):
    """
    时序对比学习损失
    
    让相邻时间步的表征相似，远离时间步的表征不同。
    这可以帮助模型学习平滑的时序表征。
    
    Args:
        temperature: 温度参数
        positive_radius: 多少个相邻时间步算正样本
        margin: 对比损失的margin
    """
    
    def __init__(
        self, 
        temperature: float = 0.1, 
        positive_radius: int = 5,
        margin: float = 0.5
    ):
        super().__init__()
        self.temperature = temperature
        self.positive_radius = positive_radius
        self.margin = margin
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, T, D)
            
        Returns:
            loss: 时序对比损失
        """
        B, T, D = features.shape
        
        if T < 2 * self.positive_radius + 1:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        # L2 normalize
        features = F.normalize(features, dim=-1)
        
        # 计算时间步间的相似度
        # features: (B, T, D) -> 对每个batch计算 T x T 相似度矩阵
        sim_matrix = torch.bmm(features, features.transpose(1, 2)) / self.temperature  # (B, T, T)
        
        # 创建正样本mask：相邻positive_radius个时间步
        pos_mask = torch.zeros(T, T, device=features.device)
        for i in range(T):
            start = max(0, i - self.positive_radius)
            end = min(T, i + self.positive_radius + 1)
            pos_mask[i, start:end] = 1
        pos_mask.fill_diagonal_(0)  # 排除自己
        
        # 负样本mask
        neg_mask = 1 - pos_mask
        neg_mask.fill_diagonal_(0)
        
        # 确保有正负样本
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        # 对于每个anchor，最大化正样本相似度，最小化负样本相似度
        pos_mask = pos_mask.unsqueeze(0)  # (1, T, T)
        neg_mask = neg_mask.unsqueeze(0)  # (1, T, T)
        
        # 计算正负样本的平均相似度
        pos_count = pos_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        neg_count = neg_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        
        pos_sim = (sim_matrix * pos_mask).sum(dim=-1) / pos_count.squeeze(-1)  # (B, T)
        neg_sim = (sim_matrix * neg_mask).sum(dim=-1) / neg_count.squeeze(-1)  # (B, T)
        
        # Margin loss: 希望 pos_sim > neg_sim + margin
        loss = F.relu(neg_sim - pos_sim + self.margin).mean()
        
        return loss


class MultiModalTrainingLoss(nn.Module):
    """
    多模态训练的组合损失
    
    包含：
    1. 主任务损失 (分类/回归)
    2. 跨模态对比损失 (可选)
    3. 时序平滑损失 (可选)
    
    Args:
        num_classes: 分类类别数
        cls_weight: 分类损失权重
        reg_weight: 回归损失权重
        contrastive_weight: 跨模态对比损失权重
        temporal_weight: 时序对比损失权重
        label_smoothing: 标签平滑系数
        focal_gamma: Focal Loss的gamma参数
        use_focal: 是否使用Focal Loss
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        cls_weight: float = 1.0,
        reg_weight: float = 0.0,
        contrastive_weight: float = 0.1,
        temporal_weight: float = 0.05,
        label_smoothing: float = 0.1,
        focal_gamma: float = 2.0,
        use_focal: bool = True
    ):
        super().__init__()
        
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.contrastive_weight = contrastive_weight
        self.temporal_weight = temporal_weight
        
        # 分类损失
        if use_focal:
            self.cls_loss = FocalLoss(gamma=focal_gamma, label_smoothing=label_smoothing)
        else:
            self.cls_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # 回归损失
        self.reg_loss = nn.HuberLoss()
        
        # 对比损失
        self.contrastive_loss = CrossModalContrastiveLoss()
        self.temporal_loss = TemporalContrastiveLoss()
        
    def forward(
        self,
        logits: torch.Tensor,              # (B, num_classes)
        labels: torch.Tensor,              # (B,)
        regression: Optional[torch.Tensor] = None,  # (B,)
        reg_target: Optional[torch.Tensor] = None,  # (B,)
        lob_feat: Optional[torch.Tensor] = None,    # (B, T, D)
        trade_feat: Optional[torch.Tensor] = None,  # (B, T, D)
        fused_feat: Optional[torch.Tensor] = None,  # (B, T, D)
        event_mask: Optional[torch.Tensor] = None,  # (B, T)
    ) -> dict:
        """
        计算组合损失
        
        Args:
            logits: 分类logits
            labels: 分类标签
            regression: 回归预测 (可选)
            reg_target: 回归目标 (可选)
            lob_feat: LOB特征 (可选，用于对比学习)
            trade_feat: Trade特征 (可选，用于对比学习)
            fused_feat: 融合特征 (可选，用于时序对比)
            event_mask: 事件掩码 (可选)
            
        Returns:
            losses: 损失字典
        """
        losses = {}
        
        # 1. 分类损失
        cls_loss = self.cls_loss(logits, labels)
        losses['cls'] = cls_loss
        total_loss = self.cls_weight * cls_loss
        
        # 2. 回归损失
        if regression is not None and reg_target is not None and self.reg_weight > 0:
            reg_loss = self.reg_loss(regression, reg_target)
            losses['reg'] = reg_loss
            total_loss = total_loss + self.reg_weight * reg_loss
        
        # 3. 跨模态对比损失
        if (lob_feat is not None and trade_feat is not None 
            and self.contrastive_weight > 0):
            contrastive_loss = self.contrastive_loss(lob_feat, trade_feat, event_mask)
            losses['contrastive'] = contrastive_loss
            total_loss = total_loss + self.contrastive_weight * contrastive_loss
        
        # 4. 时序平滑损失
        if fused_feat is not None and self.temporal_weight > 0:
            temporal_loss = self.temporal_loss(fused_feat)
            losses['temporal'] = temporal_loss
            total_loss = total_loss + self.temporal_weight * temporal_loss
        
        losses['total'] = total_loss
        
        return losses



class MultiTaskReturnLoss(nn.Module):
    """
    多任务return预测损失函数（5个时间窗口：10/30/60/300/600s）
    鲁棒Huber损失 + 自定义任务权重
    """
    def __init__(
        self,
        # loss_name: str = "mutilmask_loss",
        task_weights: list = None,  # 5个任务的权重，默认等权
        delta: float = 1.0          # Huber损失阈值，推荐1.0
    ):
        super().__init__()
        # 默认等权重：5个任务同等重要

        self.task_weights = torch.tensor(task_weights) if task_weights else torch.ones(5)
        self.delta = delta

    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        :param pred: 模型输出 [B, 5]
        :param labels: 真实标签 [B, 5]
        :return: 标量总损失
        """
        # 校验维度
        assert pred.shape == labels.shape, f"预测值{pred.shape}和标签{labels.shape}维度不匹配"
        # assert pred.size(-1) == 5, "必须输出5个任务的预测结果"

        # 1. 把权重放到和数据相同的设备（GPU/CPU）
        weights = self.task_weights.to(pred.device)

        # 2. 计算【每个任务】的Huber损失 (B,5)
        huber_loss = F.huber_loss(pred, labels, reduction="none", delta=self.delta)

        # 3. 按样本平均 → (5,) → 加权求和 → 总损失
        per_task_loss = huber_loss.mean(dim=0)  # 每个任务的平均损失
        total_loss = (per_task_loss * weights).sum() / weights.sum()  # 加权平均

        return total_loss

class AdaptiveMultiTaskLoss(nn.Module):
    """基于不确定性的自适应多任务损失（自动加权）"""
    def __init__(self, num_tasks: int = 5):
        super().__init__()
        self.num_tasks = num_tasks
        # 可学习的 log方差，用于自适应权重
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, pred, labels):
        assert pred.shape == labels.shape and pred.size(-1) == 5
        losses = F.huber_loss(pred, labels, reduction="none", delta=1.0).mean(dim=0)  # (5,)
        
        # 自适应权重公式
        total_loss = 0
        for i in range(self.num_tasks):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * losses[i] + self.log_vars[i]
        
        return total_loss / self.num_tasks