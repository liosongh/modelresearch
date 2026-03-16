# =============================================================================
# Metrics - 评估指标模块
# Version: v2.0
# =============================================================================

import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)


def compute_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    num_classes: int = 3
) -> Dict[str, float]:
    """
    计算完整的评估指标。
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        num_classes: 类别数
        
    Returns:
        metrics: 指标字典
    """
    metrics = {}
    
    # 1. 全局指标
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # 2. Per-class 指标
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, 
        labels=list(range(num_classes)),
        average=None,
        zero_division=0
    )
    
    class_names = ['Down', 'Stationary', 'Up'] if num_classes == 3 else [f'Class_{i}' for i in range(num_classes)]
    
    for i, name in enumerate(class_names):
        metrics[f'precision_{name}'] = precision[i]
        metrics[f'recall_{name}'] = recall[i]
        metrics[f'f1_{name}'] = f1[i]
        metrics[f'support_{name}'] = support[i]
    
    # 3. 宏平均
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        average='macro',
        zero_division=0
    )
    metrics['precision_macro'] = macro_precision
    metrics['recall_macro'] = macro_recall
    metrics['f1_macro'] = macro_f1
    
    # 4. 加权平均
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        average='weighted',
        zero_division=0
    )
    metrics['precision_weighted'] = weighted_precision
    metrics['recall_weighted'] = weighted_recall
    metrics['f1_weighted'] = weighted_f1
    
    # 5.updown 指标

    # mask = y_true[y_true != 1]
    # y_true_updown = y_true[mask]
    # y_pred_updown = y_pred[mask]
    
    # accuracy_updown = accuracy_score(y_true_updown, y_pred_updown)
    precision_updown = (metrics[f'precision_Up'] + metrics[f'precision_Down'])/2
    recall_updown = (metrics[f'recall_Up'] + metrics[f'recall_Down'])/2
    f1_updown = 2*precision_updown*recall_updown/(precision_updown + recall_updown)
    # metrics['accuracy_updown'] = accuracy_updown
    metrics['precision_updown'] = precision_updown
    metrics['recall_updown'] = recall_updown
    metrics['f1_updown'] = f1_updown

        # ============================================================
    # 4. 方向性准确率
    # ============================================================
    # 只看Up和Down类的预测准确性
    direction_mask = (y_true != 1)  # 排除Stationary
    if direction_mask.sum() > 0:
        metrics['directional_accuracy'] = accuracy_score(
            y_true[direction_mask], 
            y_pred[direction_mask]
        )
    else:
        metrics['directional_accuracy'] = 0.0
    
    # ============================================================
    # 5. 信号质量 (预测Up/Down时的准确率)
    # ============================================================
    signal_mask = (y_pred != 1)  # 预测为Up或Down
    metrics['signal_count'] = int(signal_mask.sum())
    if signal_mask.sum() > 0:
        metrics['signal_quality'] = accuracy_score(
            y_true[signal_mask],
            y_pred[signal_mask]
        )
    else:
        metrics['signal_quality'] = 0.0

    return metrics


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    计算基础分类指标。
    
    Returns:
        metrics: {'accuracy', 'precision', 'recall', 'f1'}
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        average='macro',
        zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def compute_updown_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    stationary_label: int = 1
) -> Dict[str, float]:
    """
    计算排除 Stationary 类后的 Up/Down 指标。
    
    这是高频交易中最重要的指标，因为我们主要关心方向性预测。
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        stationary_label: Stationary 类的标签值
        
    Returns:
        metrics: Up/Down 相关指标
    """
    # 过滤掉 Stationary 样本
    mask = y_true != stationary_label
    y_true_updown = y_true[mask]
    y_pred_updown = y_pred[mask]
    
    if len(y_true_updown) == 0:
        return {
            'accuracy_updown': 0.0,
            'precision_updown': 0.0,
            'recall_updown': 0.0,
            'f1_updown': 0.0,
            'num_updown_samples': 0
        }
    
    accuracy = accuracy_score(y_true_updown, y_pred_updown)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_updown, y_pred_updown,
        average='macro',
        zero_division=0,
        labels=[0, 2]
    )
    
    return {
        'accuracy_updown': accuracy,
        'precision_updown': precision,
        'recall_updown': recall,
        'f1_updown': f1,
        'num_updown_samples': len(y_true_updown)
    }


def get_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """
    获取混淆矩阵。
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        normalize: 是否归一化
        
    Returns:
        cm: 混淆矩阵
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    return cm


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[list] = None
) -> str:
    """
    获取详细的分类报告。
    """
    if target_names is None:
        target_names = ['Down', 'Stationary', 'Up']
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
