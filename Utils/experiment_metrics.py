# =============================================================================
# Experiment Metrics - 实验评估指标模块
# Version: v1.0
#
# 为消融实验提供全面的评估指标计算
# =============================================================================

import numpy as np
import torch
import time
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class MetricResult:
    """单次评估的指标结果"""
    accuracy: float = 0.0
    f1_macro: float = 0.0
    f1_weighted: float = 0.0
    f1_updown: float = 0.0
    
    # 分类别指标
    precision_down: float = 0.0
    recall_down: float = 0.0
    f1_down: float = 0.0
    
    precision_stationary: float = 0.0
    recall_stationary: float = 0.0
    f1_stationary: float = 0.0
    
    precision_up: float = 0.0
    recall_up: float = 0.0
    f1_up: float = 0.0
    
    # 方向性指标
    directional_accuracy: float = 0.0
    signal_quality: float = 0.0
    signal_count: int = 0
    
    # 校准指标
    ece: float = 0.0
    confidence_mean: float = 0.0
    
    # 混淆矩阵
    confusion_matrix: List = field(default_factory=list)


class ExperimentMetrics:
    """
    实验评估指标计算器
    
    提供全面的分类评估指标，特别关注：
    1. Up/Down类别的F1 (避免Stationary类陷阱)
    2. 方向性准确率 (实际交易价值)
    3. 信号质量 (预测信号的可靠性)
    """
    
    def __init__(self, num_classes: int = 3):
        self.num_classes = num_classes
        # 类别映射: 0=Down, 1=Stationary, 2=Up
        self.class_names = ['Down', 'Stationary', 'Up']
        
    def compute_all_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        计算所有评估指标
        
        Args:
            y_true: 真实标签 (N,)
            y_pred: 预测标签 (N,)
            y_prob: 预测概率 (N, 3) 可选
            
        Returns:
            metrics: 指标字典
        """
        metrics = {}
        
        # 确保是numpy数组
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        if y_prob is not None and isinstance(y_prob, torch.Tensor):
            y_prob = y_prob.cpu().numpy()
        
        # ============================================================
        # 1. 基础分类指标
        # ============================================================
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # ============================================================
        # 2. 分类别指标
        # ============================================================
        precision_per_class = precision_score(
            y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2]
        )
        recall_per_class = recall_score(
            y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2]
        )
        f1_per_class = f1_score(
            y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2]
        )
        
        for i, name in enumerate(self.class_names):
            if i < len(precision_per_class):
                metrics[f'precision_{name.lower()}'] = precision_per_class[i]
                metrics[f'recall_{name.lower()}'] = recall_per_class[i]
                metrics[f'f1_{name.lower()}'] = f1_per_class[i]
            else:
                metrics[f'precision_{name.lower()}'] = 0.0
                metrics[f'recall_{name.lower()}'] = 0.0
                metrics[f'f1_{name.lower()}'] = 0.0
        
        # ============================================================
        # 3. 核心指标: Up/Down F1 (排除Stationary)
        # ============================================================
        f1_down = metrics.get('f1_down', 0.0)
        f1_up = metrics.get('f1_up', 0.0)
        metrics['f1_updown'] = (f1_down + f1_up) / 2
        
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
            
        # ============================================================
        # 6. 混淆矩阵
        # ============================================================
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        metrics['confusion_matrix'] = cm.tolist()
        
        # ============================================================
        # 7. 校准指标 (如果有概率输出)
        # ============================================================
        if y_prob is not None:
            metrics['ece'] = self._compute_ece(y_true, y_prob)
            metrics['confidence_mean'] = float(y_prob.max(axis=1).mean())
            
            # AUC (如果可计算)
            try:
                if len(np.unique(y_true)) > 1:
                    metrics['auc_macro'] = roc_auc_score(
                        y_true, y_prob, multi_class='ovr', average='macro'
                    )
            except:
                pass
            
        return metrics
    
    def _compute_ece(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray, 
        n_bins: int = 10
    ) -> float:
        """
        计算期望校准误差 (Expected Calibration Error)
        
        ECE衡量模型置信度与实际准确率的一致性。
        低ECE意味着模型对自己的预测有良好的"自知之明"。
        """
        confidences = y_prob.max(axis=1)
        predictions = y_prob.argmax(axis=1)
        accuracies = (predictions == y_true).astype(float)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = accuracies[in_bin].mean()
                ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
                
        return float(ece)
    
    def compute_efficiency_metrics(
        self,
        model: torch.nn.Module,
        sample_input: Dict[str, torch.Tensor],
        num_warmup: int = 10,
        num_runs: int = 100
    ) -> Dict[str, float]:
        """
        计算效率指标
        
        Args:
            model: PyTorch模型
            sample_input: 样本输入
            num_warmup: 预热运行次数
            num_runs: 正式运行次数
            
        Returns:
            efficiency_metrics: 效率指标字典
        """
        metrics = {}
        device = next(model.parameters()).device
        
        # 参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        metrics['total_params_M'] = total_params / 1e6
        metrics['trainable_params_M'] = trainable_params / 1e6
        
        # 推理时间
        model.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(num_warmup):
                _ = model(sample_input)
            
            # Measure
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            for _ in range(num_runs):
                _ = model(sample_input)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            
        batch_size = sample_input['lob'].size(0)
        total_samples = batch_size * num_runs
        metrics['inference_time_ms'] = (end_time - start_time) * 1000 / total_samples
        metrics['throughput_samples_per_sec'] = total_samples / (end_time - start_time)
        
        # GPU显存
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = model(sample_input)
            metrics['gpu_memory_MB'] = torch.cuda.max_memory_allocated() / 1e6
        
        return metrics
    
    def to_result_object(self, metrics: Dict[str, Any]) -> MetricResult:
        """将字典转换为MetricResult对象"""
        result = MetricResult()
        for key, value in metrics.items():
            if hasattr(result, key):
                setattr(result, key, value)
        return result


def aggregate_metrics(
    metrics_list: List[Dict[str, Any]], 
    keys: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    聚合多次运行的指标 (计算均值和标准差)
    
    Args:
        metrics_list: 多次运行的指标列表
        keys: 要聚合的指标键 (None表示所有数值指标)
        
    Returns:
        aggregated: {metric_name: {'mean': x, 'std': y, 'values': [...]}}
    """
    if not metrics_list:
        return {}
    
    # 确定要聚合的键
    if keys is None:
        keys = []
        for key, value in metrics_list[0].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                keys.append(key)
    
    aggregated = {}
    for key in keys:
        values = [m.get(key, 0) for m in metrics_list if key in m]
        if values:
            aggregated[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'values': values
            }
    
    return aggregated


def format_metrics_table(
    results_dict: Dict[str, Dict[str, Any]], 
    model_names: List[str],
    metrics: Optional[List[str]] = None
) -> str:
    """
    格式化输出对比表格
    
    Args:
        results_dict: {model_name: aggregated_metrics}
        model_names: 模型名称列表 (用于排序)
        metrics: 要显示的指标列表
        
    Returns:
        formatted_table: 格式化的表格字符串
    """
    if metrics is None:
        metrics = [
            'accuracy', 'f1_macro', 'f1_updown', 
            'f1_up', 'f1_down', 
            'directional_accuracy', 'signal_quality'
        ]
    
    lines = []
    lines.append("=" * 100)
    
    # 表头
    header = f"{'Metric':<25} | "
    header += " | ".join([f"{m:<12}" for m in model_names])
    lines.append(header)
    lines.append("-" * 100)
    
    # 数据行
    for metric in metrics:
        row = f"{metric:<25} | "
        values = []
        best_val = -1
        best_idx = -1
        
        for i, model in enumerate(model_names):
            model_metrics = results_dict.get(model, {})
            if isinstance(model_metrics, dict) and metric in model_metrics:
                metric_data = model_metrics[metric]
                if isinstance(metric_data, dict):
                    val = metric_data.get('mean', 0)
                    std = metric_data.get('std', 0)
                    val_str = f"{val:.4f}±{std:.3f}"
                else:
                    val = metric_data
                    val_str = f"{val:.4f}"
            else:
                val = 0
                val_str = "N/A"
            
            values.append((val, val_str))
            if val > best_val:
                best_val = val
                best_idx = i
        
        # 标记最优值
        formatted_values = []
        for i, (val, val_str) in enumerate(values):
            if i == best_idx and val > 0:
                formatted_values.append(f"**{val_str}**")
            else:
                formatted_values.append(val_str)
        
        row += " | ".join([f"{v:<12}" for v in formatted_values])
        lines.append(row)
    
    lines.append("=" * 100)
    return "\n".join(lines)


def compute_significance_test(
    baseline_values: List[float],
    comparison_values: List[float],
    test_type: str = 'paired_t'
) -> Tuple[float, float, bool]:
    """
    计算统计显著性检验
    
    Args:
        baseline_values: 基线模型的指标值列表
        comparison_values: 对比模型的指标值列表
        test_type: 检验类型 ('paired_t', 'wilcoxon')
        
    Returns:
        delta: 差异值 (mean(comparison) - mean(baseline))
        p_value: p值
        is_significant: 是否显著 (p < 0.05)
    """
    from scipy import stats
    
    baseline_values = np.array(baseline_values)
    comparison_values = np.array(comparison_values)
    
    delta = float(np.mean(comparison_values) - np.mean(baseline_values))
    
    if len(baseline_values) < 2 or len(comparison_values) < 2:
        return delta, 1.0, False
    
    if len(baseline_values) != len(comparison_values):
        # 非配对检验
        t_stat, p_value = stats.ttest_ind(comparison_values, baseline_values)
    else:
        if test_type == 'paired_t':
            t_stat, p_value = stats.ttest_rel(comparison_values, baseline_values)
        elif test_type == 'wilcoxon':
            try:
                stat, p_value = stats.wilcoxon(comparison_values, baseline_values)
            except:
                p_value = 1.0
        else:
            t_stat, p_value = stats.ttest_rel(comparison_values, baseline_values)
    
    is_significant = p_value < 0.05
    
    return delta, float(p_value), is_significant


class ExperimentLogger:
    """
    实验日志记录器
    
    记录实验配置、训练过程和最终结果。
    """
    
    def __init__(self, log_dir: str, experiment_name: str):
        import os
        import json
        from datetime import datetime
        
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.start_time = datetime.now()
        
        # 创建目录
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_file = os.path.join(
            log_dir, 
            f"{experiment_name}_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        self.log_data = {
            'experiment_name': experiment_name,
            'start_time': self.start_time.isoformat(),
            'config': {},
            'training_history': [],
            'final_metrics': {},
            'notes': []
        }
        
    def log_config(self, config: Dict):
        """记录配置"""
        self.log_data['config'] = config
        self._save()
        
    def log_epoch(self, epoch: int, metrics: Dict):
        """记录单个epoch的指标"""
        self.log_data['training_history'].append({
            'epoch': epoch,
            'metrics': metrics
        })
        self._save()
        
    def log_final_metrics(self, metrics: Dict):
        """记录最终指标"""
        self.log_data['final_metrics'] = metrics
        self.log_data['end_time'] = datetime.now().isoformat()
        self._save()
        
    def add_note(self, note: str):
        """添加备注"""
        self.log_data['notes'].append({
            'time': datetime.now().isoformat(),
            'note': note
        })
        self._save()
        
    def _save(self):
        """保存日志"""
        import json
        with open(self.log_file, 'w') as f:
            json.dump(self.log_data, f, indent=2, default=str)


from datetime import datetime
