#!/usr/bin/env python
# =============================================================================
# Ablation Experiment Runner - 消融实验运行脚本
# Version: v1.0
#
# 运行多模态LOB-Trade融合模型的消融实验
# =============================================================================

import os
import sys
import json
import yaml
import argparse
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# 添加项目路径
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Model import MultiModalTransformerV2
from Utils import (
    aggregate_metrics,
    MultiModalTrainingLoss,
    compute_metrics,
    get_confusion_matrix,
)
from Data_Pipeline.dataset import create_dataloaders
from Train.trainer import Trainer

from torch.amp import autocast

# 优化器
from Train.trainer import Trainer
from Train.train_utils import (
    setup_optimizer,
    setup_scheduler,
    setup_loss_functions,
    set_seed
)

set_seed(42)

class AblationExperimentRunner:
    """
    消融实验运行器
    
    负责：
    1. 加载配置
    2. 运行多个模型变体
    3. 收集和聚合结果
    4. 生成对比报告
    """
    
    def __init__(self, config_path: str, output_dir: Optional[str] = None):
        """
        初始化实验运行器
        
        Args:
            config_path: 实验配置文件路径
            output_dir: 输出目录 (可选，会覆盖配置中的路径)
        """
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 设置输出目录
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = self.config.get('output', {}).get('results_dir', 'experiments/results')
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # # 初始化指标计算器
        # self.metrics_calculator = ExperimentMetrics()
        
        # 存储结果
        self.all_results = {}
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def _set_seed(self, seed: int):
        """设置随机种子以保证可复现性。"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _create_model(self,variant_config: dict) -> nn.Module:
        """
        创建模型
        
        Args:
            variant_name: 变体名称
            variant_config: 变体配置
            
        Returns:
            model: 模型实例
        """
        base_model_config = self.config.get('base_model', {})
        
        model = MultiModalTransformerV2(
            lob_config=base_model_config.get('lob_encoder', {}),
            trade_config=base_model_config.get('trade_encoder'),
            fusion_config=base_model_config.get('fusion', {}),
            backbone_config=base_model_config.get('transformer', {}),
            output_config=base_model_config.get('output_head', {}),
            use_cross_features=variant_config.get('use_cross_features', False),
            use_early_cross_attention=variant_config.get('use_early_cross_attention', False),
            use_hierarchical=variant_config.get('use_hierarchical', False),
            use_event_driven_trade=variant_config.get('use_event_driven_trade', False),
            use_mamba=variant_config.get('use_mamba', False),
        )
        # from Model import MultiModalTransformer
        # # 根据数据情况调整配置
        # lob_config = base_model_config.get('lob_encoder', {})
        # trade_config = base_model_config.get('trade_encoder') 
        # fusion_config = base_model_config.get('fusion', {})
        # transformer_config = base_model_config.get('transformer', {})
        # output_config = base_model_config.get('output_head', {})

        # model = MultiModalTransformer(
        #     lob_config=lob_config,
        #     trade_config=trade_config,
        #     fusion_config=fusion_config,
        #     transformer_config=transformer_config,
        #     output_config=output_config,
        #     use_revin=True
        # )
        
        return model.to(self.device)
    
    def _create_loss_fn(self, variant_config: dict) -> nn.Module:
        """创建损失函数"""
        loss_config = self.config.get('loss', {})
        
        use_contrastive = variant_config.get('use_contrastive_loss', False)
        
        if use_contrastive:
            return MultiModalTrainingLoss(
                num_classes=3,
                cls_weight=loss_config.get('cls_weight', 1.0),
                reg_weight=loss_config.get('reg_weight', 0.0),
                contrastive_weight=loss_config.get('contrastive_weight', 0.1),
                temporal_weight=loss_config.get('temporal_weight', 0.05),
                label_smoothing=loss_config.get('label_smoothing', 0.1),
                focal_gamma=loss_config.get('focal_gamma', 2.0),
                use_focal=loss_config.get('use_focal', True)
            )
        else:
            # variant_config['use_focal'] = False
            # org_loss_config  = self.config.get('org_loss', {})
            # device = 'cuda'
            # 设置损失函数
            loss_fn = setup_loss_functions(loss_config, device=self.device)
            # return FocalLoss(
            #     gamma=loss_config.get('focal_gamma', 2.0),
            #     alpha=loss_config.get('focal_alpha', 1.0),
            #     class_weights=loss_config.get('class_weights', None),
            #     # label_smoothing=loss_config.get('label_smoothing', 0.1)
            # )
            return loss_fn
    
    def _train_and_evaluate(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        loss_fn: nn.Module,
        optimizer,
        scheduler,
        variant_name = None,
        seed = None
    ) -> Dict:
        """
        训练并评估模型
        
        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            loss_fn: 损失函数
            variant_config: 变体配置
            seed: 随机种子
            
        Returns:
            result: 包含指标和训练历史的字典
        """

        
        # 创建训练器
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            config=self.config,
            device=self.device,
            variant_name=variant_name,
            seed=seed
        )
        # 开始训练
        print("=" * 50)
        history = trainer.fit()

        print("=" * 50)
        print("训练完成!")
        # print(f"最佳验证 F1 (Up/Down): {max(history['val_f1_updown']):.4f}")
        return history
    
    def _evaluate(self, model: nn.Module, data_loader) -> Dict:
        """评估模型"""
        model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        pbar = tqdm(data_loader, desc='Validation', leave=False)
        with torch.no_grad():
            for inputs, labels, returns in pbar:
                # inputs, labels, _ = batch
                # inputs = {k: v.to(self.device) for k, v in inputs.items()}
                            # 移动数据
                if isinstance(inputs, dict):
                    inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
                else:
                    inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                returns = returns.to(self.device, non_blocking=True)
                            # 前向传播
                with autocast(device_type='cuda', enabled=True):
                    logits, _ = model(inputs)
                    # losses = self.loss_fn(cls_pred, labels, reg_pred, returns)
                # logits, _ = model(inputs)
                # probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                # all_probs.append(probs.cpu().numpy())
        
        # all_preds = np.concatenate(all_preds)
        # all_labels = np.concatenate(all_labels)
        # all_probs = np.concatenate(all_probs)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        # all_probs = np.array(all_probs)
        metrics = compute_metrics(all_labels, all_preds)
        # metrics['loss'] = avg_loss
        # metrics.update(updown_metrics)
        metrics['confusion_matrix'] = get_confusion_matrix(all_labels, all_preds,normalize=False)
        
        return metrics

        # return self.metrics_calculator.compute_all_metrics(all_labels, all_preds, all_probs)
    
    def run_single_experiment(
        self,
        variant_name: str,
        variant_config: dict,
        data_dict: Dict[str, np.ndarray],
        labels: np.ndarray,
        returns: np.ndarray,
        seed: int
    ) -> Dict:
        """
        运行单次实验
        
        Args:
            variant_name: 变体名称
            variant_config: 变体配置
            data_dict: 数据字典
            labels: 标签
            returns: 收益率
            seed: 随机种子
            
        Returns:
            result: 实验结果
        """
        print(f"\n  Running {variant_name} with seed {seed}...")
        
        # 设置随机种子
        set_seed(seed)
        
        # 创建数据加载器
        data_config = self.config.get('data', {})
        dataloader_config = self.config.get('dataloader', {})
        training_config = self.config.get('training', {})

        train_loader, val_loader = create_dataloaders(
            data_dict=data_dict,
            labels=labels,
            returns=returns,
            config={'data': data_config, 'dataloader': dataloader_config},
            device=dataloader_config['device'] # 数据在CPU，训练时搬运
        )
        
        # 创建模型
        model = self._create_model(variant_config)
        # variant_name = variant_config['variant_name']
        # 创建损失函数
        loss_fn = self._create_loss_fn(variant_config)
        # print(f"loss_fn: {loss_fn}")
        # 优化器
        optimizer = setup_optimizer(model,training_config.get('optimizer', {}))
        # 学习率调度器
        scheduler = setup_scheduler(optimizer,training_config.get('scheduler', {}))
        # 训练和评估
        result = self._train_and_evaluate(
            model, 
            train_loader, 
            val_loader, 
            loss_fn,
            optimizer,
            scheduler,
            variant_name = variant_name,
            seed = seed
        )
        
        # # 计算效率指标
        # sample_input = next(iter(val_loader))[0]
        # sample_input = {k: v.to(self.device) for k, v in sample_input.items()}
        # efficiency_metrics = self.metrics_calculator.compute_efficiency_metrics(
        #     model, sample_input, num_warmup=5, num_runs=50
        # )
        # result['efficiency_metrics'] = efficiency_metrics
        
        return result
    
    def run_ablation_study(
        self,
        data_dict: Dict[str, np.ndarray],
        labels: np.ndarray,
        returns: np.ndarray,
        variants: Optional[List[str]] = None
    ) -> Dict:
        """
        运行消融实验
        
        Args:
            data_dict: 数据字典
            labels: 标签
            returns: 收益率
            variants: 要测试的变体列表 (None表示全部)
            
        Returns:
            all_results: 所有实验结果
        """
        seeds = self.config.get('seeds', [42])
        model_variants = self.config.get('model_variants', {})
        
        if variants is None:
            variants = list(model_variants.keys())
        
        for variant_name in variants:
            if variant_name not in model_variants:
                print(f"Warning: {variant_name} not found in config, skipping...")
                continue
            
            print(f"\n{'='*60}")
            print(f"Running experiments for: {variant_name}")
            print(f"{'='*60}")
            
            variant_config = model_variants[variant_name]
            seed_results = []
            
            for seed in seeds:
                # try:
                result = self.run_single_experiment(
                    variant_name, variant_config, data_dict, labels, returns, seed
                )
                print(f"confusion_matrix: {result['best_val_metrics']['confusion_matrix']}")
                seed_results.append(result)
                # except Exception as e:
                #     print(f"Error running {variant_name} with seed {seed}: {e}")
                #     continue
            
            if seed_results:
                # 聚合结果
                aggregated = self._aggregate_results(seed_results)
                self.all_results[variant_name] = aggregated
                
                # 保存中间结果
                self._save_results(f'ablation_results_partial.json')
        
        # 保存最终结果
        self._save_results('ablation_results_final.json')
        
        # 生成报告
        self._generate_report(variants)
        
        return self.all_results
    
    def _aggregate_results(self, seed_results: List[Dict]) -> Dict:
        """聚合多次运行结果"""
        if not seed_results:
            return {}
        
        # 收集指标
        val_metrics_list = [r['best_val_metrics'] for r in seed_results if 'best_val_metrics' in r]
        
        aggregated = {
            'val_metrics': aggregate_metrics(val_metrics_list),
            # 'efficiency_metrics': seed_results[0].get('efficiency_metrics', {}),
            'best_epochs': [r.get('best_epoch', 0) for r in seed_results],
        }
        
        return aggregated
    
    def _save_results(self, filename: str):
        """保存结果"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.all_results, f, indent=2, default=str)
        print(f"Results saved to: {filepath}")
    
    def _generate_report(self, variants: List[str]):
        """生成实验报告"""
        report_lines = []
        report_lines.append("# 多模态LOB-Trade模型消融实验报告")
        report_lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        report_lines.append(f"\n## 实验配置")
        report_lines.append(f"- Seeds: {self.config.get('seeds', [42])}")
        report_lines.append(f"- Epochs: {self.config.get('training', {}).get('epochs', 100)}")
        report_lines.append(f"- Batch Size: {self.config.get('dataloader', {}).get('batch_size', 256)}")
        
        # 主要指标对比表
        report_lines.append("\n## 主要指标对比\n")
        report_lines.append("| Model | Accuracy | F1-Macro | **F1-UpDown** | F1-Up | F1-Down | Dir-Acc | Signal-Q | recall_updown | precision_updown |")
        report_lines.append("|-------|----------|----------|---------------|-------|---------|---------|----------|--------------|----------------|")
        
        for variant in variants:
            if variant not in self.all_results:
                continue
            m = self.all_results[variant].get('val_metrics', {})
            
            def get_val(key):
                v = m.get(key, {})
                if isinstance(v, dict):
                    return f"{v.get('mean', 0):.4f}±{v.get('std', 0):.3f}"
                return f"{v:.4f}"
            
            report_lines.append(
                f"| {variant} | "
                f"{get_val('accuracy')} | "
                f"{get_val('f1_macro')} | "
                f"**{get_val('f1_updown')}** | "
                f"{get_val('f1_up')} | "
                f"{get_val('f1_down')} | "
                f"{get_val('directional_accuracy')} | "
                f"{get_val('signal_quality')} |"
                f"{get_val('precision_updown')} |"
                f"{get_val('recall_updown')} |"
            )
        
        # # 效率指标
        # report_lines.append("\n## 效率指标\n")
        # report_lines.append("| Model | Params (M) | Inference (ms) | GPU Mem (MB) |")
        # report_lines.append("|-------|------------|----------------|--------------|")
        
        # for variant in variants:
        #     if variant not in self.all_results:
        #         continue
        #     e = self.all_results[variant].get('efficiency_metrics', {})
        #     report_lines.append(
        #         f"| {variant} | "
        #         f"{e.get('total_params_M', 0):.2f} | "
        #         f"{e.get('inference_time_ms', 0):.2f} | "
        #         f"{e.get('gpu_memory_MB', 0):.1f} |"
        #     )
        
        # # 统计显著性
        # if 'M0_baseline' in self.all_results:
        #     report_lines.append("\n## 统计显著性检验 (vs Baseline)\n")
        #     report_lines.append("| Model | F1-UpDown Δ | p-value | Significant? |")
        #     report_lines.append("|-------|-------------|---------|--------------|")
            
        #     baseline_values = self.all_results['M0_baseline'].get('val_metrics', {}).get('f1_updown', {}).get('values', [])
            
        #     for variant in variants[1:]:
        #         if variant not in self.all_results:
        #             continue
        #         model_values = self.all_results[variant].get('val_metrics', {}).get('f1_updown', {}).get('values', [])
                
        #         if baseline_values and model_values:
        #             delta, p_value, is_sig = compute_significance_test(baseline_values, model_values)
        #             sig_mark = "✓" if is_sig else "✗"
        #             report_lines.append(f"| {variant} | {delta:+.4f} | {p_value:.4f} | {sig_mark} |")
        
        # 保存报告
        report_path = os.path.join(self.output_dir, 'experiment_report.md')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\n报告已保存至: {report_path}")




## 读取
lob_data = np.load('/root/autodl-tmp/train_data/lob_data.npy')
trade_data = np.load('/root/autodl-tmp/train_data/trade_data.npy')
labels_ret = np.load('/root/autodl-tmp/train_data/labels_ret.npy')


print(f"trade_data_agg.shape: {trade_data.shape}")
print(f"lob_data.shape: {lob_data.shape}")
print(f"labels_ret.shape: {labels_ret.shape}")

alpha= 0.001
labels_class = np.ones_like(labels_ret, dtype=np.int8)
# 3. 向量化赋值：涨→2，跌→0
# 涨：labels_ret > alpha
labels_class[labels_ret > alpha] = 2
# 跌：labels_ret < -alpha
labels_class[labels_ret < -alpha] = 0

data_dict = {
    'lob': lob_data,
    'trade': trade_data
}

labels = labels_class
returns = labels_ret
variants = ['M0_baseline','M1_cross_features']

# 运行实验
config = '/root/lio/Trade_LOB_MultiModal/Configs/experiment_config.yaml'
output_dir = '/root/lio/Trade_LOB_MultiModal/experiments/ablation_results'
runner = AblationExperimentRunner(config, output_dir)

runner.run_ablation_study(data_dict, labels, returns, variants)


# def main():
#     """主函数"""
#     parser = argparse.ArgumentParser(description='Run ablation experiments')
#     parser.add_argument('--config', type=str, default='Configs/experiment_config.yaml',
#                         help='Path to experiment config file')
#     parser.add_argument('--output_dir', type=str, default=None,
#                         help='Output directory for results')
#     parser.add_argument('--variants', type=str, nargs='+', default=None,
#                         help='Model variants to run (default: all)')
#     parser.add_argument('--lob_data', type=str, required=True,
#                         help='Path to LOB data file (.npy)')
#     parser.add_argument('--trade_data', type=str, default=None,
#                         help='Path to Trade data file (.npy)')
#     parser.add_argument('--labels', type=str, required=True,
#                         help='Path to labels file (.npy)')
#     parser.add_argument('--returns', type=str, default=None,
#                         help='Path to returns file (.npy)')
    
#     args = parser.parse_args()
    
#     # 加载数据
#     print("Loading data...")
#     lob_data = np.load(args.lob_data)
#     labels = np.load(args.labels)
    
#     data_dict = {'lob': lob_data}
    
#     if args.trade_data:
#         trade_data = np.load(args.trade_data)
#         data_dict['trade'] = trade_data
    
#     if args.returns:
#         returns = np.load(args.returns)
#     else:
#         returns = np.zeros_like(labels, dtype=np.float32)
    
#     print(f"LOB data shape: {lob_data.shape}")
#     if 'trade' in data_dict:
#         print(f"Trade data shape: {data_dict['trade'].shape}")
#     print(f"Labels shape: {labels.shape}")
    
#     # 运行实验
#     runner = AblationExperimentRunner(args.config, args.output_dir)
#     results = runner.run_ablation_study(data_dict, labels, returns, args.variants)
    
#     print("\n实验完成！")
#     print(f"结果保存在: {runner.output_dir}")


# if __name__ == '__main__':
#     main()
