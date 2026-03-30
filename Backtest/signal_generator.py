# =============================================================================
# Signal Generator - 模型加载与信号生成
# =============================================================================

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple

# from Model.multi_modal_transformer import MultiModalTransformer
from Utils.config_loader import load_config


class InferenceDataset(Dataset):
    """轻量推理数据集，仅加载 LOB 数据，CPU 上构建。"""

    def __init__(self, lob_data: np.ndarray, history_T: int, stride: int = 1):
        """
        Args:
            lob_data: (N, 40) LOB 数据
            history_T: 历史窗口长度
            stride: 采样步长
        """
        self.lob = torch.as_tensor(lob_data, dtype=torch.float32)  # (N, 2,20)
        if self.lob.dim()  == 2:
            self.lob = self.lob.unsqueeze(1)
        self.lob = self.lob.permute(1, 0, 2).cuda().contiguous() ## (2,N,20)
        self.T = history_T
        self.valid_indices = np.arange(0, len(lob_data) - history_T + 1, stride)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start = self.valid_indices[idx]
        end = start + self.T
        # (T, 2,20) — 匹配模型输入 (B, C=1, T, L=40)
        lob_window = self.lob[:,start:end,:]
        label_idx = end - 1  # 信号对应的时间索引
        return lob_window, label_idx


class SignalGenerator:
    """加载训练好的模型，对测试集进行推理，生成交易信号。"""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_model(self,model_version,model):
        """加载模型结构和权重。"""
        checkpoint_path = f'./checkpoints/{model_version}/seed_42/best_model.pt'
        # model_config_path = f'./checkpoints/{model_version}/model_config.yaml'

        # model = MultiModalTransformer.from_config(model_config_path)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        print(f"模型加载完成: {checkpoint_path}")
        if 'metrics' in checkpoint:
            print(f"  验证指标: {checkpoint['metrics']}")
        return model

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """加载测试集数据。返回 (lob_data, labels_ret, time_bucket)。"""
        data_cfg = self.config['data']

        lob_data = np.load(data_cfg['lob_path'])
        labels_ret = np.load(data_cfg['label_path'])
        time_bucket = np.load(data_cfg['time_bucket_path'])
        ## 原来的数据是 ask price，ask vol，bid price，bid vol
        def transform_lob_data(lob_data):
            """
            lob_data: (N, 40) 
            排列: [ap1, av1, bp1, bv1, ap2, av2, bp2, bv2, ..., ap10, av10, bp10, bv10]
            返回: (N, 2, 20) -> Channel 0: Price (bid10...bid1, ask1...ask10), Channel 1: Volume
            """
            # 如果是 numpy 则转为 tensor 处理更方便（或者直接用 numpy 索引）
            if isinstance(lob_data, np.ndarray):
                lob_data = torch.from_numpy(lob_data)
            
            N = lob_data.shape[0]
            
            # 1. 提取所有的 Ask 和 Bid
            # 原始索引中：0, 4, 8... 是 ask_p; 1, 5, 9... 是 ask_v
            #            2, 6, 10... 是 bid_p; 3, 7, 11... 是 bid_v
            ask_p = lob_data[:, 0::2]  # (N, 10) [ap1, ap2, ..., ap10]
            ask_v = lob_data[:, 1::2]  # (N, 10)
            # bid_p = lob_data[:, 2::4]  # (N, 10) [bp1, bp2, ..., bp10]
            # bid_v = lob_data[:, 3::4]  # (N, 10)

            # # 2. 按照 [bid10...bid1, ask1...ask10] 顺序重排
            # # flip(1) 将 [p1, p2...p10] 变为 [p10, p9...p1]
            # price_channel = torch.cat([bid_p.flip(dims=[1]), ask_p], dim=1)  # (N, 20)
            # vol_channel = torch.cat([bid_v.flip(dims=[1]), ask_v], dim=1)    # (N, 20)

            # 3. 堆叠成 (N, 2, 20)
            output = torch.stack([ask_p, ask_v], dim=1)
            
            return output
        lob_data = transform_lob_data(lob_data)
        # lob_data = lob_data

        print(f"数据加载完成:")
        print(f"  LOB: {lob_data.shape}, Labels: {labels_ret.shape}, TimeBucket: {time_bucket.shape}")
        return lob_data, labels_ret, time_bucket

    def generate_signals(
        self, model, lob_data: np.ndarray, time_bucket: np.ndarray
    ) -> pd.DataFrame:
        """
        批量推理生成信号。

        Returns:
            DataFrame: [timestamp, pred_class, prob_down, prob_stationary, prob_up, signal_lob_idx]
        """
        sig_cfg = self.config['signal_config']
        history_T = sig_cfg['history_T']
        stride = sig_cfg['signal_stride']
        batch_size = sig_cfg['batch_size']
        use_amp = sig_cfg.get('use_amp', False)

        dataset = InferenceDataset(lob_data, history_T, stride)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=sig_cfg.get('num_workers', 0),
            pin_memory=sig_cfg.get('pin_memory', False),
        )
        print()

        all_timestamps = []
        all_pred_classes = []
        all_probs = []
        all_lob_indices = []

        with torch.no_grad():
            for batch_lob, batch_label_idx in dataloader:
                batch_lob = batch_lob.to(self.device)

                inputs = {'lob': batch_lob}

                if use_amp and self.device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        logits = model(inputs)
                else:
                    logits = model(inputs)

                probs = F.softmax(logits, dim=1)  # (B, 3)
                pred_classes = torch.argmax(probs, dim=1)  # (B,)

                batch_label_idx_np = batch_label_idx.numpy()
                all_timestamps.extend(time_bucket[batch_label_idx_np].tolist())
                all_pred_classes.extend(pred_classes.cpu().numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())
                all_lob_indices.extend(batch_label_idx_np.tolist())

        probs_array = np.array(all_probs)
        signals = pd.DataFrame({
            'timestamp': all_timestamps,
            'pred_class': all_pred_classes,
            'prob_down': probs_array[:, 0],
            'prob_stationary': probs_array[:, 1],
            'prob_up': probs_array[:, 2],
            'signal_lob_idx': all_lob_indices,
        })

        # 按时间排序
        signals = signals.sort_values('timestamp').reset_index(drop=True)
        print(f"信号生成完成: {len(signals)} 条信号")
        print(f"  类别分布: {signals['pred_class'].value_counts().to_dict()}")
        return signals

    def run(self,model_version,model) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """
        完整流程：加载模型 → 加载数据 → 生成信号 → 保存。

        Returns:
            (signals_df, lob_data, labels_ret, time_bucket)
        """
        model = self.load_model(model_version,model)
        lob_data, labels_ret, time_bucket = self.load_data()

        signals = self.generate_signals(model, lob_data, time_bucket)

        lob_data = np.asarray(lob_data)
        labels_ret = np.asarray(labels_ret)
        time_bucket = np.asarray(time_bucket)
        # 保存信号
        result_dir = Path(self.config.get('output', {}).get('result_dir', 'Backtest/result'))
        result_dir.mkdir(parents=True, exist_ok=True)
        signals.to_csv(result_dir / 'signals.csv', index=False)
        print(f"信号已保存至: {result_dir / 'signals.csv'}")

        return signals, lob_data, labels_ret, time_bucket
