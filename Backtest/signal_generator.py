"""
回测信号生成模块。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import polars as pl
import torch
import yaml
from torch.amp import autocast
from tqdm import tqdm

from Data_Pipeline.dataset import create_dataloaders_for_test
from Model import MultiModalTransformer


def _build_class_labels(labels_ret: np.ndarray, alpha: float = 0.001) -> np.ndarray:
    """把连续收益率标签转换为三分类标签。"""
    labels_class = np.ones_like(labels_ret, dtype=np.int64)
    labels_class[labels_ret > alpha] = 2
    labels_class[labels_ret < -alpha] = 0
    return labels_class


def _load_model(
    model_config_path: str,
    checkpoint_path: str,
    device: str,
) -> Tuple[torch.nn.Module, bool]:
    """加载模型结构与权重，并返回是否需要 trade 模态。"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"checkpoint 缺少 model_state_dict: {checkpoint_path}")
    state_dict = checkpoint["model_state_dict"]

    with open(model_config_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)


    # 优先按 checkpoint 判断是否真的包含 trade 分支参数
    has_trade_weights = any(k.startswith("trade_encoder.") for k in state_dict.keys())
    cfg_has_trade = model_cfg.get("trade_encoder") is not None
    use_trade = bool(has_trade_weights and cfg_has_trade)

    if use_trade:
        model = MultiModalTransformer.from_config(model_config_path)
    else:
        # 兼容 LOB-only checkpoint
        model = MultiModalTransformer(
            lob_config=model_cfg.get("lob_encoder", {}),
            trade_config=None,
            fusion_config=model_cfg.get("fusion", {}),
            transformer_config=model_cfg.get("transformer", {}),
            output_config=model_cfg.get("output_head", {}),
        )


    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, use_trade


def _prepare_test_data(
    lob_data: np.ndarray,
    trade_data: np.ndarray | None,
    labels_ret: np.ndarray,
    alpha: float,
    use_trade: bool,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """直接使用传入的测试集数据，不做任何切分。"""
    data_dict: Dict[str, np.ndarray] = {"lob": lob_data}
    if use_trade and trade_data is not None:
        data_dict["trade"] = trade_data

    test_labels_ret = labels_ret
    test_labels_class = _build_class_labels(test_labels_ret, alpha=alpha)
    return data_dict, test_labels_class, test_labels_ret


def generate_signals(backtest_config: Dict[str, Any]) -> pl.DataFrame:
    """
    在测试集上按固定间隔生成交易信号。

    Returns:
        pl.DataFrame: [
            tick_idx, pred_class, p_down, p_stationary, p_up,
            pred_score, actual_return, actual_class, price
        ]
    """
    signal_cfg = backtest_config.get("signal_config", {})
    data_cfg = backtest_config.get("data", {})

    history_t = int(signal_cfg.get("history_T", 3000))
    signal_stride = int(signal_cfg.get("signal_stride", 100))
    batch_size = int(signal_cfg.get("batch_size", 512))
    num_workers = int(signal_cfg.get("num_workers", 4))
    pin_memory = bool(signal_cfg.get("pin_memory", True))
    alpha = float(signal_cfg.get("alpha", 0.001))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(signal_cfg.get("use_amp", True)) and device.startswith("cuda")

    lob_data = np.load(data_cfg["lob_path"])
    trade_path = data_cfg.get("trade_path")
    trade_data = np.load(trade_path) if trade_path else None
    labels_ret = np.load(data_cfg["label_path"])
    price_data = np.load(data_cfg["price_path"]).astype(np.float64)

    model, use_trade = _load_model(
        model_config_path=backtest_config["model_config"],
        checkpoint_path=backtest_config["model_checkpoint"],
        device=device,
    )

    test_dict, test_labels_class, test_labels_ret = _prepare_test_data(
        lob_data=lob_data,
        trade_data=trade_data,
        labels_ret=labels_ret,
        alpha=alpha,
        use_trade=use_trade,
    )

    loader_cfg = {
        "history_T": history_t,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "stride": signal_stride,
    }
    test_loader = create_dataloaders_for_test(
        data_dict=test_dict,
        labels=test_labels_class,
        returns=test_labels_ret,
        config=loader_cfg,
        device="cpu",
    )

    pred_classes = []
    pred_probas = []
    actual_returns = []
    tick_indices = []
    prices = []

    with torch.no_grad():
        global_row = 0
        for inputs, _, returns in tqdm(test_loader, desc="Generating signals", leave=False):
            if not isinstance(inputs, dict):
                raise TypeError("输入格式异常，期望 dict[str, Tensor]。")
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

            with autocast(device_type="cuda", enabled=use_amp):
                logits = model(inputs)
                probas = torch.softmax(logits, dim=1)

            preds = torch.argmax(probas, dim=1).cpu().numpy()
            probas_np = probas.cpu().numpy()
            returns_np = returns.numpy()

            batch_now = preds.shape[0]
            for j in range(batch_now):
                dataset_idx = global_row + j
                start_idx_in_test = dataset_idx * signal_stride
                test_tick_idx = (history_t - 1) + start_idx_in_test
                if test_tick_idx >= len(price_data):
                    continue

                raw_tick_idx = test_tick_idx
                tick_indices.append(int(raw_tick_idx))
                pred_classes.append(int(preds[j]))
                pred_probas.append(probas_np[j])
                actual_returns.append(float(returns_np[j]))
                prices.append(float(price_data[test_tick_idx]))

            global_row += batch_now

    if len(pred_classes) == 0:
        raise RuntimeError("未生成任何信号，请检查数据长度与 history_T / stride 配置。")

    pred_probas_arr = np.asarray(pred_probas, dtype=np.float64)
    pred_score = pred_probas_arr[:, 2] - pred_probas_arr[:, 0]

    actual_return_arr = np.asarray(actual_returns, dtype=np.float64)
    actual_class = np.ones_like(actual_return_arr, dtype=np.int64)
    actual_class[actual_return_arr > alpha] = 2
    actual_class[actual_return_arr < -alpha] = 0

    signal_df = pl.DataFrame(
        {
            "tick_idx": tick_indices,
            "pred_class": pred_classes,
            "p_down": pred_probas_arr[:, 0],
            "p_stationary": pred_probas_arr[:, 1],
            "p_up": pred_probas_arr[:, 2],
            "pred_score": pred_score,
            "actual_return": actual_returns,
            "actual_class": actual_class,
            "price": prices,
        }
    )
    return signal_df


def save_signals(signal_df: pl.DataFrame, output_dir: str) -> None:
    """保存信号表。"""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    signal_df.write_parquet(out_dir / "signals.parquet")
    signal_df.write_csv(out_dir / "signals.csv")
