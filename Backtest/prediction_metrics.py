"""
预测能力评估模块：胜率为主，IC 为辅。
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import polars as pl
from scipy.stats import pearsonr, spearmanr


def _safe_mean(arr: np.ndarray) -> float:
    return float(np.mean(arr)) if arr.size > 0 else 0.0


def _safe_median(arr: np.ndarray) -> float:
    return float(np.median(arr)) if arr.size > 0 else 0.0


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator != 0 else 0.0


def _safe_corr(func, x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    val, _ = func(x, y)
    if np.isnan(val):
        return 0.0
    return float(val)


def _confidence_binned_analysis(
    pred_score: np.ndarray,
    pred_class: np.ndarray,
    actual_class: np.ndarray,
    actual_return: np.ndarray,
    bins: int = 5,
) -> Tuple[List[Dict[str, Any]], pl.DataFrame]:
    """按 |pred_score| 做等宽分箱，输出方向胜率与平均收益。"""
    conf = np.abs(pred_score)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket_ids = np.digitize(conf, edges[1:-1], right=True)

    rows = []
    for b in range(bins):
        mask = bucket_ids == b
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {
                    "bin_idx": b,
                    "bin_left": float(edges[b]),
                    "bin_right": float(edges[b + 1]),
                    "count": 0,
                    "directional_win_rate": 0.0,
                    "avg_actual_return": 0.0,
                }
            )
            continue

        pred_b = pred_class[mask]
        actual_b = actual_class[mask]
        ret_b = actual_return[mask]
        directional_mask = pred_b != 1
        directional_correct = (pred_b == actual_b) & directional_mask
        directional_wr = _safe_ratio(directional_correct.sum(), directional_mask.sum())

        rows.append(
            {
                "bin_idx": b,
                "bin_left": float(edges[b]),
                "bin_right": float(edges[b + 1]),
                "count": n,
                "directional_win_rate": float(directional_wr),
                "avg_actual_return": float(np.mean(ret_b)),
            }
        )

    df = pl.DataFrame(rows)
    return rows, df


def _rolling_ic(
    pred_score: np.ndarray,
    actual_return: np.ndarray,
    window: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """滚动计算 Pearson IC 与 Rank IC。"""
    n = pred_score.shape[0]
    if n < window:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    ic_series = np.zeros(n - window + 1, dtype=np.float64)
    ric_series = np.zeros(n - window + 1, dtype=np.float64)

    for i in range(n - window + 1):
        x = pred_score[i : i + window]
        y = actual_return[i : i + window]
        ic_series[i] = _safe_corr(pearsonr, x, y)
        ric_series[i] = _safe_corr(spearmanr, x, y)

    return ic_series, ric_series


def evaluate_prediction_performance(
    signal_df: pl.DataFrame,
    prediction_eval_config: Dict[str, Any],
) -> Dict[str, Any]:
    """评估预测能力（分类质量 + IC质量）。"""
    confidence_bins = int(prediction_eval_config.get("confidence_bins", 5))
    ic_rolling_window = int(prediction_eval_config.get("ic_rolling_window", 500))

    pred_class = signal_df["pred_class"].to_numpy()
    actual_class = signal_df["actual_class"].to_numpy()
    pred_score = signal_df["pred_score"].to_numpy()
    actual_return = signal_df["actual_return"].to_numpy()

    # 1) 胜率体系
    overall_win_rate = _safe_ratio((pred_class == actual_class).sum(), pred_class.size)
    directional_mask = pred_class != 1
    directional_win_rate = _safe_ratio(
        ((pred_class == actual_class) & directional_mask).sum(),
        directional_mask.sum(),
    )

    pred_up_mask = pred_class == 2
    pred_down_mask = pred_class == 0
    pred_sta_mask = pred_class == 1

    class_win_rates = {
        "precision_up": _safe_ratio(((actual_class == 2) & pred_up_mask).sum(), pred_up_mask.sum()),
        "precision_down": _safe_ratio(((actual_class == 0) & pred_down_mask).sum(), pred_down_mask.sum()),
        "precision_stationary": _safe_ratio(
            ((actual_class == 1) & pred_sta_mask).sum(), pred_sta_mask.sum()
        ),
    }

    conditional_returns = {
        "up": {
            "mean": _safe_mean(actual_return[pred_up_mask]),
            "median": _safe_median(actual_return[pred_up_mask]),
        },
        "down": {
            "mean": _safe_mean(actual_return[pred_down_mask]),
            "median": _safe_median(actual_return[pred_down_mask]),
        },
        "stationary": {
            "mean": _safe_mean(actual_return[pred_sta_mask]),
            "median": _safe_median(actual_return[pred_sta_mask]),
        },
    }

    conf_rows, confidence_df = _confidence_binned_analysis(
        pred_score=pred_score,
        pred_class=pred_class,
        actual_class=actual_class,
        actual_return=actual_return,
        bins=confidence_bins,
    )

    counts = {
        "up_count": int(pred_up_mask.sum()),
        "stationary_count": int(pred_sta_mask.sum()),
        "down_count": int(pred_down_mask.sum()),
        "total_count": int(pred_class.size),
        "up_ratio": _safe_ratio(pred_up_mask.sum(), pred_class.size),
        "stationary_ratio": _safe_ratio(pred_sta_mask.sum(), pred_class.size),
        "down_ratio": _safe_ratio(pred_down_mask.sum(), pred_class.size),
    }

    # 方向变化频率（仅 Up/Down 切换）
    direction = np.where(pred_class == 2, 1, np.where(pred_class == 0, -1, 0))
    direction_change = (direction[1:] != direction[:-1]) & (direction[1:] != 0) & (direction[:-1] != 0)
    changes = int(direction_change.sum())
    signal_stats = {
        **counts,
        "direction_change_count": changes,
        "avg_periods_per_direction_change": _safe_ratio(direction.shape[0] - 1, changes) if changes > 0 else 0.0,
    }

    # 2) IC 体系
    ic = _safe_corr(pearsonr, pred_score, actual_return)
    rank_ic = _safe_corr(spearmanr, pred_score, actual_return)
    ic_series, ric_series = _rolling_ic(pred_score, actual_return, window=ic_rolling_window)

    ic_mean = _safe_mean(ic_series)
    ic_std = float(np.std(ic_series)) if ic_series.size > 0 else 0.0
    ric_mean = _safe_mean(ric_series)
    ric_std = float(np.std(ric_series)) if ric_series.size > 0 else 0.0

    ic_metrics = {
        "ic": ic,
        "rank_ic": rank_ic,
        "rolling_window": ic_rolling_window,
        "rolling_ic_mean": ic_mean,
        "rolling_ic_std": ic_std,
        "rolling_icir": _safe_ratio(ic_mean, ic_std),
        "rolling_rank_ic_mean": ric_mean,
        "rolling_rank_ic_std": ric_std,
        "rolling_ricir": _safe_ratio(ric_mean, ric_std),
        "rolling_ic_positive_ratio": _safe_ratio((ic_series > 0).sum(), ic_series.size) if ic_series.size > 0 else 0.0,
    }

    results = {
        "classification_metrics": {
            "overall_win_rate": float(overall_win_rate),
            "directional_win_rate": float(directional_win_rate),
            **class_win_rates,
            "conditional_returns": conditional_returns,
            "confidence_binned": conf_rows,
            "signal_stats": signal_stats,
        },
        "ic_metrics": ic_metrics,
        "rolling_ic_series": ic_series.tolist(),
        "rolling_rank_ic_series": ric_series.tolist(),
        "artifacts": {
            "confidence_binned_df": confidence_df,
        },
    }
    return results
