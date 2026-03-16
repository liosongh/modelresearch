"""
回测引擎：信号驱动持仓管理。
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import polars as pl


def _annualization_factor() -> int:
    # 6.5 小时/天，10 秒一个信号周期，250 天/年
    return 250 * 2340


def _positions_from_signals(
    pred_class: np.ndarray,
    pred_score: np.ndarray,
    strategy: str,
    score_threshold: float,
) -> np.ndarray:
    if strategy == "discrete":
        # Up=+1, Down=-1, Stationary=0
        return np.where(pred_class == 2, 1.0, np.where(pred_class == 0, -1.0, 0.0))
    if strategy == "continuous":
        pos = pred_score.astype(np.float64).copy()
        pos[np.abs(pos) < score_threshold] = 0.0
        return pos
    raise ValueError(f"未知策略类型: {strategy}")


def _compute_drawdown(nav: np.ndarray) -> Dict[str, Any]:
    running_max = np.maximum.accumulate(nav)
    drawdown = (running_max - nav) / np.maximum(running_max, 1e-12)
    mdd = float(np.max(drawdown))
    end_idx = int(np.argmax(drawdown))
    start_idx = int(np.argmax(nav[: end_idx + 1])) if end_idx >= 0 else 0
    return {
        "drawdown_series": drawdown,
        "mdd": mdd,
        "mdd_start_idx": start_idx,
        "mdd_end_idx": end_idx,
    }


def run_backtest(signal_df: pl.DataFrame, backtest_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    运行单策略回测，返回完整指标与时序结果。
    """
    strategy = backtest_config.get("strategy", "discrete")
    fee_rate = float(backtest_config.get("fee_rate", 0.0001))
    slippage = float(backtest_config.get("slippage", 0.0))
    score_threshold = float(backtest_config.get("score_threshold", 0.0))
    risk_free_rate = float(backtest_config.get("risk_free_rate", 0.0))

    pred_class = signal_df["pred_class"].to_numpy()
    pred_score = signal_df["pred_score"].to_numpy()
    price = signal_df["price"].to_numpy().astype(np.float64)
    tick_idx = signal_df["tick_idx"].to_numpy().astype(np.int64)

    if price.size < 2:
        raise ValueError("信号点数量不足，至少需要 2 个信号点进行回测。")

    # 区间收益：第 i 个信号点到 i+1 信号点
    start_price = price[:-1]
    end_price = price[1:]
    period_return = (end_price - start_price) / np.maximum(start_price, 1e-12)

    # 持仓在当前周期固定为第 i 个信号决定
    position_all = _positions_from_signals(pred_class, pred_score, strategy, score_threshold)
    position = position_all[:-1]

    # 交易成本：仓位变化时扣费
    pos_change = np.abs(position_all[1:] - position_all[:-1])
    trans_cost = pos_change * (fee_rate + slippage)

    gross_pnl = position * period_return
    net_pnl = gross_pnl - trans_cost

    nav = np.cumprod(1.0 + net_pnl)
    cumulative_return = float(nav[-1] - 1.0)
    periods = net_pnl.size
    ann_factor = _annualization_factor()

    annualized_return = float((1.0 + cumulative_return) ** (ann_factor / max(periods, 1)) - 1.0)
    pnl_mean = float(np.mean(net_pnl))
    pnl_std = float(np.std(net_pnl))
    sharpe = float((pnl_mean - risk_free_rate / ann_factor) / pnl_std * np.sqrt(ann_factor)) if pnl_std > 1e-12 else 0.0

    dd = _compute_drawdown(nav)
    mdd = dd["mdd"]
    calmar = float(annualized_return / mdd) if mdd > 1e-12 else 0.0

    win_mask = net_pnl > 0
    loss_mask = net_pnl < 0
    avg_profit = float(np.mean(net_pnl[win_mask])) if win_mask.any() else 0.0
    avg_loss = float(np.mean(np.abs(net_pnl[loss_mask]))) if loss_mask.any() else 0.0
    profit_loss_ratio = float(avg_profit / avg_loss) if avg_loss > 1e-12 else 0.0

    metrics = {
        "strategy": strategy,
        "annualized_return": annualized_return,
        "cumulative_return": cumulative_return,
        "max_drawdown": float(mdd),
        "max_drawdown_start_tick": int(tick_idx[dd["mdd_start_idx"]]),
        "max_drawdown_end_tick": int(tick_idx[min(dd["mdd_end_idx"] + 1, tick_idx.size - 1)]),
        "sharpe_ratio": sharpe,
        "calmar_ratio": calmar,
        "total_trades": int(np.sum(pos_change > 0)),
        "total_cost": float(np.sum(trans_cost)),
        "win_rate": float(np.mean(win_mask)),
        "profit_loss_ratio": profit_loss_ratio,
        "avg_period_return": pnl_mean,
        "period_return_std": pnl_std,
        "num_periods": int(periods),
        "annualization_factor": int(ann_factor),
    }

    result_df = pl.DataFrame(
        {
            "tick_idx_start": tick_idx[:-1],
            "tick_idx_end": tick_idx[1:],
            "position": position,
            "period_return": period_return,
            "gross_pnl": gross_pnl,
            "transaction_cost": trans_cost,
            "net_pnl": net_pnl,
            "nav": nav,
            "drawdown": dd["drawdown_series"],
        }
    )

    return {
        "metrics": metrics,
        "result_df": result_df,
    }
