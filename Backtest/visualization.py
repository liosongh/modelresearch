"""
可视化报告模块。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.metrics import confusion_matrix


def _ensure_output_dir(output_dir: str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _rolling_sharpe(net_pnl: np.ndarray, window: int, annualization_factor: int) -> np.ndarray:
    if net_pnl.size < window:
        return np.array([], dtype=np.float64)
    out = np.zeros(net_pnl.size - window + 1, dtype=np.float64)
    for i in range(out.size):
        seg = net_pnl[i : i + window]
        seg_std = np.std(seg)
        out[i] = (np.mean(seg) / seg_std * np.sqrt(annualization_factor)) if seg_std > 1e-12 else 0.0
    return out


def plot_prediction_figures(
    signal_df: pl.DataFrame,
    prediction_results: Dict[str, Any],
    output_dir: str,
) -> None:
    out_dir = _ensure_output_dir(output_dir)
    sns.set_style("whitegrid")

    pred_class = signal_df["pred_class"].to_numpy()
    actual_class = signal_df["actual_class"].to_numpy()
    pred_score = signal_df["pred_score"].to_numpy()
    actual_return = signal_df["actual_return"].to_numpy()

    # 1) 混淆矩阵热力图（归一化）
    cm = confusion_matrix(actual_class, pred_class, labels=[0, 1, 2], normalize="true")
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".3f", cmap="Blues", xticklabels=["Down", "Stationary", "Up"], yticklabels=["Down", "Stationary", "Up"])
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_dir / "prediction_confusion_matrix.png", dpi=160)
    plt.close()

    # 2) 条件收益箱线图
    box_df = pl.DataFrame(
        {
            "pred_label": np.where(pred_class == 2, "Up", np.where(pred_class == 0, "Down", "Stationary")),
            "actual_return": actual_return,
        }
    ).to_pandas()
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=box_df, x="pred_label", y="actual_return", order=["Down", "Stationary", "Up"])
    plt.title("Conditional Return by Predicted Class")
    plt.tight_layout()
    plt.savefig(out_dir / "prediction_conditional_return_boxplot.png", dpi=160)
    plt.close()

    # 3) 置信度分箱 vs 胜率
    conf_df = prediction_results["artifacts"]["confidence_binned_df"].to_pandas()
    conf_df["bin_label"] = conf_df.apply(lambda r: f"[{r['bin_left']:.1f},{r['bin_right']:.1f}]", axis=1)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=conf_df, x="bin_label", y="directional_win_rate", color="#3b82f6")
    plt.ylim(0, 1)
    plt.title("Confidence Bin vs Directional Win Rate")
    plt.xlabel("abs(pred_score) bin")
    plt.ylabel("Directional Win Rate")
    plt.tight_layout()
    plt.savefig(out_dir / "prediction_confidence_bin_winrate.png", dpi=160)
    plt.close()

    # 4) 滚动 IC 时序
    ic_series = np.asarray(prediction_results["rolling_ic_series"], dtype=np.float64)
    if ic_series.size > 0:
        mu = np.mean(ic_series)
        sigma = np.std(ic_series)
        plt.figure(figsize=(10, 4))
        plt.plot(ic_series, label="Rolling IC", linewidth=1.2)
        plt.axhline(mu, color="red", linestyle="--", label="mean")
        plt.axhline(mu + sigma, color="gray", linestyle=":", label="+1σ")
        plt.axhline(mu - sigma, color="gray", linestyle=":")
        plt.title("Rolling IC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "prediction_rolling_ic.png", dpi=160)
        plt.close()

    # 5) pred_score vs actual_return 散点图（下采样）
    n = pred_score.size
    if n > 12000:
        idx = np.linspace(0, n - 1, 12000).astype(int)
        x = pred_score[idx]
        y = actual_return[idx]
    else:
        x = pred_score
        y = actual_return
    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, s=6, alpha=0.25)
    plt.title("Pred Score vs Actual Return")
    plt.xlabel("pred_score")
    plt.ylabel("actual_return")
    plt.tight_layout()
    plt.savefig(out_dir / "prediction_score_vs_return.png", dpi=160)
    plt.close()


def plot_backtest_figures(
    backtest_result: Dict[str, Any],
    output_dir: str,
    rolling_sharpe_window: int = 200,
) -> None:
    out_dir = _ensure_output_dir(output_dir)
    sns.set_style("whitegrid")

    result_df = backtest_result["result_df"]
    metrics = backtest_result["metrics"]

    tick_idx = result_df["tick_idx_end"].to_numpy()
    nav = result_df["nav"].to_numpy()
    drawdown = result_df["drawdown"].to_numpy()
    position = result_df["position"].to_numpy()
    net_pnl = result_df["net_pnl"].to_numpy()

    # 1) 累计净值
    plt.figure(figsize=(10, 4))
    plt.plot(tick_idx, nav, linewidth=1.2)
    plt.title(f"Strategy NAV ({metrics['strategy']})")
    plt.xlabel("tick_idx")
    plt.ylabel("NAV")
    plt.tight_layout()
    plt.savefig(out_dir / f"backtest_nav_{metrics['strategy']}.png", dpi=160)
    plt.close()

    # 2) 回撤曲线
    plt.figure(figsize=(10, 3))
    plt.fill_between(tick_idx, -drawdown, 0.0, alpha=0.35)
    plt.title(f"Underwater Curve ({metrics['strategy']})")
    plt.xlabel("tick_idx")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(out_dir / f"backtest_drawdown_{metrics['strategy']}.png", dpi=160)
    plt.close()

    # 3) 仓位时序
    plt.figure(figsize=(10, 3))
    plt.step(tick_idx, position, where="post")
    plt.title(f"Position Time Series ({metrics['strategy']})")
    plt.xlabel("tick_idx")
    plt.ylabel("position")
    plt.tight_layout()
    plt.savefig(out_dir / f"backtest_position_{metrics['strategy']}.png", dpi=160)
    plt.close()

    # 4) 每期 PnL 分布
    plt.figure(figsize=(7, 4))
    sns.histplot(net_pnl, bins=80, kde=True)
    plt.title(f"Period Net PnL Distribution ({metrics['strategy']})")
    plt.tight_layout()
    plt.savefig(out_dir / f"backtest_pnl_hist_{metrics['strategy']}.png", dpi=160)
    plt.close()

    # 5) 滚动夏普
    rs = _rolling_sharpe(net_pnl, rolling_sharpe_window, metrics["annualization_factor"])
    if rs.size > 0:
        rs_tick = tick_idx[rolling_sharpe_window - 1 :]
        plt.figure(figsize=(10, 3))
        plt.plot(rs_tick, rs, linewidth=1.1)
        plt.axhline(0.0, color="gray", linestyle="--", linewidth=0.9)
        plt.title(f"Rolling Sharpe ({metrics['strategy']})")
        plt.xlabel("tick_idx")
        plt.ylabel("Sharpe")
        plt.tight_layout()
        plt.savefig(out_dir / f"backtest_rolling_sharpe_{metrics['strategy']}.png", dpi=160)
        plt.close()


def plot_dashboard_table(
    summary: Dict[str, Any],
    output_dir: str,
    filename: str = "dashboard_metrics.png",
) -> None:
    out_dir = _ensure_output_dir(output_dir)

    # 扁平化展示
    rows = []
    for section, values in summary.items():
        if isinstance(values, dict):
            for k, v in values.items():
                if isinstance(v, (dict, list)):
                    continue
                rows.append((section, k, v))

    if len(rows) == 0:
        return

    fig_h = min(0.32 * len(rows) + 1.5, 24)
    plt.figure(figsize=(12, fig_h))
    plt.axis("off")

    cell_text = [[r[0], r[1], f"{r[2]:.6f}" if isinstance(r[2], float) else str(r[2])] for r in rows]
    table = plt.table(
        cellText=cell_text,
        colLabels=["section", "metric", "value"],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)

    plt.title("Backtest Summary Dashboard", pad=16)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=160)
    plt.close()
