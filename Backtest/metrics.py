# =============================================================================
# Prediction Metrics - 绩效评估体系
# =============================================================================

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Optional

from .backtest_engine import BacktestResult


class PredictionMetrics:
    """信号质量 + 实战收益指标计算。"""

    def __init__(self, risk_free_rate: float = 0.0):
        self.risk_free_rate = risk_free_rate

    # =========================================================================
    # 信号质量指标
    # =========================================================================
    def compute_signal_metrics(
        self,
        signals: pd.DataFrame,
        labels_ret: np.ndarray,
        time_bucket: np.ndarray,
        alpha: float = 0.001,
    ) -> dict:
        """
        评估模型信号本身的质量（与实际涨跌对比）。

        Args:
            signals: 信号 DataFrame，含 signal_lob_idx, pred_class
            labels_ret: (N,) 连续收益率
            time_bucket: (N,) 时间戳
            alpha: 涨跌阈值
        """
        indices = signals['signal_lob_idx'].values.astype(int)
        actual_ret = labels_ret[indices]

        # 实际类别
        actual_class = np.ones(len(actual_ret), dtype=int)  # 默认 1=Stationary
        actual_class[actual_ret > alpha] = 2   # Up
        actual_class[actual_ret < -alpha] = 0  # Down

        pred_class = signals['pred_class'].values

        # 仅评估非 Stationary 的预测
        non_stat_mask = pred_class != 1
        if non_stat_mask.sum() == 0:
            return {'signal_win_rate': 0.0, 'signal_count': 0}

        pred_direction = pred_class[non_stat_mask]
        actual_direction = actual_class[non_stat_mask]

        # 胜率：预测方向与实际方向一致
        correct = (pred_direction == actual_direction).sum()
        win_rate = correct / len(pred_direction)

        # 盈亏比：按预测方向计算收益
        pred_returns = actual_ret[non_stat_mask].copy()
        # 做空的信号，收益取反
        short_mask = pred_direction == 0
        pred_returns[short_mask] = -pred_returns[short_mask]

        gains = pred_returns[pred_returns > 0]
        losses = pred_returns[pred_returns < 0]
        avg_gain = gains.mean() if len(gains) > 0 else 0.0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 1e-9
        profit_loss_ratio = avg_gain / avg_loss

        # 各类预测的准确率
        accuracy_overall = (pred_class == actual_class).mean()

        return {
            'signal_win_rate': float(win_rate),
            'signal_profit_loss_ratio': float(profit_loss_ratio),
            'signal_accuracy': float(accuracy_overall),
            'signal_count_non_stationary': int(non_stat_mask.sum()),
            'signal_count_total': len(signals),
            'pred_class_dist': {int(k): int(v) for k, v in zip(*np.unique(pred_class, return_counts=True))},
            'actual_class_dist': {int(k): int(v) for k, v in zip(*np.unique(actual_class, return_counts=True))},
        }

    # =========================================================================
    # 实战收益指标
    # =========================================================================
    def compute_strategy_metrics(self, result: BacktestResult) -> dict:
        """从回测结果计算策略绩效指标。"""
        trades = result.trades
        equity = result.equity_curve
        initial_capital = result.config['initial_capital']

        if equity.empty:
            return {'error': 'no equity data'}

        equity_values = equity['equity'].values
        final_equity = equity_values[-1]

        # 累计收益率
        cumulative_return = (final_equity / initial_capital) - 1

        # 时间跨度（假设 timestamp 为毫秒级 epoch）
        timestamps = equity['timestamp'].values
        time_span_ms = float(timestamps[-1] - timestamps[0])
        time_span_seconds = time_span_ms / 1000 if time_span_ms > 1e9 else time_span_ms  # 兼容秒/毫秒

        # 年化收益率
        if time_span_seconds > 0:
            seconds_per_year = 365.25 * 24 * 3600
            years = time_span_seconds / seconds_per_year
            annualized_return = (final_equity / initial_capital) ** (1 / years) - 1 if years > 0 else 0.0
        else:
            annualized_return = 0.0

        # 最大回撤
        peak = np.maximum.accumulate(equity_values)
        drawdown = (peak - equity_values) / peak
        max_drawdown = float(drawdown.max())

        # 夏普比率（基于逐步收益率）
        if len(equity_values) > 1:
            returns = np.diff(equity_values) / equity_values[:-1]
            returns = returns[np.isfinite(returns)]
            if len(returns) > 1 and returns.std() > 0:
                # 每步间隔约为 signal_stride * 100ms
                steps_per_year = seconds_per_year / (time_span_seconds / len(returns)) if time_span_seconds > 0 else 1
                excess_return = returns.mean() - self.risk_free_rate / steps_per_year
                sharpe = float(np.sqrt(steps_per_year) * excess_return / returns.std())
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        # 交易统计
        if not trades.empty:
            close_trades = trades[trades['action'].str.startswith('close') | trades['action'].str.startswith('final')]
            trade_pnls = close_trades['pnl'].values if not close_trades.empty else np.array([])

            trade_count = len(close_trades)
            if len(trade_pnls) > 0:
                trade_win_rate = float((trade_pnls > 0).mean())
                gains = trade_pnls[trade_pnls > 0]
                losses = trade_pnls[trade_pnls < 0]
                avg_gain = gains.mean() if len(gains) > 0 else 0.0
                avg_loss = abs(losses.mean()) if len(losses) > 0 else 1e-9
                trade_profit_loss_ratio = float(avg_gain / avg_loss)
            else:
                trade_win_rate = 0.0
                trade_profit_loss_ratio = 0.0
        else:
            trade_count = 0
            trade_win_rate = 0.0
            trade_profit_loss_ratio = 0.0

        # 日均收益
        if time_span_seconds > 0:
            days = time_span_seconds / 86400
            daily_return = cumulative_return / days if days > 0 else 0.0
        else:
            daily_return = 0.0

        return {
            'initial_capital': float(initial_capital),
            'final_equity': float(final_equity),
            'cumulative_return': float(cumulative_return),
            'annualized_return': float(annualized_return),
            'max_drawdown': float(max_drawdown),
            'sharpe_ratio': sharpe,
            'trade_count': trade_count,
            'trade_win_rate': float(trade_win_rate),
            'trade_profit_loss_ratio': float(trade_profit_loss_ratio),
            'daily_return': float(daily_return),
        }

    # =========================================================================
    # 基准：买入持有
    # =========================================================================
    def compute_benchmark(
        self,
        lob_data: np.ndarray,
        time_bucket: np.ndarray,
        signal_lob_indices: np.ndarray,
        initial_capital: float,
        execution_delay: int = 1,
    ) -> pd.DataFrame:
        """
        买入持有基准：首个信号时刻买入，持有到最后。

        Returns:
            DataFrame: [timestamp, equity]
        """
        ask1_idx = 0
        bid1_idx = 1

        start_idx = int(signal_lob_indices[0]) + execution_delay
        end_idx = int(signal_lob_indices[-1]) + execution_delay

        buy_price = lob_data[start_idx,ask1_idx]
        shares = initial_capital / buy_price
        # print('buy_price: ', buy_price)
        # print('shares: ', shares)

        records = []
        for lob_idx in signal_lob_indices.astype(int) + execution_delay:
            if lob_idx >= len(lob_data):
                break
            mid = (lob_data[lob_idx, ask1_idx] + lob_data[lob_idx, bid1_idx]) / 2.0
            equity = shares * mid
            # if lob_idx < signal_lob_indices[1]:

            #     print('equity: ', equity)
            records.append({
                'timestamp': time_bucket[lob_idx],
                'equity': equity,
            })

        return pd.DataFrame(records)

    # =========================================================================
    # 汇总报告
    # =========================================================================
    def generate_report(
        self,
        result: BacktestResult,
        signals: pd.DataFrame,
        labels_ret: np.ndarray,
        time_bucket: np.ndarray,
        lob_data: np.ndarray,
        config: dict,
    ) -> dict:
        """生成完整绩效报告并保存为 JSON。"""
        alpha = config.get('signal_config', {}).get('alpha', 0.001)

        signal_metrics = self.compute_signal_metrics(signals, labels_ret, time_bucket, alpha)
        strategy_metrics = self.compute_strategy_metrics(result)

        benchmark = self.compute_benchmark(
            lob_data, time_bucket,
            signals['signal_lob_idx'].values,
            result.config['initial_capital'],
            result.config['execution_delay'],
        )
        if not benchmark.empty:
            bm_initial = benchmark['equity'].iloc[0]
            bm_final = benchmark['equity'].iloc[-1]
            benchmark_return = benchmark_return = float( ((bm_final / bm_initial) - 1).item() )
        else:
            benchmark_return = 0.0

        report = {
            'signal_metrics': signal_metrics,
            'strategy_metrics': strategy_metrics,
            'benchmark_return': benchmark_return,
            'backtest_config': result.config,
        }

        # 保存
        result_dir = Path(config.get('output', {}).get('result_dir', 'Backtest/result'))
        result_dir.mkdir(parents=True, exist_ok=True)
        with open(result_dir / 'metrics_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"指标报告已保存至: {result_dir / 'metrics_report.json'}")

        return report, benchmark
