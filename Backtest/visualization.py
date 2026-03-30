# =============================================================================
# Backtest Visualization - 回测可视化
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from typing import Optional, List

from .backtest_engine import BacktestResult

# 中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class BacktestVisualizer:
    """回测结果可视化。"""

    def __init__(self, output_dir: str = "Backtest/result"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save_fig(self, fig, name: str):
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  图表已保存: {path}")

    def plot_equity_curve(
        self,
        equity_curve: pd.DataFrame,
        benchmark: Optional[pd.DataFrame] = None,
        initial_capital: float = 1_000_000,
    ):
        """收益曲线（策略 vs 买入持有基准）。"""
        fig, ax = plt.subplots(figsize=(14, 6))

        # 归一化为收益率
        strategy_ret = equity_curve['equity'].values / initial_capital - 1
        ax.plot(range(len(strategy_ret)), strategy_ret * 100, label='Strategy', linewidth=1.2)

        if benchmark is not None and not benchmark.empty:
            bm_ret = benchmark['equity'].values / benchmark['equity'].iloc[0] - 1
            # 对齐长度
            bm_len = min(len(bm_ret), len(strategy_ret))
            ax.plot(range(bm_len), bm_ret[:bm_len] * 100, label='Buy & Hold', linewidth=1.0, alpha=0.7)

        ax.set_xlabel('Signal Step')
        ax.set_ylabel('Return (%)')
        ax.set_title('Equity Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        self._save_fig(fig, 'equity_curve')

    def plot_drawdown(self, equity_curve: pd.DataFrame):
        """回撤曲线。"""
        fig, ax = plt.subplots(figsize=(14, 4))

        equity_values = equity_curve['equity'].values
        peak = np.maximum.accumulate(equity_values)
        drawdown = (peak - equity_values) / peak * 100

        ax.fill_between(range(len(drawdown)), drawdown, color='red', alpha=0.3)
        ax.plot(range(len(drawdown)), drawdown, color='red', linewidth=0.8)

        ax.set_xlabel('Signal Step')
        ax.set_ylabel('Drawdown (%)')
        ax.set_title('Drawdown Curve')
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()

        self._save_fig(fig, 'drawdown')

    def plot_signal_distribution(self, signals: pd.DataFrame):
        """信号类别分布。"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 类别分布柱状图
        class_names = {0: 'Down', 1: 'Stationary', 2: 'Up'}
        counts = signals['pred_class'].value_counts().sort_index()
        colors = ['#e74c3c', '#95a5a6', '#2ecc71']
        bars = axes[0].bar(
            [class_names.get(i, str(i)) for i in counts.index],
            counts.values,
            color=[colors[i] for i in counts.index],
        )
        axes[0].set_title('Signal Class Distribution')
        axes[0].set_ylabel('Count')
        for bar, val in zip(bars, counts.values):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         str(val), ha='center', va='bottom', fontsize=10)

        # 概率分布直方图
        axes[1].hist(signals['prob_up'], bins=50, alpha=0.6, label='P(Up)', color='#2ecc71')
        axes[1].hist(signals['prob_down'], bins=50, alpha=0.6, label='P(Down)', color='#e74c3c')
        axes[1].set_title('Prediction Probability Distribution')
        axes[1].set_xlabel('Probability')
        axes[1].set_ylabel('Count')
        axes[1].legend()

        fig.tight_layout()
        self._save_fig(fig, 'signal_distribution')

    def plot_pnl_distribution(self, trades: pd.DataFrame):
        """单笔交易盈亏分布。"""
        close_trades = trades[trades['action'].str.startswith('close') | trades['action'].str.startswith('final')]
        if close_trades.empty:
            print("  无平仓交易，跳过盈亏分布图")
            return

        fig, ax = plt.subplots(figsize=(10, 5))

        pnls = close_trades['pnl'].values
        colors = ['#2ecc71' if p > 0 else '#e74c3c' for p in pnls]

        ax.hist(pnls, bins=50, color='#3498db', alpha=0.7, edgecolor='white')
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax.axvline(x=pnls.mean(), color='orange', linestyle='--', label=f'Mean: {pnls.mean():.2f}')

        ax.set_xlabel('PnL')
        ax.set_ylabel('Count')
        ax.set_title(f'Trade PnL Distribution (n={len(pnls)})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        self._save_fig(fig, 'pnl_distribution')

    def plot_cumulative_trades(self, trades: pd.DataFrame):
        """累计盈亏曲线（按交易笔数）。"""
        close_trades = trades[trades['action'].str.startswith('close') | trades['action'].str.startswith('final')]
        if close_trades.empty:
            return

        fig, ax = plt.subplots(figsize=(12, 5))

        cum_pnl = close_trades['pnl'].cumsum()
        ax.plot(range(len(cum_pnl)), cum_pnl.values, linewidth=1.2)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Cumulative PnL')
        ax.set_title('Cumulative PnL by Trade')
        ax.grid(True, alpha=0.3)

        self._save_fig(fig, 'cumulative_pnl')

    def plot_signal_midprice_detail(
        self,
        result: BacktestResult,
        lob_data: np.ndarray,
        time_bucket: np.ndarray,
        window_seconds: float = 60.0,
        num_windows: int = 4,
        execution_delay: int = 1,
    ):
        """
        信号与 midprice 交互可视化。

        在指定时长窗口内绘制 100ms 粒度的 midprice 走势，
        并用标记标注买入（开多/平空）和卖出（开空/平多）交易点。

        Args:
            result: 回测结果
            lob_data: (N, 40) LOB 原始数据
            time_bucket: (N,) 时间戳
            window_seconds: 每个窗口时长（秒），默认 60s
            num_windows: 生成多少个窗口子图
            execution_delay: 执行延迟 tick 数
        """
        trades = result.trades
        if trades.empty:
            print("  无交易记录，跳过信号-midprice 交互图")
            return

        ASK1_IDX, BID1_IDX = 0, 1

        # 推断时间戳单位：如果值 > 1e12 视为毫秒，否则视为秒
        ts_sample = float(time_bucket[0])
        ts_is_ms = ts_sample > 1e12
        ticks_per_second = 10  # 100ms 间隔
        window_ticks = int(window_seconds * ticks_per_second)

        # 找到有交易的时间戳，选取分散的窗口
        trade_timestamps = trades['timestamp'].values
        trade_lob_indices = []
        for ts in trade_timestamps:
            idx = np.searchsorted(time_bucket, ts)
            if idx < len(time_bucket):
                trade_lob_indices.append(idx)
        trade_lob_indices = np.array(trade_lob_indices)

        if len(trade_lob_indices) == 0:
            print("  无法匹配交易时间戳，跳过信号-midprice 交互图")
            return

        # 选取 num_windows 个有交易的窗口，均匀分布在整个回测时段
        window_centers = self._select_window_centers(
            trade_lob_indices, window_ticks, len(lob_data), num_windows
        )

        if len(window_centers) == 0:
            print("  无法找到合适的窗口，跳过信号-midprice 交互图")
            return

        # 绘图
        n_plots = len(window_centers)
        fig, axes = plt.subplots(n_plots, 1, figsize=(16, 5 * n_plots))
        if n_plots == 1:
            axes = [axes]

        for i, center in enumerate(window_centers):
            ax = axes[i]
            start = max(0, center - window_ticks // 2)
            end = min(len(lob_data), start + window_ticks)
            start = max(0, end - window_ticks)  # 保证窗口长度

            # midprice 曲线
            ask1 = lob_data[start:end, ASK1_IDX]
            bid1 = lob_data[start:end,BID1_IDX]
            mid = (ask1 + bid1) / 2.0

            # X 轴：相对秒数
            x_seconds = np.arange(len(mid)) * 0.1  # 100ms = 0.1s

            ax.plot(x_seconds, mid, color='#2c3e50', linewidth=0.8, label='Mid Price', zorder=1)
            ax.fill_between(x_seconds, bid1, ask1, color='#3498db', alpha=0.1, label='Bid-Ask Spread')

            # 标注交易
            window_ts_start = time_bucket[start]
            window_ts_end = time_bucket[end - 1]

            buy_actions = ('open_long', 'close_short')
            sell_actions = ('open_short', 'close_long', 'final_close_long', 'final_close_short')

            for _, trade in trades.iterrows():
                t_ts = trade['timestamp']
                if t_ts < window_ts_start or t_ts > window_ts_end:
                    continue

                # 找到这笔交易在窗口内的位置
                t_idx = np.searchsorted(time_bucket, t_ts)
                if t_idx < start or t_idx >= end:
                    continue

                x_pos = (t_idx - start) * 0.1
                action = trade['action']

                if action in buy_actions:
                    ax.scatter(x_pos, trade['exec_price'], marker='^', color='#e74c3c',
                               s=120, zorder=3, edgecolors='black', linewidths=0.5)
                    ax.annotate(action.replace('_', ' '), (x_pos, trade['exec_price']),
                                textcoords='offset points', xytext=(5, 10),
                                fontsize=7, color='#e74c3c', fontweight='bold')
                else:
                    ax.scatter(x_pos, trade['exec_price'], marker='v', color='#2ecc71',
                               s=120, zorder=3, edgecolors='black', linewidths=0.5)
                    ax.annotate(action.replace('_', ' '), (x_pos, trade['exec_price']),
                                textcoords='offset points', xytext=(5, -15),
                                fontsize=7, color='#2ecc71', fontweight='bold')

            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Price')
            window_label = f"Window {i+1}: LOB index [{start}, {end})"
            if ts_is_ms:
                from datetime import datetime
                t_start_str = datetime.fromtimestamp(window_ts_start / 1000).strftime('%H:%M:%S')
                t_end_str = datetime.fromtimestamp(window_ts_end / 1000).strftime('%H:%M:%S')
                window_label += f"  ({t_start_str} ~ {t_end_str})"
            ax.set_title(window_label)
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle('Signal & Mid-Price Detail (100ms resolution)', fontsize=14, y=1.01)
        fig.tight_layout()
        self._save_fig(fig, 'signal_midprice_detail')

    @staticmethod
    def _select_window_centers(
        trade_lob_indices: np.ndarray,
        window_ticks: int,
        data_len: int,
        num_windows: int,
    ) -> List[int]:
        """从有交易的位置中均匀选取窗口中心点。"""
        # 按位置排序去重
        unique_indices = np.unique(trade_lob_indices)

        # 过滤掉边界附近的（确保窗口不越界）
        half = window_ticks // 2
        valid = unique_indices[(unique_indices >= half) & (unique_indices < data_len - half)]
        if len(valid) == 0:
            return []

        if len(valid) <= num_windows:
            return valid.tolist()

        # 均匀选取
        step = len(valid) / num_windows
        selected = [valid[int(i * step)] for i in range(num_windows)]
        return selected

    def generate_all(
        self,
        result: BacktestResult,
        signals: pd.DataFrame,
        benchmark: Optional[pd.DataFrame] = None,
        lob_data: Optional[np.ndarray] = None,
        time_bucket: Optional[np.ndarray] = None,
    ):
        """生成所有图表。"""
        print("生成可视化图表:")
        initial_capital = result.config.get('initial_capital', 1_000_000)

        self.plot_equity_curve(result.equity_curve, benchmark, initial_capital)
        self.plot_drawdown(result.equity_curve)
        self.plot_signal_distribution(signals)
        self.plot_pnl_distribution(result.trades)
        self.plot_cumulative_trades(result.trades)

        if lob_data is not None and time_bucket is not None:
            self.plot_signal_midprice_detail(
                result, lob_data, time_bucket,
                execution_delay=result.config.get('execution_delay', 1),
                window_seconds = 1800
            )

        print("所有图表生成完成")
