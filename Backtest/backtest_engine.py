# =============================================================================
# Backtest Engine - 高频回测引擎
# =============================================================================

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BacktestResult:
    """回测结果容器。"""
    trades: pd.DataFrame            # 逐笔交易记录
    equity_curve: pd.DataFrame      # 权益曲线
    signals_with_match: pd.DataFrame  # 信号-行情匹配日志
    config: dict = field(default_factory=dict)


class BacktestEngine:
    """
    高频回测引擎。

    核心规则：
    - 做多（买入）：按 LOB 卖一价（ask1）成交
    - 做空（卖出）：按 LOB 买一价（bid1）成交
    - 100ms 执行延迟：信号 LOB 索引 + execution_delay
    - 全仓操作，无杠杆
    - 手续费 + 滑点双向收取
    """

    # LOB 列索引
    ASK1_PRICE_IDX = 0
    BID1_PRICE_IDX = 1

    def __init__(self, config: dict):
        bt_cfg = config['backtest']
        self.fee_rate = bt_cfg.get('fee_rate', 0.0001)
        self.slippage = bt_cfg.get('slippage', 0.0001)
        self.initial_capital = bt_cfg.get('initial_capital', 1_000_000)
        self.execution_delay = bt_cfg.get('execution_delay', 1)

    def _extract_prices(self, lob_row: np.ndarray):
        """提取 ask1 和 bid1 价格。"""
        return lob_row[self.ASK1_PRICE_IDX], lob_row[self.BID1_PRICE_IDX]

    def _apply_cost_buy(self, price: float) -> float:
        """买入成本：价格上浮（滑点 + 手续费）。"""
        return price * (1 + self.slippage + self.fee_rate)

    def _apply_cost_sell(self, price: float) -> float:
        """卖出成本：价格下压（滑点 + 手续费）。"""
        return price * (1 - self.slippage - self.fee_rate)

    def run(
        self,
        signals: pd.DataFrame,
        lob_data: np.ndarray,
        time_bucket: np.ndarray,
    ) -> BacktestResult:
        """
        执行回测。

        Args:
            signals: SignalGenerator 输出的信号 DataFrame
            lob_data: (N, 40) LOB 原始数据
            time_bucket: (N,) 时间戳数组

        Returns:
            BacktestResult
        """
        N = len(lob_data)

        # 状态变量
        position = 0        # 0=空仓, 1=多头, -1=空头
        capital = self.initial_capital
        entry_price = 0.0
        shares = 0.0

        # 记录
        trades: List[dict] = []
        equity_records: List[dict] = []
        signal_match_records: List[dict] = []

        for _, row in signals.iterrows():
            signal_lob_idx = int(row['signal_lob_idx'])
            exec_lob_idx = signal_lob_idx + self.execution_delay
            pred_class = int(row['pred_class'])

            # 边界检查
            if exec_lob_idx >= N:
                break

            ask1, bid1 = self._extract_prices(lob_data[exec_lob_idx])
            mid_price = (ask1 + bid1) / 2.0
            exec_timestamp = time_bucket[exec_lob_idx]

            # 信号-行情匹配记录
            signal_match_records.append({
                'signal_timestamp': row['timestamp'],
                'exec_timestamp': exec_timestamp,
                'pred_class': pred_class,
                'prob_down': row['prob_down'],
                'prob_stationary': row['prob_stationary'],
                'prob_up': row['prob_up'],
                'ask1': ask1,
                'bid1': bid1,
                'mid_price': mid_price,
                'signal_lob_idx': signal_lob_idx,
                'exec_lob_idx': exec_lob_idx,
            })

            # ========== 交易逻辑 ==========
            if pred_class == 2:  # 预测涨 → 做多
                if position == -1:
                    # 先平空（买入平仓）
                    close_price = self._apply_cost_buy(ask1)
                    pnl = shares * (entry_price - close_price)
                    capital += pnl
                    trades.append({
                        'timestamp': exec_timestamp,
                        'action': 'close_short',
                        'exec_price': close_price,
                        'shares': shares,
                        'pnl': pnl,
                        'position_after': 0,
                        'capital_after': capital,
                    })
                    position = 0
                    shares = 0.0

                if position == 0:
                    # 开多（买入）
                    exec_price = self._apply_cost_buy(ask1)
                    shares = capital / exec_price
                    entry_price = exec_price
                    position = 1
                    trades.append({
                        'timestamp': exec_timestamp,
                        'action': 'open_long',
                        'exec_price': exec_price,
                        'shares': shares,
                        'pnl': 0.0,
                        'position_after': 1,
                        'capital_after': capital,
                    })

            elif pred_class == 0:  # 预测跌 → 做空
                if position == 1:
                    # 先平多（卖出平仓）
                    close_price = self._apply_cost_sell(bid1)
                    pnl = shares * (close_price - entry_price)
                    capital += pnl
                    trades.append({
                        'timestamp': exec_timestamp,
                        'action': 'close_long',
                        'exec_price': close_price,
                        'shares': shares,
                        'pnl': pnl,
                        'position_after': 0,
                        'capital_after': capital,
                    })
                    position = 0
                    shares = 0.0

                if position == 0:
                    # 开空（卖出）
                    exec_price = self._apply_cost_sell(bid1)
                    shares = capital / exec_price
                    entry_price = exec_price
                    position = -1
                    trades.append({
                        'timestamp': exec_timestamp,
                        'action': 'open_short',
                        'exec_price': exec_price,
                        'shares': shares,
                        'pnl': 0.0,
                        'position_after': -1,
                        'capital_after': capital,
                    })
            ## 如果预测类别为1，则看之前是否有仓位，如果有仓位则根据仓位平仓，如果没仓位则不操作
            elif pred_class == 1:
                if position == 1:
                    close_price = self._apply_cost_sell(bid1)
                    pnl = shares * (close_price - entry_price)
                    capital += pnl
                    trades.append({
                        'timestamp': exec_timestamp,
                        'action': 'close_long',
                    })
                    position = 0
                    shares = 0.0
                elif position == -1:
                    close_price = self._apply_cost_buy(ask1)
                    pnl = shares * (entry_price - close_price)
                    capital += pnl
                    trades.append({
                        'timestamp': exec_timestamp,
                        'action': 'close_short',
                        'exec_price': close_price,
                        'shares': shares,
                        'pnl': pnl,
                        'position_after': 0,
                        'capital_after': capital,
                    })
                    position = 0
                    shares = 0.0
                else:
                    pass


            # pred_class == 1 (Stationary) → 不操作

            # ========== 逐步权益计算 ==========
            if position == 1:
                unrealized = shares * (mid_price - entry_price)
            elif position == -1:
                unrealized = shares * (entry_price - mid_price)
            else:
                unrealized = 0.0

            equity = capital + unrealized
            equity_records.append({
                'timestamp': exec_timestamp,
                'equity': equity,
                'capital': capital,
                'unrealized': unrealized,
                'position': position,
            })

        # 收尾：如果有持仓，按最后时刻平仓
        if position != 0 and len(equity_records) > 0:
            last_exec_idx = int(signals.iloc[-1]['signal_lob_idx']) + self.execution_delay
            if last_exec_idx < N:
                ask1, bid1 = self._extract_prices(lob_data[last_exec_idx])
                if position == 1:
                    close_price = self._apply_cost_sell(bid1)
                    pnl = shares * (close_price - entry_price)
                else:
                    close_price = self._apply_cost_buy(ask1)
                    pnl = shares * (entry_price - close_price)
                capital += pnl
                trades.append({
                    'timestamp': time_bucket[last_exec_idx],
                    'action': f'final_close_{"long" if position == 1 else "short"}',
                    'exec_price': close_price,
                    'shares': shares,
                    'pnl': pnl,
                    'position_after': 0,
                    'capital_after': capital,
                })

        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
            columns=['timestamp', 'action', 'exec_price', 'shares', 'pnl', 'position_after', 'capital_after']
        )
        equity_df = pd.DataFrame(equity_records) if equity_records else pd.DataFrame(
            columns=['timestamp', 'equity', 'capital', 'unrealized', 'position']
        )
        signal_match_df = pd.DataFrame(signal_match_records)

        print(f"回测完成:")
        print(f"  总交易次数: {len(trades_df)}")
        print(f"  最终资金: {capital:,.2f}")
        print(f"  总收益率: {(capital / self.initial_capital - 1) * 100:.4f}%")

        return BacktestResult(
            trades=trades_df,
            equity_curve=equity_df,
            signals_with_match=signal_match_df,
            config={
                'fee_rate': self.fee_rate,
                'slippage': self.slippage,
                'initial_capital': self.initial_capital,
                'execution_delay': self.execution_delay,
            },
        )
