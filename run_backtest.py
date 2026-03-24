# =============================================================================
# Run Backtest - 回测框架入口
# =============================================================================

import argparse
from pathlib import Path

from Utils.config_loader import load_config
from Backtest import SignalGenerator, BacktestEngine, PredictionMetrics, BacktestVisualizer


def print_summary(report: dict):
    """打印回测摘要。"""
    sig = report.get('signal_metrics', {})
    strat = report.get('strategy_metrics', {})

    print("\n" + "=" * 60)
    print("回测结果摘要")
    print("=" * 60)

    print("\n--- 信号质量 ---")
    print(f"  信号总数:         {sig.get('signal_count_total', 0)}")
    print(f"  非平信号数:       {sig.get('signal_count_non_stationary', 0)}")
    print(f"  信号胜率:         {sig.get('signal_win_rate', 0):.4f}")
    print(f"  信号盈亏比:       {sig.get('signal_profit_loss_ratio', 0):.4f}")
    print(f"  预测准确率:       {sig.get('signal_accuracy', 0):.4f}")

    print("\n--- 实战收益 ---")
    print(f"  初始资金:         {strat.get('initial_capital', 0):,.2f}")
    print(f"  最终权益:         {strat.get('final_equity', 0):,.2f}")
    print(f"  累计收益率:       {strat.get('cumulative_return', 0) * 100:.4f}%")
    print(f"  年化收益率:       {strat.get('annualized_return', 0) * 100:.4f}%")
    print(f"  最大回撤:         {strat.get('max_drawdown', 0) * 100:.4f}%")
    print(f"  夏普比率:         {strat.get('sharpe_ratio', 0):.4f}")
    print(f"  交易次数:         {strat.get('trade_count', 0)}")
    print(f"  交易胜率:         {strat.get('trade_win_rate', 0):.4f}")
    print(f"  交易盈亏比:       {strat.get('trade_profit_loss_ratio', 0):.4f}")
    print(f"  日均收益率:       {strat.get('daily_return', 0) * 100:.6f}%")

    print(f"\n--- 基准对比 ---")
    print(f"  买入持有收益率:   {report.get('benchmark_return', 0) * 100:.4f}%")
    print(f"  超额收益:         {(strat.get('cumulative_return', 0) - report.get('benchmark_return', 0)) * 100:.4f}%")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='高频LOB回测框架')
    parser.add_argument('--config', default='Configs/backtest_config.yaml', help='回测配置文件路径')
    args = parser.parse_args()

    config = load_config(args.config)

    # 创建输出目录
    result_dir = Path(config.get('output', {}).get('result_dir', 'Backtest/result'))
    log_dir = Path(config.get('output', {}).get('log_dir', 'Backtest/log'))
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: 生成信号
    print("=" * 60)
    print("Step 1: 生成预测信号")
    print("=" * 60)
    sig_gen = SignalGenerator(config)
    signals, lob_data, labels_ret, time_bucket = sig_gen.run()

    # Step 2: 运行回测
    print("\n" + "=" * 60)
    print("Step 2: 运行回测引擎")
    print("=" * 60)
    engine = BacktestEngine(config)
    result = engine.run(signals, lob_data, time_bucket)

    # 保存交易日志
    result.trades.to_csv(log_dir / 'trade_log.csv', index=False)
    result.signals_with_match.to_csv(log_dir / 'signal_market_log.csv', index=False)
    result.equity_curve.to_csv(log_dir / 'equity_curve.csv', index=False)
    print(f"交易日志已保存至: {log_dir}")

    # Step 3: 计算指标
    print("\n" + "=" * 60)
    print("Step 3: 计算绩效指标")
    print("=" * 60)
    metrics = PredictionMetrics(
        risk_free_rate=config.get('backtest', {}).get('risk_free_rate', 0.0)
    )
    report, benchmark = metrics.generate_report(
        result, signals, labels_ret, time_bucket, lob_data, config
    )

    # Step 4: 可视化
    print("\n" + "=" * 60)
    print("Step 4: 生成可视化")
    print("=" * 60)
    viz = BacktestVisualizer(output_dir=str(result_dir))
    viz.generate_all(result, signals, benchmark, lob_data=lob_data, time_bucket=time_bucket)

    # 打印摘要
    print_summary(report)


if __name__ == '__main__':
    main()
