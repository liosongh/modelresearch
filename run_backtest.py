#!/usr/bin/env python
"""
回测系统入口脚本。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from Backtest.backtest_engine import run_backtest
from Backtest.prediction_metrics import evaluate_prediction_performance
from Backtest.signal_generator import generate_signals, save_signals
from Backtest.visualization import (
    plot_backtest_figures,
    plot_dashboard_table,
    plot_prediction_figures,
)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="MultiModal LOB+Trade 回测评估系统")
    parser.add_argument(
        "--config",
        type=str,
        default="Configs/backtest_config.yaml",
        help="回测配置文件路径",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    output_dir = Path(config.get("output_dir", "backtest_results/"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) 信号生成
    signal_df = generate_signals(config)
    save_signals(signal_df, str(output_dir))

    # 2) 预测能力评估
    prediction_results = evaluate_prediction_performance(
        signal_df=signal_df,
        prediction_eval_config=config.get("prediction_eval", {}),
    )

    # 3) 回测：离散 + 连续都跑
    bt_cfg_base = config.get("backtest", {}).copy()

    bt_cfg_discrete = bt_cfg_base.copy()
    bt_cfg_discrete["strategy"] = "discrete"
    backtest_discrete = run_backtest(signal_df=signal_df, backtest_config=bt_cfg_discrete)
    backtest_discrete["result_df"].write_parquet(output_dir / "backtest_discrete.parquet")

    bt_cfg_continuous = bt_cfg_base.copy()
    bt_cfg_continuous["strategy"] = "continuous"
    backtest_continuous = run_backtest(signal_df=signal_df, backtest_config=bt_cfg_continuous)
    backtest_continuous["result_df"].write_parquet(output_dir / "backtest_continuous.parquet")

    # 4) 可视化
    plot_prediction_figures(
        signal_df=signal_df,
        prediction_results=prediction_results,
        output_dir=str(output_dir),
    )
    plot_backtest_figures(backtest_discrete, output_dir=str(output_dir))
    plot_backtest_figures(backtest_continuous, output_dir=str(output_dir))

    summary = {
        "classification_metrics": prediction_results["classification_metrics"],
        "ic_metrics": prediction_results["ic_metrics"],
        "backtest_discrete": backtest_discrete["metrics"],
        "backtest_continuous": backtest_continuous["metrics"],
    }
    plot_dashboard_table(summary, output_dir=str(output_dir))

    # 5) 输出 JSON 汇总
    json_payload = {
        "classification_metrics": prediction_results["classification_metrics"],
        "ic_metrics": prediction_results["ic_metrics"],
        "backtest_discrete": backtest_discrete["metrics"],
        "backtest_continuous": backtest_continuous["metrics"],
    }
    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("回测与评估完成")
    print(f"输出目录: {output_dir.resolve()}")
    print(f"信号数量: {signal_df.height}")
    print(f"离散策略累计收益: {backtest_discrete['metrics']['cumulative_return']:.6f}")
    print(f"连续策略累计收益: {backtest_continuous['metrics']['cumulative_return']:.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
