import os
import polars as pl

from typing import List, Optional, Tuple
# ============================================================
# 1. 基础数据加载
# ============================================================

def load_trade_data(trade_dir: str, 
                    days: Optional[int] = None, 
                    date: Optional[List[str]] = None) -> pl.DataFrame:
    """
    加载Trade数据
    
    Args:
        trade_dir: Trade数据目录路径
        days: 加载天数（从date列表开头截取）
        date: 日期列表，格式如 ['2025-11-25', '2025-11-26', ...]
    
    Returns:
        合并后的Trade DataFrame
    """
    if date is None:
        date = ['2025-11-25', '2025-11-26', '2025-11-27', '2025-11-28', '2025-11-29', '2025-11-30', '2025-12-01',
                '2025-12-02', '2025-12-03', '2025-12-04', '2025-12-05', '2025-12-06', '2025-12-07', '2025-12-08', 
                '2025-12-09', '2025-12-10']
    
    if days is not None:
        date = date[:days]
    
    all_dfs = []
    for d in date:
        # Trade文件命名格式: ETHUSDT2025-12-01.csv.gz
        filename = f'ETHUSDT{d}.csv.gz'
        filepath = os.path.join(trade_dir, filename)
        
        if os.path.exists(filepath):
            df = pl.read_csv(filepath)
            all_dfs.append(df)
        else:
            print(f"Trade文件不存在: {filepath}")
            continue
    
    if len(all_dfs) == 0:
        raise ValueError("没有找到任何Trade数据文件")
    
    df_all = pl.concat(all_dfs, how='vertical')
    
    # 将timestamp从Float64转换为Int64（毫秒），与LOB对齐
    df_all = df_all.with_columns(
        (pl.col('timestamp')*1000).cast(pl.Int64).alias('timestamp_ms')
    )
    
    return df_all.sort('timestamp_ms', descending=False)


def load_lob_data(data_dir,days = None,date = None,levels = 10):
    """读取parquet数据文件"""
    if date is None:

        date = ['2025-11-25','2025-11-26','2025-11-27','2025-11-28','2025-11-29','2025-11-30','2025-12-01',
                '2025-12-02','2025-12-03','2025-12-04','2025-12-05','2025-12-06','2025-12-07','2025-12-08','2025-12-09','2025-12-10',
        ]
    if days is not None:
        date = date[:days]
    all_dfs = []
    select_cols = [f'a{i}' for i in range(1, levels + 1)]
    select_cols += [f'aq{i}' for i in range(1, levels + 1)]
    select_cols += [f'b{i}' for i in range(1, levels + 1)]
    select_cols += [f'bq{i}' for i in range(1, levels + 1)]

    select_cols += ['timestamp']
    
    for i in date:
        parquet_filename = f'{i}_ETHUSDT_ob_20levels.parquet'
        output_path = os.path.join(data_dir, parquet_filename)
        if os.path.exists(output_path):
            df = pl.read_parquet(output_path)


            df = df.select(
                select_cols
            )
            # df = df.with_columns(
            #     pl.col(pl.Datetime).dt.replace_time_zone(None)
            # )
            all_dfs.append(df)
        else:
            print(f"文件不存在: {output_path}")
            continue

    df_all = pl.concat(all_dfs, how='vertical')
    df_all = df_all.sort('timestamp', descending=False)
    # df_all = df_all.with_columns(
    #     pl.from_epoch("timestamp",time_unit="ms").alias("datetime")
    # )
    ### 时间戳对齐处理
    window_ms = 100
    df_all = df_all.with_columns(
        ((pl.col('timestamp') // window_ms - (pl.col('timestamp') % window_ms <50)+1) * window_ms).alias('time_bucket')
    )
    df_all = df_all.sort('timestamp', descending=False)
    df_all = df_all.unique(subset=['time_bucket'],keep='last',maintain_order=True)
    return df_all