# =============================================================================
# Label Generator - 标签生成模块
# Version: v2.0
# =============================================================================

import numpy as np
from typing import Tuple, Optional, Union
import polars as pl
### 根据数据生成标签
from Data_Pipeline.preprocessors.trade_data_process import align_trade_with_lob

from Data_Pipeline.preprocessors.lob_data_process import generate_channel_data


def generate_trade_labels(trade_data: pl.DataFrame,
                          window=3000,
                          ) -> pl.DataFrame:
    """
    生成trade的标签
    """
    m_previous = pl.col('vwap')
    rolling_vwap = (pl.col('total_notional') * pl.col('has_trade')).rolling_sum(window_size=window) / (pl.col('total_volume').rolling_sum(window_size=window) + 1e-9)
    m_future = rolling_vwap.shift(-window)
    # ret = (m_future - m_previous) / m_previous
    ret = (m_future / m_previous).log()
    trade_data = trade_data.with_columns(
        ret.alias(f'trade_return')
    )
    trade_data = trade_data.drop_nulls(subset=['trade_return'])
    trade_labels = trade_data.select(
        pl.col('time_bucket'),
        pl.col('trade_return')
    )


    return trade_labels


import polars as pl
def generate_data_dict(lob_data_path: str,
                        trade_data_path: str,
                        levels = 10,
                        label_window=3000,

                        need_price: bool = False,
                        need_time: bool = False
                        ) -> dict:
    lob_data = pl.read_parquet(lob_data_path)
    trade_data = pl.read_parquet(trade_data_path)


    ### trade 对齐lob
    trade_data = align_trade_with_lob(trade_data = trade_data,full_lob_data = lob_data)
    ## 生成trade的标签
    trade_labels = generate_trade_labels(trade_data,window=label_window)
    ## 过滤trade和lob数据
    trade_data = trade_data.filter(pl.col('time_bucket').is_in(trade_labels['time_bucket']))
    lob_data = lob_data.filter(pl.col('time_bucket').is_in(trade_labels['time_bucket']))

    ### 生成channel数据
    lob_data = lob_data.sort('time_bucket',descending=False)
    lob_data = generate_channel_data(lob_data,levels = levels)

    trade_labels = trade_labels.sort('time_bucket',descending=False)
    time_bucker = None
    if need_time:
        time_bucker = trade_labels.select(
        pl.col('time_bucket')).to_numpy().flatten()
    trade_labels = trade_labels.select(
        pl.col('trade_return')
    ).to_numpy().flatten()

    
    trade_data = trade_data.sort('time_bucket',descending=False)
    price_data= None
    if need_price:
        price_data = trade_data.select(
            pl.col('price_last')
        ).to_numpy().flatten()

    ## 去除时间戳然后转为numpy
    trade_cols = [
    'has_trade',
    'time_since_last',
    'vwap',
    'total_volume',
    'total_notional',
    'trade_count',
    'buy_notional',
    'sell_notional',
    'price_high',
    'price_low',
    'price_first',
    'price_last',
    'price_std',
    'buy_count',
    'sell_count',
    'net_flow',
    'net_notional',
    'trade_imbalance',
    'notional_imbalance',
    'price_range',
    'price_change',
    'avg_trade_size',
    'trade_count_ratio',
    ]
    trade_data = trade_data.select(trade_cols).to_numpy()
    
    data_dict = {
        'lob':lob_data,
        'trade':trade_data
    }
    return data_dict,trade_labels,price_data,time_bucker







