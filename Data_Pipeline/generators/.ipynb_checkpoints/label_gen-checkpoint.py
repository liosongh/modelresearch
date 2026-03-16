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
def generate_ret_labels(data,
                       k=3000, 
                       change_window=None,
                       mode='smooth'):

    window = k
    if change_window is None:
        m_previous = pl.col('mid_price').rolling_mean(window_size=window)
        m_future = pl.col('mid_price').rolling_mean(window_size=window).shift(-window)
    else:
        m_previous = pl.col('mid_price').rolling_mean(window_size=change_window)
        m_future = pl.col('mid_price').rolling_mean(window_size=window).shift(-window)
    if mode == 'smooth':
        # 计算收益率 return
        ret = (m_future - m_previous) / m_previous
    # ========== 新增 volatility 波动率调整模式 ==========
    elif mode == 'volatility':
        # 1. 计算对数收益率
        log_return = pl.col('mid_price').log().diff()
        
        # 2. 计算收益率的滚动标准差 (波动率)
        volatility = log_return.rolling_std(window_size=window).shift(-window)
        
        # 3. 分子：使用与 smooth 模式一致的收益率计算方式
        # 也就是 (Smoothed_Future - Smoothed_Past) / Smoothed_Past
        ret_raw = (m_future - m_previous) / m_previous
        
        # 4. 计算调整后数值
        # 关键修正：需要除以 sqrt(window) 来标准化累计收益率
        # 否则大窗口的 ret 会显著大于小窗口，导致标签分布不一致
        ret = ret_raw / (volatility * np.sqrt(window) + 1e-9)

    else:
        ret = (m_future - pl.col('mid_price')) / pl.col('mid_price')


    data = data.with_columns(
        ret.alias(f'mid_price_return')
    )
    return data

def generate_trade_labels(trade_data: pl.DataFrame,k=3000,mode='smooth',change_window=None) -> pl.DataFrame:
    window = k
    if change_window is None:
        m_previous = (pl.col('vwap') * pl.col('has_trade')).rolling_sum(window_size=window) / (pl.col('has_trade').rolling_sum(window_size=window) + 1e-9)
        m_future = m_previous.shift(-window)
    else:
        m_previous = (pl.col('vwap') * pl.col('has_trade')).rolling_sum(window_size=change_window) / (pl.col('has_trade').rolling_sum(window_size=change_window) + 1e-9)
        m_future = ((pl.col('vwap') * pl.col('has_trade')).rolling_sum(window_size=window) / (pl.col('has_trade').rolling_sum(window_size=window) + 1e-9)).shift(-window)
    if mode == 'smooth':
        ret = (m_future - m_previous) / m_previous
    elif mode == 'volatility':
        log_return = pl.col('vwap').log().diff()
        volatility = log_return.rolling_std(window_size=window).shift(-window)
        ret_raw = (m_future - m_previous) / m_previous
        ret = ret_raw / (volatility * np.sqrt(window) + 1e-9)
    else:
        ret = (m_future - pl.col('vwap')) / pl.col('vwap')
    trade_data = trade_data.with_columns(
        ret.alias(f'trade_return')
    )
    return trade_data


import polars as pl
def generate_data_dict(lob_data_path: str,trade_data_path: str,levels = 10,k=3000) -> dict:
    lob_data = pl.read_parquet(lob_data_path)
    trade_data = pl.read_parquet(trade_data_path)


    ### trade 对齐lob
    trade_data = align_trade_with_lob(trade_data = trade_data,full_lob_data = lob_data)

    ## 先生成标签,mid_price 和trade的标签不一样，所以需要分开生成
    lob_data = lob_data.with_columns(
        pl.col('time_bucket'),
    ((pl.col('a1') + pl.col('a2')) / 2).alias('mid_price'))
    lob_data = generate_ret_labels(lob_data,k=k,mode='smooth',change_window=None)
    ### lobdata 对齐labels
    lob_data = lob_data.drop_nulls(subset=['mid_price_return'])
    lob_labels = lob_data.select(
        pl.col('mid_price_return')
    ).to_numpy().flatten()
    ### 生成channel数据
    lob_data = generate_channel_data(lob_data,levels = levels)


    ## 生成trade的标签
    trade_data = generate_trade_labels(trade_data,k=k,mode='smooth',change_window=None)
    trade_data = trade_data.drop_nulls(subset=['trade_return'])
    trade_labels = trade_data.select(
        pl.col('trade_return')
    ).to_numpy().flatten()

    ## 去除时间戳然后转为numpy
    trade_cols = [
        'has_trade',
        'log_time_since_last',
        'rel_vwap',
        'rel_price_last',
        'log_total_volume',
        'log_total_notional',
        'log_trade_count',
        'log_avg_trade_size',
        'trade_imbalance',
        'notional_imbalance',
        'trade_count_ratio',
        'rel_price_std'
    ]
    trade_data = trade_data.select(trade_cols).to_numpy()
    
    data_dict = {
        'lob':lob_data,
        'trade':trade_data
    }
    return data_dict,lob_labels,trade_labels










# def compute_future_returns(
#     mid_prices: np.ndarray,
#     horizons: Union[int, list] = 10,
#     method: str = 'simple'
# ) -> np.ndarray:
#     """
#     计算未来收益率。
    
#     Args:
#         mid_prices: 中间价序列 (N,)
#         horizons: 预测horizon，可以是单个整数或列表
#         method: 收益率计算方式
#             - 'simple': (P_{t+k} - P_t) / P_t
#             - 'log': log(P_{t+k} / P_t)
            
#     Returns:
#         returns: (N, num_horizons) 或 (N,) 如果 horizons 是单个值
#     """
#     if isinstance(horizons, int):
#         horizons = [horizons]
#         squeeze = True
#     else:
#         squeeze = False
        
#     N = len(mid_prices)
#     returns = np.zeros((N, len(horizons)), dtype=np.float32)
    
#     for i, h in enumerate(horizons):
#         if method == 'simple':
#             # (P_{t+h} - P_t) / P_t
#             future = np.roll(mid_prices, -h)
#             ret = (future - mid_prices) / (mid_prices + 1e-10)
#         elif method == 'log':
#             # log(P_{t+h} / P_t)
#             future = np.roll(mid_prices, -h)
#             ret = np.log(future / (mid_prices + 1e-10) + 1e-10)
#         else:
#             raise ValueError(f"Unknown method: {method}")
            
#         # 最后 h 个位置设为 NaN（无法计算未来收益）
#         ret[-h:] = np.nan
#         returns[:, i] = ret
        
#     return returns.squeeze() if squeeze else returns


# def generate_labels(
#     returns: np.ndarray,
#     thresholds: Tuple[float, float] = (-0.0002, 0.0002),
#     smoothing: float = 0.0
# ) -> np.ndarray:
#     """
#     根据收益率生成分类标签。
    
#     Args:
#         returns: 收益率数组 (N,)
#         thresholds: (lower, upper) 阈值，定义 Down/Stationary/Up
#         smoothing: 标签平滑系数 (0.0 表示不平滑)
        
#     Returns:
#         labels: (N,) 分类标签，0=Down, 1=Stationary, 2=Up
#     """
#     lower, upper = thresholds
    
#     labels = np.ones(len(returns), dtype=np.int64)  # 默认 1 = Stationary
#     labels[returns < lower] = 0  # Down
#     labels[returns > upper] = 2  # Up
    
#     # 处理 NaN
#     labels[np.isnan(returns)] = 1  # 或设为特殊值
    
#     return labels


# def triple_barrier_labels(
#     mid_prices: np.ndarray,
#     horizon: int = 100,
#     upper_barrier: float = 0.001,
#     lower_barrier: float = -0.001,
#     vertical_barrier: bool = True
# ) -> np.ndarray:
#     """
#     Triple Barrier 标签生成方法 (De Prado 方法)。
    
#     触发条件:
#     1. 价格先触及上界 -> Up (2)
#     2. 价格先触及下界 -> Down (0)
#     3. 到达 horizon 仍未触及任何边界 -> Stationary (1)
    
#     Args:
#         mid_prices: 中间价序列
#         horizon: 最大观察窗口
#         upper_barrier: 上界收益率阈值
#         lower_barrier: 下界收益率阈值
#         vertical_barrier: 是否启用时间边界
        
#     Returns:
#         labels: 三分类标签
#     """
#     N = len(mid_prices)
#     labels = np.ones(N, dtype=np.int64)  # 默认 Stationary
    
#     for i in range(N - 1):
#         entry_price = mid_prices[i]
        
#         for j in range(1, min(horizon + 1, N - i)):
#             current_price = mid_prices[i + j]
#             ret = (current_price - entry_price) / (entry_price + 1e-10)
            
#             if ret >= upper_barrier:
#                 labels[i] = 2  # Up
#                 break
#             elif ret <= lower_barrier:
#                 labels[i] = 0  # Down
#                 break
#         # 如果循环结束仍未触发，保持 Stationary
        
#     return labels


# def smooth_labels(
#     labels: np.ndarray,
#     alpha: float = 0.1,
#     num_classes: int = 3
# ) -> np.ndarray:
#     """
#     标签平滑 (Label Smoothing)。
    
#     将 hard label 转换为 soft label，减少过拟合。
    
#     Args:
#         labels: 硬标签 (N,)
#         alpha: 平滑系数，0 表示不平滑
#         num_classes: 类别数
        
#     Returns:
#         soft_labels: (N, num_classes) 软标签
#     """
#     N = len(labels)
#     soft_labels = np.full((N, num_classes), alpha / num_classes, dtype=np.float32)
    
#     for i, label in enumerate(labels):
#         if not np.isnan(label) and 0 <= label < num_classes:
#             soft_labels[i, int(label)] = 1.0 - alpha + alpha / num_classes
            
#     return soft_labels


# def dynamic_threshold_labels(
#     returns: np.ndarray,
#     volatility: np.ndarray,
#     multiplier: float = 0.5
# ) -> np.ndarray:
#     """
#     基于波动率的动态阈值标签。
    
#     阈值 = multiplier * volatility
#     在高波动时期使用更宽的阈值，低波动时期使用更窄的阈值。
    
#     Args:
#         returns: 收益率序列
#         volatility: 滚动波动率序列 (与 returns 等长)
#         multiplier: 波动率乘数
        
#     Returns:
#         labels: 分类标签
#     """
#     upper = multiplier * volatility
#     lower = -multiplier * volatility
    
#     labels = np.ones(len(returns), dtype=np.int64)
#     labels[returns < lower] = 0
#     labels[returns > upper] = 2
#     labels[np.isnan(returns)] = 1
    
#     return labels