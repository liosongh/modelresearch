# ============================================================
# Trade Data Processing - 将非均匀Tick数据聚合到100ms时间窗口
# ============================================================
import os
import select
import polars as pl
import numpy as np
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


# ============================================================
# 2. 核心聚合函数 - 将Tick数据聚合到100ms窗口
# ============================================================

def aggregate_trade_to_100ms(df_trade: pl.DataFrame, 
                              window_ms: int = 100) -> pl.DataFrame:
    """
    将逐笔交易数据聚合到固定时间窗口（默认100ms）
    
    计算的特征包括:
    - 基础量价特征: total_volume, total_notional, vwap, trade_count
    - 方向性特征: buy_volume, sell_volume, net_flow, trade_imbalance
    - 价格特征: price_high, price_low, price_range, price_std
    - 大单特征: large_trade_volume, large_trade_ratio
    - 高级微观结构特征: buy_notional, sell_notional, net_notional, avg_trade_size
    
    Args:
        df_trade: 原始Trade DataFrame
        window_ms: 时间窗口大小（毫秒），默认100ms
    
    Returns:
        聚合后的DataFrame，每行代表一个时间窗口
    """
    df_trade = df_trade if df_trade is pl.LazyFrame else df_trade.lazy()
    # 创建时间窗口标签（向下取整到window_ms）
    df_trade = df_trade.with_columns(
        (pl.col('timestamp_ms') // window_ms * window_ms).alias('time_bucket')
    )
    
    # 添加辅助列
    df_trade = df_trade.with_columns([
        # 成交金额
        (pl.col('price') * pl.col('size')).alias('notional'),
        # 买入标志
        (pl.col('side') == 'Buy').cast(pl.Int32).alias('is_buy'),
        # 卖出标志
        (pl.col('side') == 'Sell').cast(pl.Int32).alias('is_sell'),
        # 大单判断（大于平均的2倍，后面会用rolling计算更精确的阈值）
        # 这里先用简单阈值，实际使用时可以动态调整
    ])
    
    # # 计算全局大单阈值（使用整体的75分位数作为大单标准）
    # large_threshold = df_trade.select(pl.col('size').quantile(0.75)).item()
    
    # df_trade = df_trade.with_columns(
    #     (pl.col('size') > large_threshold).cast(pl.Int32).alias('is_large_trade')
    # )
    
    # 按时间窗口聚合
    df_agg = df_trade.group_by('time_bucket').agg([
        # ========== 基础量价特征 ==========
        # 总成交量 (ETH)
        pl.col('size').sum().alias('total_volume'),
        # 总成交额 (USDT)
        pl.col('foreignNotional').sum().alias('total_notional'),
        # 交易笔数
        pl.col('size').count().alias('trade_count'),
        
        # ========== VWAP (Volume Weighted Average Price) ==========
        # 成交量加权平均价
        (pl.col('notional').sum() / pl.col('size').sum()).alias('vwap'),
        
        # ========== 方向性特征 (Directional Features) ==========
        # 主动买入量
        (pl.col('size') * pl.col('is_buy')).sum().alias('buy_volume'),
        # 主动卖出量
        (pl.col('size') * pl.col('is_sell')).sum().alias('sell_volume'),
        # 主动买入金额
        (pl.col('foreignNotional') * pl.col('is_buy')).sum().alias('buy_notional'),
        # 主动卖出金额
        (pl.col('foreignNotional') * pl.col('is_sell')).sum().alias('sell_notional'),
        
        # ========== 价格特征 ==========
        # 最高价
        pl.col('price').max().alias('price_high'),
        # 最低价
        pl.col('price').min().alias('price_low'),
        # 首笔价格
        pl.col('price').first().alias('price_first'),
        # 末笔价格
        pl.col('price').last().alias('price_last'),
        # 价格标准差
        pl.col('price').std().alias('price_std'),
        
        # # ========== 大单特征 ==========
        # # 大单成交量
        # (pl.col('size') * pl.col('is_large_trade')).sum().alias('large_trade_volume'),
        # # 大单笔数
        # pl.col('is_large_trade').sum().alias('large_trade_count'),
        
        # ========== 买卖单独统计（用于计算更多指标） ==========
        # 买入笔数
        pl.col('is_buy').sum().alias('buy_count'),
        # 卖出笔数
        pl.col('is_sell').sum().alias('sell_count'),
    ])
    
    # 计算派生特征
    df_agg = df_agg.with_columns([
        # ========== 净流量 (Net Flow) - 核心Alpha特征 ==========
        # 净成交量（买-卖）
        (pl.col('buy_volume') - pl.col('sell_volume')).alias('net_flow'),
        # 净成交额
        (pl.col('buy_notional') - pl.col('sell_notional')).alias('net_notional'),
        
        # ========== Trade Imbalance (交易失衡度) ==========
        # 量失衡: (买量-卖量)/(买量+卖量)，范围[-1, 1]
        ((pl.col('buy_volume') - pl.col('sell_volume')) / 
         (pl.col('buy_volume') + pl.col('sell_volume') + 1e-9)).alias('trade_imbalance'),
        
        # 金额失衡
        ((pl.col('buy_notional') - pl.col('sell_notional')) / 
         (pl.col('buy_notional') + pl.col('sell_notional') + 1e-9)).alias('notional_imbalance'),
        
        # ========== 价格变动 ==========
        # 窗口内价格范围
        (pl.col('price_high') - pl.col('price_low')).alias('price_range'),
        # 窗口内价格变动（末-首）
        (pl.col('price_last') - pl.col('price_first')).alias('price_change'),
        
        # # ========== 大单比例 ==========
        # # 大单成交量占比
        # (pl.col('large_trade_volume') / (pl.col('total_volume') + 1e-9)).alias('large_trade_ratio'),
        
        # ========== 平均单笔大小 ==========
        (pl.col('total_volume') / (pl.col('trade_count') + 1e-9)).alias('avg_trade_size'),
        
        # ========== 交易频率相关 ==========
        # 买卖笔数比
        (pl.col('buy_count') / (pl.col('sell_count') + 1e-9)).alias('trade_count_ratio'),
    ])
    df_agg = df_agg.sort('time_bucket',descending=False).with_columns(
        pl.lit(1).alias('has_trade'),
        pl.col('time_bucket').diff().fill_null(0).alias('time_since_last')
    )
    select_cols = [
                    'time_bucket',  ## 时间窗口
                    'has_trade',  ## 是否有真实交易
                    'time_since_last',  ## 距离上次交易的时间
                    'total_volume',  ## 总成交量
                   'total_notional',  ## 总成交额
                    'buy_notional',  ## 主动买入金额
                   'sell_notional',  ## 主动卖出金额
                   'trade_count',  ## 交易笔数
                    'buy_count',  ## 买入笔数
                   'sell_count',  ## 卖出笔数
                   'vwap',  ## 成交量加权平均价
                #    'buy_volume',  ## 主动买入量
                #    'sell_volume',  ## 主动卖出量

                   'price_high',  ## 最高价
                   'price_low',  ## 最低价
                   'price_first',  ## 首笔价格
                   'price_last',  ## 末笔价格
                   'price_std',  ## 价格标准差

                   'net_flow',  ## 净流量
                   'net_notional',  ## 净成交额
                   'trade_imbalance',  ## 量失衡
                   'notional_imbalance',  ## 金额失衡
                   'price_range',  ## 价格范围
                   'price_change',  ## 价格变动
                   'avg_trade_size',  ## 平均单笔大小
                   'trade_count_ratio',  ## 买卖笔数比
                   
    ]
    return df_agg.select(select_cols).sort('time_bucket').collect()


# ============================================================
# 3. 高级微观结构特征（滚动窗口特征）
# ============================================================

def add_rolling_features(df_agg: pl.DataFrame, 
                         windows: List[int] = [5, 10, 30, 60]) -> pl.DataFrame:
    """
    添加滚动窗口特征，捕捉短期趋势和动量
    
    Args:
        df_agg: 聚合后的DataFrame
        windows: 滚动窗口大小列表（以100ms为单位，如10表示1秒）
    
    Returns:
        添加滚动特征后的DataFrame
    """
    
    for w in windows:
        df_agg = df_agg.with_columns([
            # 累计净流量（短期订单流动量）
            pl.col('net_flow').rolling_sum(window_size=w).alias(f'net_flow_sum_{w}'),
            
            # 累计成交量
            pl.col('total_volume').rolling_sum(window_size=w).alias(f'volume_sum_{w}'),
            
            # 平均失衡度
            pl.col('trade_imbalance').rolling_mean(window_size=w).alias(f'imbalance_ma_{w}'),
            
            # VWAP变化率
            (pl.col('vwap') / pl.col('vwap').shift(w) - 1).alias(f'vwap_return_{w}'),
            
            # 成交量变化（当前vs过去平均）
            (pl.col('total_volume') / (pl.col('total_volume').rolling_mean(window_size=w) + 1e-9)).alias(f'volume_ratio_{w}'),
            
            # 交易频率变化
            (pl.col('trade_count') / (pl.col('trade_count').rolling_mean(window_size=w) + 1e-9)).alias(f'trade_freq_ratio_{w}'),
        ])
    
    return df_agg


# ============================================================
# 4. 与LOB数据对齐
# ============================================================

def align_trade_with_lob(trade_data: pl.DataFrame, 
                         full_lob_data: pl.DataFrame,
                         ) -> pl.DataFrame:
    """
    将聚合后的Trade数据与LOB数据对齐
    
    LOB数据是均匀的100ms切片，Trade数据在某些时间窗口可能没有成交。
    此函数确保Trade特征与LOB的timestamp完全对齐。
    
    Args:
        df_trade_agg: 聚合后的Trade DataFrame
        df_lob: LOB DataFrame（需包含timestamp列）
        fill_strategy: 缺失值填充策略
            - 'forward': 前向填充（用上一个有效值）
            - 'zero': 填充0
            - 'interpolate': 线性插值
    
    Returns:
        与LOB对齐的Trade特征DataFrame
    """
        
    # 尝试识别 Bid/Ask 列名 (兼容常见命名)
    ap_col = 'a1'
    bp_col = 'b1'
    
    # 既然我们要计算相对价格，必须保留价格列
    cols_to_keep = ['time_bucket']
    # if ap_col in lob_cols and bp_col in lob_cols:
    #     cols_to_keep.extend([ap_col, bp_col])
    cols_to_keep.extend([ap_col, bp_col])
    full_trade_data = full_lob_data.select(
        cols_to_keep
    ).join(trade_data,on='time_bucket',how='left')
    ## 增加 ：距上次交易的时间间隔
    full_trade_data = full_trade_data.with_columns(
        pl.col('has_trade').fill_null(0),
    ).with_columns(
        # 步骤1：仅保留有交易行的timestamp，无交易行设为null
        pl.when(pl.col("has_trade") ==1)
        .then(pl.col("time_bucket"))
        .otherwise(None)
        .alias("last_trade_timestamp"),
    ).with_columns(
        # 步骤2：向前填充最近一次交易的timestamp（核心）
        pl.col("last_trade_timestamp")
        .forward_fill()
        .alias("filled_last_trade_ts"),
    ).with_columns(
        # 步骤3：计算时间间隔（当前时间 - 最近交易时间）
        (pl.col("time_bucket") - pl.col("filled_last_trade_ts"))
        .alias("time_since_last")
    ).drop(["last_trade_timestamp", "filled_last_trade_ts"]) 

    fill_zero_cols = [
                    'total_volume',  ## 总成交量
                    'total_notional',  ## 总成交额
                    'trade_count',  ## 交易笔数
                    #    'vwap',  ## 成交量加权平均价
                    # 'buy_volume',  ## 主动买入量
                    # 'sell_volume',  ## 主动卖出量
                    'buy_notional',  ## 主动买入金额
                    'sell_notional',  ## 主动卖出金额
                    #    'price_high',  ## 最高价
                    #    'price_low',  ## 最低价
                    #    'price_first',  ## 首笔价格
                    #    'price_last',  ## 末笔价格
                    'price_std',  ## 价格标准差
                    'buy_count',  ## 买入笔数
                    'sell_count',  ## 卖出笔数
                    'net_flow',  ## 净流量
                    'net_notional',  ## 净成交额
                    'trade_imbalance',  ## 量失衡
                    'notional_imbalance',  ## 金额失衡
                    'price_range',  ## 价格范围
                    'price_change',  ## 价格变动
                    'avg_trade_size',  ## 平均单笔大小
                    #    'trade_count_ratio',  ## 买卖笔数比
                    
        ]
    fill_one_cols = ['trade_count_ratio']
    fill_forward_cols = ['price_last']
    full_trade_data = full_trade_data.with_columns(
        [pl.col(col).fill_null(0) for col in fill_zero_cols] +
        [pl.col(col).fill_null(1) for col in fill_one_cols]+
        [pl.col(col).forward_fill() for col in fill_forward_cols]
    )
    # 最高价,最低价,首笔价格,成交量价格使用last
    full_trade_data = full_trade_data.with_columns(
        pl.col('price_high').fill_null(pl.col('price_last')),
        pl.col('price_low').fill_null(pl.col('price_last')),
        pl.col('price_first').fill_null(pl.col('price_last')),
        pl.col('vwap').fill_null(pl.col('price_last')),
    )

    ## 第一行数据填充为0
    full_trade_data = full_trade_data.fill_null(0)

    # ============================================================
    # 新增: 核心特征构造 (相对价格 & Log变换)
    # ============================================================
    
    # 1. 计算 LOB Mid Price
    try:
        full_trade_data = full_trade_data.with_columns(
            ((pl.col(ap_col) + pl.col(bp_col)) / 2).alias('mid_price')
        )
    except:
        print("Warning: LOB columns not found, using price_last as mid_price proxy.")


    # 2. 构造相对特征和 Log 特征
    full_trade_data = full_trade_data.with_columns([
        # --- 相对价格特征 (解决非平稳性) ---
        # 相对 VWAP: (VWAP - Mid) / Mid
        ((pl.col('vwap') - pl.col('mid_price')) / (pl.col('mid_price') + 1e-9)).alias('rel_vwap'),
        
        # 相对 Last Price: (Last - Mid) / Mid
        ((pl.col('price_last') - pl.col('mid_price')) / (pl.col('mid_price') + 1e-9)).alias('rel_price_last'),
        
        # --- Log 变换 (解决长尾分布) ---
        # Log Volume
        (pl.col('total_volume') + 1).log().alias('log_total_volume'),
        # Log Notional
        (pl.col('total_notional') + 1).log().alias('log_total_notional'),
        # Log Count
        (pl.col('trade_count') + 1).log().alias('log_trade_count'),
        # Log Avg Trade Size
        (pl.col('avg_trade_size') + 1).log().alias('log_avg_trade_size'),
        # Log Time Since Last
        (pl.col('time_since_last') + 1).log().alias('log_time_since_last'),
        
        # Relative Price Std (Volatility)
        (pl.col('price_std') / (pl.col('mid_price') + 1e-9)).alias('rel_price_std'),
    ])
    
    # # 3. 清理 NaN/Inf (由除 0 产生)
    # full_trade_data = full_trade_data.with_columns([
    #     pl.col('rel_vwap').fill_nan(0)
    #     pl.col('rel_price_last').fill_nan(0)
    #     pl.col('rel_price_std').fill_nan(0)
    # ])
    # 基础特征
    base_features = [
        'time_bucket',
        # --- 核心特征 (经过专家修正) ---
        'has_trade',          # 0/1 状态
        'log_time_since_last',# 稀疏性特征 (Log)
        'vwap',
        # 相对价格 (Stationary)
        'rel_vwap',
        'rel_price_last',
        
        # Log Volumes (Normalized Distribution)
        'log_total_volume',
        'log_total_notional',
        'log_trade_count',
        'log_avg_trade_size',
        
        # Microstructure Ratios (Already normalized)
        'trade_imbalance',
        'notional_imbalance',
        'trade_count_ratio',
        'rel_price_std',     # 相对波动率 (已修正)
        
        # --- 以下绝对值特征已被移除/替换 ---
        # 'total_volume', 'total_notional', 'trade_count', 'vwap',
        # 'price_high', 'price_low', 'price_first', 'price_last',
        # 'buy_volume', 'sell_volume', 'net_flow', 
        # 'buy_notional', 'sell_notional', 'net_notional',
    ]
    full_trade_data = full_trade_data.select(base_features)
    # full_trade_data.write_parquet('/root/autodl-tmp/Trade_ETHUSDT_train_agg.parquet')
    return full_trade_data


def process_trade_data(data_dir: str,date: list[str],window_ms: int = 100) -> pl.DataFrame:
    # date = ['2025-11-04','2025-11-05','2025-11-06','2025-11-07','2025-11-08','2025-11-09','2025-11-10',
    #         '2025-11-11','2025-11-12','2025-11-13','2025-11-14','2025-11-15','2025-11-16','2025-11-17',
    #         '2025-11-18','2025-11-19','2025-11-20','2025-11-21','2025-11-22','2025-11-23','2025-11-24',
    #         '2025-11-25','2025-11-26','2025-11-27','2025-11-28','2025-11-29','2025-11-30','2025-12-01',
    #         '2025-12-02','2025-12-03','2025-12-04','2025-12-05','2025-12-06','2025-12-07',
    #         ]

    trade_data = load_trade_data(data_dir,date=date)
    # trade_data.write_parquet('/root/autodl-tmp/Trade_ETHUSDT_train.parquet')

    # ### 获取原始的逐笔trade数据
    # trade_data = pl.read_parquet('/root/autodl-tmp/Trade_ETHUSDT_train.parquet')
    ### 聚合数据
    trade_data = aggregate_trade_to_100ms(trade_data,window_ms=window_ms)
    # trade_data.write_parquet('/root/autodl-tmp/Trade_ETHUSDT_train_agg100ms.parquet')
    return trade_data



