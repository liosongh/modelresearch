# ============================================================
# 1. 读取数据 (与cnn.py相同)
# ============================================================
import os
import polars as pl
import numpy as np


def load_lob_data(data_dir,days = None,date = None,levels = 10):
    """读取parquet数据文件,并把事件戳转换为固定整数的100ms时间戳"""
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
    # df_all = df_all.sort('timestamp', descending=False)
    df_all = df_all.unique(subset=['time_bucket'],keep='last',maintain_order=True)
    df_all = df_all.drop('timestamp')
    return df_all


## 加载原始数据
def fullfill_lob_data(lob_data: pl.DataFrame) -> pl.DataFrame:

    # lob_data = load_lob_data(data_dir,date=date,levels=levels)
    ## 添加has_lob列，表示是否存在LOB数据
    lob_data = lob_data.with_columns(
        pl.lit(1).alias('has_lob')
    )

    ## 对齐100ms时间，对齐方式为前向填充
    full_time_range = pl.DataFrame(range(lob_data['time_bucket'].min(),lob_data['time_bucket'].max()+100,100),schema=['time_bucket'])

    lob_data = full_time_range.join(lob_data,on='time_bucket',how='left')
    ## 填充has_lob列
    lob_data = lob_data.with_columns(
        pl.col('has_lob').fill_null(0)
    )
    ## 填充LOB数据
    lob_data = lob_data.fill_null(strategy="forward")
    # lob_data.write_parquet('/root/autodl-tmp/LOB_ETHUSDT_train.parquet')
    return lob_data



# def generate_channel_data(data: pl.DataFrame,levels = 10) -> pl.DataFrame:
#     ## 生成X
#     channel_list = []
#     expr_a_notional = [(pl.col(f"a{i}") * pl.col(f"aq{i}")).log1p().alias(f'a_notional_{i}') for i in range(1, levels + 1)]
#     expr_b_notional = [(pl.col(f"b{i}") * pl.col(f"bq{i}")).log1p().alias(f'b_notional_{i}') for i in range(1, levels + 1)]
#     data = data.with_columns(expr_a_notional+expr_b_notional)
#     ##
#     c1_cols =  [f"a{i}" for i in range(1, levels + 1)] 
#     c2_cols = [f"aq{i}" for i in range(1, levels + 1)]
#     c3_cols = [f"b{i}" for i in range(1, levels + 1)]
#     c4_cols = [f"bq{i}" for i in range(1, levels + 1)]
#     c5_cols = [f'a_notional_{i}' for i in range(1, levels + 1)]
#     c6_cols = [f'b_notional_{i}' for i in range(1, levels + 1)]
#     # c7_cols = [f'b_notional_{i}' for i in range(1, levels + 1)]
#     # c8_cols = [f'b_notional_{i}' for i in range(1, levels + 1)]
    
#     c1 = data.select(c1_cols).to_numpy()
#     channel_list.append(c1)

#     c3 = data.select(c3_cols).to_numpy()
#     channel_list.append(c3)

#     c5 = data.select(c5_cols).to_numpy()
#     channel_list.append(c5)
#     c6 = data.select(c6_cols).to_numpy()
#     channel_list.append(c6)


#     ## 原始特征的进入模型
#     X = np.stack(channel_list, axis=1)
#     return X

def generate_channel_data(data: pl.DataFrame,levels = 10) -> pl.DataFrame:
    ## 生成X
    cols = []
    for i in range(1,levels+1):
        cols.append(f'a{i}')
        cols.append(f'aq{i}')
        cols.append(f'b{i}')
        cols.append(f'bq{i}')
    X = data.select(cols).to_numpy()
    return X


def compute_ofi(price,volume):
    """
    price: [T, L]
    volume: [T, L]
    L 包含 bid 和 ask 两个部分。
    假设前 L//2 是 Bid (从高到低)，后 L//2 是 Ask (从低到高)。
    """
    L = price.shape[1]
    half_L = L // 2
    # 计算时间差分 (t - (t-1))
    p_diff = price[1:, :] - price[:-1, :]
    v_diff = volume[1:, :] - volume[:-1, :]

    # 初始化 OFI 张量 [B, T-1, L]
    # 对 Bid 端的逻辑
    L = 20
    half_L = 10
    bid_p_diff = p_diff[:, :half_L]
    bid_v_diff = v_diff[:, :half_L]

    # Bid OFI 逻辑简化实现
    bid_ofi = torch.where(bid_p_diff > 0, volume[1:, :half_L],
                torch.where(bid_p_diff < 0, -volume[:-1, :half_L], bid_v_diff))

    # Ask OFI 逻辑 (注意 Ask 价格上涨通常意味着卖压减弱，逻辑与 Bid 相反)
    ask_p_diff = p_diff[:, half_L:]
    ask_v_diff = v_diff[:, half_L:]

    ask_ofi = torch.where(ask_p_diff < 0, volume[1:, half_L:],
                torch.where(ask_p_diff > 0, -volume[:-1, half_L:], ask_v_diff))

    # 拼接并补齐第一帧 (Padding)
    ofi = torch.cat([bid_ofi, ask_ofi], dim=-1)
    padding = torch.zeros(1, L)
    ofi = torch.cat([padding, ofi], dim=0) 
    
    return ofi # [T, L]


import torch
## 原来的数据是 ask price，ask vol，bid price，bid vol
def level_wise_encoder_transform(lob_data):
    """
    lob_data: (N, 40) 
    排列: [ap1, av1, bp1, bv1, ap2, av2, bp2, bv2, ..., ap10, av10, bp10, bv10]
    返回: (N, 20, 5) -> 
    Channel 0: Price (bid10...bid1, ask1...ask10), 
    Channel 1: Volume, 
    Channel 2: Price Difference, 
    Channel 3: Relative Price, 
    Channel 4: OFI
    """
    # 如果是 numpy 则转为 tensor 处理更方便（或者直接用 numpy 索引）
    if isinstance(lob_data, np.ndarray):
        lob_data = torch.from_numpy(lob_data)

    N,L = lob_data.shape
    # 1. 提取所有的 Ask 和 Bid
    # 原始索引中：0, 4, 8... 是 ask_p; 1, 5, 9... 是 ask_v
    #            2, 6, 10... 是 bid_p; 3, 7, 11... 是 bid_v
    ask_p = lob_data[:, 0::4]  # (N, 10) [ap1, ap2, ..., ap10]
    ask_v = lob_data[:, 1::4]  # (N, 10)
    bid_p = lob_data[:, 2::4]  # (N, 10) [bp1, bp2, ..., bp10]
    bid_v = lob_data[:, 3::4]  # (N, 10)
    mid = (ask_p[:, 0] + bid_p[:, 0]) / 2.0
    
    # 2. 按照 [bid10...bid1, ask1...ask10] 顺序重排
    # flip(1) 将 [p1, p2...p10] 变为 [p10, p9...p1]
    price_channel = torch.cat([bid_p.flip(dims=[1]), ask_p], dim=1)  # (N, 20)
    vol_channel = torch.cat([bid_v.flip(dims=[1]), ask_v], dim=1)    # (N, 20)

    # 计算价差
    price_diff = price_channel[1:, :] - price_channel[:-1, :]
    padding = torch.zeros(1, L)
    price_diff = torch.cat([padding, price_diff], dim=0) 

    # 3. 计算相对价格
    rel_price = (price_channel - mid.unsqueeze(2))
    # 4. 计算 OFI
    ofi = compute_ofi(rel_price, vol_channel)

    # 3. 堆叠成 (N, 4, 20)
    output = torch.stack([price_channel, vol_channel,price_diff, rel_price, ofi], dim=-1)
    
    return output

def lob_encoder_transform(lob_data):
    """
    lob_data: (N, 40) 
    排列: [ap1, av1, bp1, bv1, ap2, av2, bp2, bv2, ..., ap10, av10, bp10, bv10]
    返回: (N, 2, 20) -> 
    Channel 0: Price [ap1, bp1, ap2, bp2, ..., ap10, bp10],
    Channel 1: Volume [av1, bv1, av2, bv2, ..., av10, bv10],
    """
    # 如果是 numpy 则转为 tensor 处理更方便（或者直接用 numpy 索引）
    if isinstance(lob_data, np.ndarray):
        lob_data = torch.from_numpy(lob_data)

    N,L = lob_data.shape
    # 1. 提取price和volume

    price = lob_data[:, 0::2]  # (N, 10) [ap1,bp1,ap2,bp2,...,ap10,bp10]
    volume = lob_data[:, 1::2]  # (N, 10) [av1,bv1,av2,bv2,...,av10,bv10]

    # 3. 堆叠成 (N, 2, 20)
    output = torch.stack([price, volume], dim=1)
    
    return output

def deeplob_encoder_transform(lob_data):
    """
    lob_data: (N, 40) 
    排列: [ap1, av1, bp1, bv1, ap2, av2, bp2, bv2, ..., ap10, av10, bp10, bv10]
    返回: (N, 1, 40) 
    
    """
    if isinstance(lob_data, np.ndarray):
        lob_data = torch.from_numpy(lob_data)
    lob_data = lob_data.unsqueeze(1)
    return lob_data

def transform_lob_data(lob_data,model_name):
    if model_name == 'level_wise_encoder':
        return level_wise_encoder_transform(lob_data)
    elif model_name == 'lob_encoder':
        return lob_encoder_transform(lob_data)
    elif model_name == 'deeplob_encoder':
        return deeplob_encoder_transform(lob_data)
    else:
        raise ValueError(f"Invalid model name: {model_name}")



def generate_labels(lob_data: pl.DataFrame,
                    windows=[100,300,600,3000,6000],
                          ) -> pl.DataFrame:
    """
    生成标签
    """
    ## 计算Weighted Mid-price
    lob_data = lob_data.with_columns (
        ((pl.col('a1') * pl.col('bq1') + pl.col('b1') * pl.col('aq1')) 
        / (pl.col('aq1') + pl.col('bq1'))).alias('weighted_mid_price')
    )

    for idx,window in enumerate(windows):

        m_previous = pl.col('weighted_mid_price')
        m_future = (pl.col('weighted_mid_price').rolling_mean(window_size=window)).shift(-window)
        # ret = (m_future - m_previous) / m_previous
        return_label = (m_future / m_previous).log()
        lob_data = lob_data.with_columns(
            return_label.alias(f'return_label_{idx}')
        )

    ## Triple Barrier Method
    ## 在未来 10 分钟内，
    # 只要价格在任何时刻触碰到了预设的上涨阈值（比如 +1%），标签立即设为 1 ，
    # 如果到达下跌阈值（比如 -1%），标签立即设为 -1 ，
    # 如果价格在 10 分钟内没有触碰任何阈值，标签设为 0
    # lob_data = lob_data.with_columns(
    #     (pl.col('weighted_mid_price').shift(-1) / pl.col('weighted_mid_price')).log1p().alias('return')
    # ).with_columns(
    #     pl.when(pl.col('return') > 0.01).then(1).when(pl.col('return') < -0.01).then(-1).otherwise(0).alias('Triple_label')
    # )
    lob_data = lob_data.drop_nulls()


    # lob_labels = lob_data.select(
    #     pl.col('time_bucket'),
    #     pl.col('return_label')
    # )
    return lob_data