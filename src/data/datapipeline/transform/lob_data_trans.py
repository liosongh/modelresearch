import torch
import numpy as np
import polars as pl
# def compute_ofi(price,volume):
#     """
#     price: [T, L]
#     volume: [T, L]
#     L 包含 bid 和 ask 两个部分。
#     假设前 L//2 是 Bid (从高到低)，后 L//2 是 Ask (从低到高)。
#     """
#     L = price.shape[1]
#     half_L = L // 2
#     # 计算时间差分 (t - (t-1))
#     p_diff = price[1:, :] - price[:-1, :]
#     v_diff = volume[1:, :] - volume[:-1, :]

#     # 初始化 OFI 张量 [B, T-1, L]
#     # 对 Bid 端的逻辑
#     L = 20
#     half_L = 10
#     bid_p_diff = p_diff[:, :half_L]
#     bid_v_diff = v_diff[:, :half_L]

#     # Bid OFI 逻辑简化实现
#     bid_ofi = torch.where(bid_p_diff > 0, volume[1:, :half_L],
#                 torch.where(bid_p_diff < 0, -volume[:-1, :half_L], bid_v_diff))

#     # Ask OFI 逻辑 (注意 Ask 价格上涨通常意味着卖压减弱，逻辑与 Bid 相反)
#     ask_p_diff = p_diff[:, half_L:]
#     ask_v_diff = v_diff[:, half_L:]

#     ask_ofi = torch.where(ask_p_diff < 0, volume[1:, half_L:],
#                 torch.where(ask_p_diff > 0, -volume[:-1, half_L:], ask_v_diff))

#     # 拼接并补齐第一帧 (Padding)
#     ofi = torch.cat([bid_ofi, ask_ofi], dim=-1)
#     padding = torch.zeros(1, L)
#     ofi = torch.cat([padding, ofi], dim=0) 
    
#     return ofi # [T, L]

import numpy as np

def compute_ofi(price, volume):
    """
    price: [T, L]
    volume: [T, L]
    L = 20 (前10是bid，后10是ask)
    """

    T, L = price.shape
    half_L = L // 2

    # 时间差分
    p_diff = price[1:, :] - price[:-1, :]
    v_diff = volume[1:, :] - volume[:-1, :]

    # --- Bid side ---
    bid_p_diff = p_diff[:, :half_L]
    bid_v_diff = v_diff[:, :half_L]

    bid_ofi = np.where(
        bid_p_diff > 0,
        volume[1:, :half_L],
        np.where(
            bid_p_diff < 0,
            -volume[:-1, :half_L],
            bid_v_diff
        )
    )

    # --- Ask side ---
    ask_p_diff = p_diff[:, half_L:]
    ask_v_diff = v_diff[:, half_L:]

    ask_ofi = np.where(
        ask_p_diff < 0,
        volume[1:, half_L:],
        np.where(
            ask_p_diff > 0,
            -volume[:-1, half_L:],
            ask_v_diff
        )
    )

    # 拼接
    ofi = np.concatenate([bid_ofi, ask_ofi], axis=-1)

    # padding 第一帧
    padding = np.zeros((1, L), dtype=price.dtype)
    ofi = np.concatenate([padding, ofi], axis=0)

    return ofi  # [T, L]


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
    # if isinstance(lob_data, np.ndarray):
    #     lob_data = torch.from_numpy(lob_data)


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
    price_channel = np.concatenate([np.flip(bid_p, axis=1), ask_p], axis=1)  # (N, 20)
    vol_channel = np.concatenate([np.flip(bid_v, axis=1), ask_v], axis=1)    # (N, 20)

    # 计算价差
    price_diff = np.zeros_like(price_channel)
    price_diff[1:] = price_channel[1:] - price_channel[:-1]


    # 3. 计算相对价格
    rel_price = price_channel - mid[:, None]
    # 4. 计算 OFI
    ofi = compute_ofi(price_channel, vol_channel)

    # 3. 堆叠成 (N, 5, 20)
    output = np.stack(
        [price_channel, vol_channel, price_diff, rel_price, ofi],
        axis=1
    )

    ## 转为num
    
    return output

def lob_encoder_transform(lob_data):
    """
    lob_data: (N, 40) 
    排列: [ap1, av1, bp1, bv1, ap2, av2, bp2, bv2, ..., ap10, av10, bp10, bv10]
    返回: (N, 2, 20) -> 
    Channel 0: Price [ap1, bp1, ap2, bp2, ..., ap10, bp10],
    Channel 1: Volume [av1, bv1, av2, bv2, ..., av10, bv10],
    """
    # 1. 提取price和volume

    price = lob_data[:, 0::2]  # (N, 10) [ap1,bp1,ap2,bp2,...,ap10,bp10]
    volume = lob_data[:, 1::2]  # (N, 10) [av1,bv1,av2,bv2,...,av10,bv10]

    # 3. 堆叠成 (N, 2, 20)
    output = np.stack([price, volume], axis=1)
    
    return output

def deeplob_encoder_transform(lob_data):
    """
    lob_data: (N, 40) 
    排列: [ap1, av1, bp1, bv1, ap2, av2, bp2, bv2, ..., ap10, av10, bp10, bv10]
    返回: (N, 1, 40) 
    
    """

    lob_data = lob_data[:,np.newaxis,...] # (N, 1, 40)
    return lob_data

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
def transform_lob_data(lob_data,lob_encoder_name,levels = 10):
    """
    lob_data: (N, 40) 
    排列: [ap1, av1, bp1, bv1, ap2, av2, bp2, bv2, ..., ap10, av10, bp10, bv10]
    返回: (N, 1, 40) 
    """
    lob_data = generate_channel_data(lob_data,levels = levels)

    if lob_encoder_name == 'level_wise_encoder':
        return level_wise_encoder_transform(lob_data)
    elif lob_encoder_name == 'lob_encoder':
        return lob_encoder_transform(lob_data)
    elif lob_encoder_name == 'Deeplob_encoder_simple' or lob_encoder_name == 'DeepLOB_encoder':
        return deeplob_encoder_transform(lob_data)
    else:
        raise ValueError(f"Invalid model name: {lob_encoder_name}")