# ============================================================
# 1. 读取数据 (与cnn.py相同)
# ============================================================
import os
import polars as pl
import numpy as np


## 加载原始数据
def process_lob_data(data_dir: str,date: list[str],levels: int = 10) -> pl.DataFrame:
    # date = ['2025-11-04','2025-11-05','2025-11-06','2025-11-07','2025-11-08','2025-11-09','2025-11-10',
    #         '2025-11-11','2025-11-12','2025-11-13','2025-11-14','2025-11-15','2025-11-16','2025-11-17',
    #         '2025-11-18','2025-11-19','2025-11-20','2025-11-21','2025-11-22','2025-11-23','2025-11-24',
    #         '2025-11-25','2025-11-26','2025-11-27','2025-11-28','2025-11-29','2025-11-30','2025-12-01',
    #         '2025-12-02','2025-12-03','2025-12-04','2025-12-05','2025-12-06','2025-12-07',
    #         ]

    lob_data = load_lob_data(data_dir,date=date,levels=levels)

    ## 对齐100ms时间，对齐方式为前向填充
    full_time_range = pl.DataFrame(range(lob_data['time_bucket'].min(),lob_data['time_bucket'].max()+100,100),schema=['time_bucket'])
    lob_data = full_time_range.join(lob_data,on='time_bucket',how='left')
    lob_data = lob_data.fill_null(strategy="forward")
    # lob_data.write_parquet('/root/autodl-tmp/LOB_ETHUSDT_train.parquet')
    return lob_data



def generate_channel_data(data: pl.DataFrame,levels = 10) -> pl.DataFrame:
    ##     ## 生成X
    channel_list = []
    expr_a_notional = [(pl.col(f"a{i}") * pl.col(f"aq{i}")).log1p().alias(f'a_notional_{i}') for i in range(1, levels + 1)]
    expr_b_notional = [(pl.col(f"b{i}") * pl.col(f"bq{i}")).log1p().alias(f'b_notional_{i}') for i in range(1, levels + 1)]
    data = data.with_columns(expr_a_notional+expr_b_notional)
    ##
    c1_cols =  [f"a{i}" for i in range(1, levels + 1)] 
    c2_cols = [f"aq{i}" for i in range(1, levels + 1)]
    c3_cols = [f"b{i}" for i in range(1, levels + 1)]
    c4_cols = [f"bq{i}" for i in range(1, levels + 1)]
    c5_cols = [f'a_notional_{i}' for i in range(1, levels + 1)]
    c6_cols = [f'b_notional_{i}' for i in range(1, levels + 1)]
    # c7_cols = [f'b_notional_{i}' for i in range(1, levels + 1)]
    # c8_cols = [f'b_notional_{i}' for i in range(1, levels + 1)]
    
    c1 = data.select(c1_cols).to_numpy()
    channel_list.append(c1)
    # c2 = data.select(c2_cols).to_numpy()
    # channel_list.append(c2)
    c3 = data.select(c3_cols).to_numpy()
    channel_list.append(c3)
    # c4 = data.select(c4_cols).to_numpy()
    # channel_list.append(c4)
    c5 = data.select(c5_cols).to_numpy()
    channel_list.append(c5)
    c6 = data.select(c6_cols).to_numpy()
    channel_list.append(c6)
    # # Log 后的 Imbalance: log(Ask) - log(Bid) = log(Ask/Bid) -> 相对强度，这也是合理的
    # c7 = c2 - c4 
    # channel_list.append(c7)
    # # C8: Spread (Price based)
    # c8 = c1-c3
    # channel_list.append(c8)

    ## 原始特征的进入模型
    X = np.stack(channel_list, axis=1)
    return X