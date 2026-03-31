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
    return df_all


## 加载原始数据
def fullfill_lob_data(lob_data: pl.DataFrame,date: list[str],levels: int = 10) -> pl.DataFrame:
    # date = ['2025-11-04','2025-11-05','2025-11-06','2025-11-07','2025-11-08','2025-11-09','2025-11-10',
    #         '2025-11-11','2025-11-12','2025-11-13','2025-11-14','2025-11-15','2025-11-16','2025-11-17',
    #         '2025-11-18','2025-11-19','2025-11-20','2025-11-21','2025-11-22','2025-11-23','2025-11-24',
    #         '2025-11-25','2025-11-26','2025-11-27','2025-11-28','2025-11-29','2025-11-30','2025-12-01',
    #         '2025-12-02','2025-12-03','2025-12-04','2025-12-05','2025-12-06','2025-12-07',
    #         ]

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




import numpy as np
from scipy.interpolate import CubicSpline

# ==============================================
# 核心函数：处理1帧100ms LOB快照，输出供需结构特征
# 输入：bid/ask = [价格数组, 数量数组]（默认10档，可改）
# 输出：结构化特征（斜率/曲率/凸性/流动性）
# ==============================================
def process_lob_frame(bid_data, ask_data):
    """
    逐帧处理LOB快照，构建Supply/Demand连续曲线 + 提取结构特征
    :param bid_data: [bid_prices(降序), bid_volumes] 买盘（买1~买10）
    :param ask_data: [ask_prices(升序), ask_volumes] 卖盘（卖1~卖10）
    :return: 字典格式核心特征（直接可用）
    """
    # ---------------------- 1. 实盘数据校验（必加，防崩溃）----------------------
    bid_prices, bid_volumes = bid_data
    ask_prices, ask_volumes = ask_data

    # 剔除异常盘口（卖一 ≤ 买一 为无效数据）
    if ask_prices[0] <= bid_prices[0]:
        return {"error": "交叉报价，无效LOB"}

    # ---------------------- 2. 计算累计深度（构造离散供需点）----------------------
    # 供给曲线 Supply(p)：卖方累计挂单量（价格升序）
    supply_cum = np.cumsum(ask_volumes)
    # 需求曲线 Demand(p)：买方累计挂单量（价格降序）
    demand_cum = np.cumsum(bid_volumes)

    # 中间价（核心计算点，只算这里，速度最快）
    mid_price = (bid_prices[0] + ask_prices[0]) / 2

    # ---------------------- 3. 三次样条拟合连续函数（高频标准）----------------------
    # 自然边界样条，速度最快、无过冲
    cs_supply = CubicSpline(ask_prices, supply_cum, bc_type="natural")
    cs_demand = CubicSpline(bid_prices, demand_cum, bc_type="natural")

    # ---------------------- 4. 提取核心结构特征（仅计算中间价，极致高效）----------------------
    # 1. 斜率（一阶导数）
    supply_slope = cs_supply.derivative(1)(mid_price)
    demand_slope = cs_demand.derivative(1)(mid_price)

    # 2. 曲率（曲线弯曲程度，流动性突变信号）
    def curvature(y1, y2):
        """曲率公式：|y''| / (1 + y'^2)^1.5"""
        return np.abs(y2) / np.power(1 + y1**2, 1.5)

    supply_2nd = cs_supply.derivative(2)(mid_price)
    demand_2nd = cs_demand.derivative(2)(mid_price)

    supply_curvature = curvature(supply_slope, supply_2nd)
    demand_curvature = curvature(demand_slope, demand_2nd)

    # 3. 凸性（二阶导数符号，资金行为信号）
    supply_convex = 1 if supply_2nd > 0 else (-1 if supply_2nd < 0 else 0)
    demand_convex = 1 if demand_2nd > 0 else (-1 if demand_2nd < 0 else 0)

    # ---------------------- 5. 输出结构化特征 ----------------------
    return {
        "mid_price": round(mid_price, 4),
        # 供给曲线（卖盘）特征
        "supply_slope": round(supply_slope, 4),
        "supply_curvature": round(supply_curvature, 6),
        "supply_convexity": supply_convex,
        # 需求曲线（买盘）特征
        "demand_slope": round(demand_slope, 4),
        "demand_curvature": round(demand_curvature, 6),
        "demand_convexity": demand_convex,
        # 原始累计深度（校验用）
        "total_supply": int(supply_cum[-1]),
        "total_demand": int(demand_cum[-1])
    }
    
