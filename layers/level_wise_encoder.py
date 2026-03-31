
import torch
import torch.nn as nn


def compute_ofi(x):
    """
    x: [B, T, L, 2] -> (price, volume)
    L 包含 bid 和 ask 两个部分。
    假设前 L//2 是 Bid (从高到低)，后 L//2 是 Ask (从低到高)。
    """
    B, T, L, _ = x.shape
    half_L = L // 2
    
    # 提取价格和成交量
    prices = x[..., 0]
    volumes = x[..., 1]
    
    # 计算时间差分 (t - (t-1))
    p_diff = prices[:, 1:] - prices[:, :-1]
    v_diff = volumes[:, 1:] - volumes[:, :-1]
    
    # 初始化 OFI 张量 [B, T-1, L]
    # 对 Bid 端的逻辑
    bid_p_diff = p_diff[:, :, :half_L]
    bid_v_diff = v_diff[:, :, :half_L]
    
    # Bid OFI 逻辑简化实现
    bid_ofi = torch.where(bid_p_diff > 0, volumes[:, 1:, :half_L],
                torch.where(bid_p_diff < 0, -volumes[:, :-1, :half_L], bid_v_diff))
    
    # Ask OFI 逻辑 (注意 Ask 价格上涨通常意味着卖压减弱，逻辑与 Bid 相反)
    ask_p_diff = p_diff[:, :, half_L:]
    ask_v_diff = v_diff[:, :, half_L:]
    
    ask_ofi = torch.where(ask_p_diff < 0, volumes[:, 1:, half_L:],
                torch.where(ask_p_diff > 0, -volumes[:, :-1, half_L:], ask_v_diff))
    
    # 拼接并补齐第一帧 (Padding)
    ofi = torch.cat([bid_ofi, ask_ofi], dim=-1)
    padding = torch.zeros(B, 1, L, device=x.device)
    ofi = torch.cat([padding, ofi], dim=1) 
    
    return ofi.unsqueeze(-1) # [B, T, L, 1]
# =========================
# [1] Level-wise Encoder
# =========================

class LevelEncoder(nn.Module):
    def __init__(self, in_dim, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x):
        # x: [B, T, L, F]
        # rel_price: [B, T, L, 1]

        return self.mlp(x)  # [B, T, L, D]


# =========================
# [2] Side-aware Cross Attention
# =========================
## 显式的建模ask和bid的博弈
class SideCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, bid, ask):
        # bid/ask: [B, T, L/2, D]
        B, T, L, D = bid.shape

        bid = bid.reshape(B*T, L, D)
        ask = ask.reshape(B*T, L, D)

        bid2, _ = self.attn(bid, ask, ask)
        ask2, _ = self.attn(ask, bid, bid)

        return bid2.reshape(B, T, L, D), ask2.reshape(B, T, L, D)


class LevelWiseEncoder(nn.Module):
    def __init__(self, in_dim, d_model, n_heads):
        super().__init__()

        self.level_encoder = LevelEncoder(in_dim, d_model)

        self.side_attn = SideCrossAttention(d_model, n_heads)

    def forward(self, x):
        # x: [B, T, L, F]

        B, T, L, F = x.shape
        # 1. 计算相对价格
        mid = (x[:, :, L//2, 0:1] + x[:, :, L//2-1, 0:1]) / 2.0
        rel_price = (x[:, :, :, 0:1] - mid.unsqueeze(2))
        
        # 2. 计算 OFI 并拼接
        ofi = compute_ofi(x) # [B, T, L, 1]
        x_extended = torch.cat([x, rel_price, ofi], dim=-1) # [B, T, L, 3]

        # 3. 编码
        x_extended = self.level_encoder(x_extended)
        # 4. 分离 bid 和 ask
        bid = x_extended[:, :, :L//2]
        ask = x_extended[:, :, L//2:]
        # 5. 显式的建模ask和bid的博弈
        bid, ask = self.side_attn(bid, ask)
        # 6. 拼接 bid 和 ask
        x_extended = torch.cat([bid, ask], dim=2) # [B, T, L, D]
        return x_extended