
import torch
import torch.nn as nn


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

class LevelAttentionPooling(nn.Module):
    def __init__(self, d_model, bottleneck_dim=16):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(d_model, bottleneck_dim), # 第一层：投影到低维空间
            nn.Tanh(),                          # 非线性激活（Tanh 在注意力机制中比 ReLU 常用，因为它有界且对称）
            nn.Linear(bottleneck_dim, 1)        # 第二层：计算最终 Score
        )
        self.softmax = nn.Softmax(dim=2)

    def forward(self, h):
        # h: [B, T, L, D]
        scores = self.attn_net(h)      # [B, T, L, 1]
        weights = self.softmax(scores) # [B, T, L, 1]
        out = torch.sum(h * weights, dim=2)
        return out

class LevelConvPooling(nn.Module):
    def __init__(self, d_model, L):
        super().__init__()
        # 卷积核覆盖 3-5 个档位，捕捉局部深度特征
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.flat = nn.Linear(L * d_model, d_model)

    def forward(self, h):
        # h: [B, T, L, D] -> [B*T, D, L]
        B, T, L, D = h.shape
        h = h.view(B * T, L, D).transpose(1, 2)
        
        h = F.relu(self.conv(h))
        h = h.view(B, T, -1) # 展平 L 维度
        return self.flat(h) # [B, T, D]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_L=20):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_L, d_model)

        self.max_L = max_L

    def forward(self):
        # 注意：这里我们假设前 L/2 是 Bid，后 L/2 是 Ask
        # 如果你想让 Bid1 和 Ask1 共享相同的位置感，可以构造对称索引
        half_L = self.max_L // 2
        # 生成类似 [4, 3, 2, 1, 0, 0, 1, 2, 3, 4] 的索引
        bid_indices = torch.arange(half_L - 1, -1, -1)
        ask_indices = torch.arange(0, half_L)
        pos_indices = torch.cat([bid_indices, ask_indices], dim=0) # [L]
        pos_enc = self.pos_embedding(pos_indices).view(1, 1, self.max_L, -1)
        return pos_enc

class LevelWiseEncoder(nn.Module):
    def __init__(self, in_dim, d_model, n_heads, max_L=20):
        super().__init__()

        self.level_encoder = LevelEncoder(in_dim, d_model)

        # 定义位置嵌入：每个档位一个 D 维向量
        self.pos_embedding = PositionalEncoding( d_model,max_L=max_L)

        self.side_attn = SideCrossAttention(d_model, n_heads)

        self.level_pooling = LevelAttentionPooling(d_model)

    def forward(self, x):
        # x: [B, T, L, F]

        B, T, L, F = x.shape
        # 1. 计算相对价格
        mid = (x[:, :, L//2, 0:1] + x[:, :, L//2-1, 0:1]) / 2.0
        rel_price = (x[:, :, :, 0:1] - mid.unsqueeze(2))
        
        # 2. 计算 OFI 并拼接
        ofi = compute_ofi(x) # [B, T, L, 1]

        # 3. 拼接相rel_price和OFI
        x_extended = torch.cat([x, rel_price, ofi], dim=-1) # [B, T, L, 3]

        # 3. 映射到d_model 维度
        h = self.level_encoder(x_extended)

        # 4. 生成位置嵌入
        pos_enc = self.pos_embedding()      
        h = h + pos_enc
        # 5.显式的建模ask和bid的博弈
        bid = h[:, :, :L//2]
        ask = h[:, :, L//2:]
        bid, ask = self.side_attn(bid, ask)
        # 6. 拼接 bid 和 ask
        h = torch.cat([bid, ask], dim=2) # [B, T, L, D]
        # 7. 池化
        output = self.level_pooling(h) # [B, T, D]
        return output