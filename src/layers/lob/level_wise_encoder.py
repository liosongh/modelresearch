
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
        self.attn = TinySideCrossAttention(d_model, n_heads)

    def forward(self, bid, ask):
        # bid/ask: [B, T, L/2, D]
        B, T, L_half, D = bid.shape

        # # 合并 B 和 T 维，但使用 view 保持连续性，这样对编译器更友好
        # bid = bid.view(-1, L_half, D)
        # ask = ask.view(-1, L_half, D)

        # 显式指定归一化前的维度
        bid2 = self.attn(bid, ask)
        ask2 = self.attn(ask, bid)

        return bid2, ask2


class TinySideCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, bid, ask):
        # bid/ask: [B, T, L, D]
        B, T, L, D = bid.shape
        
        # 1. 关键步骤：合并 B 和 T
        # 变 4D 为 3D: [B*T, L, D]
        bid = bid.view(-1, L, D)
        ask = ask.view(-1, L, D)
        
        BT = B * T # 新的 Batch 维度

        def split_heads(x):
            # 拆分后变为 4 维: [BT, n_heads, L, d_k]
            return x.view(BT, L, self.n_heads, self.d_k).transpose(1, 2)

        q = split_heads(self.q_proj(bid)) 
        k = split_heads(self.k_proj(ask)) 
        v = split_heads(self.v_proj(ask)) 

        # 2. 计算 Attention (此时是标准的 4 维操作)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (self.d_k ** 0.5)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        out = torch.matmul(attn_probs, v) # [BT, n_heads, L, d_k]
        
        # 3. 还原维度
        out = out.transpose(1, 2).contiguous().view(B, T, L, D)
        return self.out_proj(out)

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


# class LevelConvPooling(nn.Module):
#     def __init__(self, d_model, L):
#         super().__init__()
#         # 卷积核覆盖 3-5 个档位，捕捉局部深度特征
#         self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
#         self.flat = nn.Linear(L * d_model, d_model)

#     def forward(self, h):
#         # h: [B, T, L, D] -> [B*T, D, L]
#         B, T, L, D = h.shape
#         h = h.view(B * T, L, D).transpose(1, 2)
        
#         h = F.relu(self.conv(h))
#         h = h.view(B, T, -1) # 展平 L 维度
#         return self.flat(h) # [B, T, D]


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
        bid_indices = torch.arange(half_L - 1, -1, -1,device=self.pos_embedding.weight.device)
        ask_indices = torch.arange(0, half_L,device=self.pos_embedding.weight.device)
        pos_indices = torch.cat([bid_indices, ask_indices], dim=0) # [L]
        pos_enc = self.pos_embedding(pos_indices).view(1, 1, self.max_L, -1)
        return pos_enc

class LevelWiseEncoder(nn.Module):
    def __init__(self, in_channels, d_model, n_heads,output_dim, max_L=20):
        super().__init__()

        self.level_encoder = LevelEncoder(in_channels, d_model)

        # 定义位置嵌入：每个档位一个 D 维向量
        self.pos_embedding = PositionalEncoding( d_model,max_L=max_L)

        self.side_attn = SideCrossAttention(d_model, n_heads)

        self.level_pooling = LevelAttentionPooling(d_model)

        self.output_dim = output_dim

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.ReLU(),
            nn.LayerNorm(d_model*2),
            nn.Dropout(0.2),
            nn.Linear(d_model*2, output_dim),
        )

    def forward(self, x):
        # x: [B, C, T, L]

        B, C, T, L = x.shape
        x = x.permute(0, 2, 3, 1) # [B, T, L, C]


        # 3. 映射到d_model 维度
        h = self.level_encoder(x) # [B, T, L, D]

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
        output = self.ff(output) # [B, T, output_dim]
        return output