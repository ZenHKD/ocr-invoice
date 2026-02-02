import torch
import torch.nn as nn
from functools import partial

class DecoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = norm_layer(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.norm3 = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )

    def forward(self, query, key, tgt_mask=None, memory_mask=None):
        # query: (B, T, C) - Decoder input
        # key: (B, S, C) - Encoder output (Memory)

        # Self Attention
        q = k = v = self.norm1(query)
        # For nn.MultiheadAttention, attn_mask should be (T, T) or (B*num_heads, T, T)
        # If float, it's additive mask. If bool, it's True for ignored positions.
        query = query + self.self_attn(q, k, v, attn_mask=tgt_mask)[0]

        # Cross Attention
        q = self.norm2(query)
        k = v = key
        query = query + self.cross_attn(q, k, v)[0]

        # MLP
        query = query + self.mlp(self.norm3(query))
        return query

class Decoder(nn.Module):
    def __init__(self, embed_dim=384, depth=1, num_heads=6, mlp_ratio=4., qkv_bias=True, max_len=25):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.blocks = nn.ModuleList([
            DecoderLayer(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, max_len + 1, embed_dim)) # +1 for EOS/BOS alignment

    def forward(self, tgt, memory, tgt_mask=None):
        # tgt: (B, T, C) - Embeddings of targets
        # memory: (B, S, C) - Encoder output

        # Add positional embeddings to target
        tgt = tgt + self.pos_embed[:, :tgt.shape[1], :]

        for blk in self.blocks:
            tgt = blk(tgt, memory, tgt_mask=tgt_mask)

        tgt = self.norm(tgt)
        return tgt
