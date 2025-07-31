import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

import geoopt 
from geoopt import ManifoldParameter


# Hyperbolic Linear Layers
class HyperbolicLinear(nn.Module):
    def __init__(self, in_features, out_features, manifold, c):
        super().__init__()
        self.weight = ManifoldParameter(torch.randn(out_features, in_features) * 0.01, manifold=manifold)
        self.bias = ManifoldParameter(torch.zeros(out_features), manifold=manifold)
        self.manifold = manifold
        self.c = c

    def forward(self, x):
        x = self.manifold.mobius_matvec(self.weight, x)
        x = self.manifold.mobius_add(x, self.bias)
        return x

# Patch Embeddings
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# 2. Adding Positional Embeddings
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, seq_len):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len + 1, embed_dim))  # Adjusted for [CLS] token

    def forward(self, x):
        return (x + self.pos_embed)


# Multi-head self-attention block
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first = True)

    def forward(self, x):
        return self.attn(x, x, x)[0]

# ViT Encoder
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, manifold, c):
        super().__init__()
        self.manifold = manifold
        self.c = c
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        if self.c != 0:
            self.lin1 = HyperbolicLinear(embed_dim, mlp_dim, self.manifold, self.c)
            self.lin2 = HyperbolicLinear(mlp_dim, embed_dim, self.manifold, self.c)
        else:
            self.lin1 = nn.Linear(embed_dim, mlp_dim)
            self.lin2 = nn.Linear(mlp_dim, embed_dim)
        # self.mlp = nn.Sequential(
        #     nn.Linear(embed_dim, mlp_dim),
        #     nn.ReLU(),
        #     nn.Linear(mlp_dim, embed_dim)
        # )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x_in1 = x
        if self.c != 0:
            x_in1 = self.manifold.expmap0(x_in1)
        x = self.norm1(x)
        x = self.attn(x)
        if self.c != 0:
            x = self.manifold.expmap0(x)
            x = self.manifold.mobius_add(x, x_in1)
            x = self.manifold.logmap0(x)
        else:
            x = x + x_in1

        x_in2 = x
        if self.c != 0:
            x_in2 = self.manifold.expmap0(x_in2)
        x = self.norm2(x)
        if self.c != 0:
            x = self.manifold.expmap0(x)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        if self.c != 0:
            x = self.manifold.mobius_add(x, x_in2)
            x = self.manifold.logmap0(x)
        else:
            x = x + x_in2

        # x = x + self.attn(self.norm1(x))
        # x = x + self.mlp(self.norm2(x))
        return x

# Final ViT Block
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=1, embed_dim=768, depth=6, num_heads=8, dropout = 0.1, num_classes=10, mlp_dim=1024, c=0.0):
        super().__init__()
        self.c = c
        self.manifold = geoopt.PoincareBall(c=self.c)
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, (img_size // patch_size) ** 2)
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, self.manifold, self.c) for _ in range(depth)
        ])
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.mlp_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(x)
        return self.mlp_head(x[:, 0])