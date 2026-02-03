import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.vision_transformer import Block, PatchEmbed
from functools import partial


class ViTEncoder(nn.Module):
    def __init__(self, img_size=(32, 128), patch_size=(4, 8), in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True):
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=0.)

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        
        # Add dynamic pos_embed resizing
        if x.shape[1] != self.pos_embed.shape[1]:
            # Resizing logic: 2D interpolation
            # x: [B, N, C]
            # self.pos_embed: [1, N_p, C] where N_p = H_p * W_p
            
            # 1. Get current patch grid size from PatchEmbed
            H_p, W_p = self.patch_embed.grid_size
            
            # 2. Estimate new patch grid size
            nt = x.shape[1]
            pos_embed = self.pos_embed.transpose(1, 2) # [1, C, N_p]
            
            # Reshape to 2D
            pos_embed = pos_embed.reshape(1, -1, H_p, W_p) # [1, C, H_p, W_p]
            
            # Calculate new spatial dimensions
            new_H_p = H_p
            new_W_p = nt // new_H_p
            
            # Interpolate
            pos_embed = F.interpolate(pos_embed, size=(new_H_p, new_W_p), mode='bicubic', align_corners=False)
            
            # Flatten back
            pos_embed = pos_embed.flatten(2).transpose(1, 2) # [1, new_N, C]
            x = x + pos_embed
        else:
            x = x + self.pos_embed
            
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x
