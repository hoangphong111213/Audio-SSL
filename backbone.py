import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTConfig
from transformers.models.vit.modeling_vit import ViTEncoder


def get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w, device):
    grid_h_arr = torch.arange(grid_h, dtype=torch.float32, device=device)
    grid_w_arr = torch.arange(grid_w, dtype=torch.float32, device=device)
    grid = torch.meshgrid(grid_w_arr, grid_h_arr, indexing="ij")
    grid = torch.stack(grid, dim=0)

    omega = torch.arange(embed_dim // 4, dtype=torch.float32, device=device)
    omega = 1.0 / (10000.0 ** (omega / (embed_dim / 4.0)))

    out_w = torch.einsum("m,d->md", grid[0].flatten(), omega)
    out_h = torch.einsum("m,d->md", grid[1].flatten(), omega)

    pos_w = torch.cat([torch.sin(out_w), torch.cos(out_w)], dim=1)
    pos_h = torch.cat([torch.sin(out_h), torch.cos(out_h)], dim=1)

    pos_embed = torch.cat([pos_h, pos_w], dim=1)
    return pos_embed.unsqueeze(0)


class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w3 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class MacaronTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn1 = SwiGLU(embed_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.ffn2 = SwiGLU(embed_dim, hidden_dim)

    def forward(self, x):
        x = x + 0.5 * self.ffn1(self.norm1(x))
        attn_out, _ = self.attn(self.norm2(x), self.norm2(x), self.norm2(x))
        x = x + attn_out
        x = x + 0.5 * self.ffn2(self.norm3(x))
        return x


class NormalViTBackbone(nn.Module):
    def __init__(self, patch_size=(16, 16), in_chans=1, embed_dim=192, depth=12, num_heads=3):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        config = ViTConfig(
            hidden_size=embed_dim,
            num_hidden_layers=depth,
            num_attention_heads=num_heads,
            intermediate_size=embed_dim * 12,
            hidden_act="gelu",
        )
        self.encoder = ViTEncoder(config)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        grid_f = H // self.patch_size[0]
        grid_t = W // self.patch_size[1]

        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        pos_embed = get_2d_sincos_pos_embed(self.embed_dim, grid_f, grid_t, x.device)
        x = x + pos_embed

        x = self.encoder(x, output_attentions=False, return_dict=False)[0]
        return self.norm(x)


class SOTAViTBackbone(nn.Module):
    def __init__(self, patch_size=(16, 16), in_chans=1, embed_dim=192, depth=12, num_heads=3):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.ModuleList([
            MacaronTransformerBlock(embed_dim, num_heads, hidden_dim=embed_dim * 4)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        grid_f = H // self.patch_size[0]
        grid_t = W // self.patch_size[1]

        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        pos_embed = get_2d_sincos_pos_embed(self.embed_dim, grid_f, grid_t, x.device)
        x = x + pos_embed

        for block in self.blocks:
            x = block(x)
        return self.norm(x)