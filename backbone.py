import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTConfig
from transformers.models.vit.modeling_vit import ViTEncoder


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
    """Standard ViT-Tiny via HuggingFace. Shared baseline for MAE, JEPA, and DINO."""
    def __init__(self, img_size=(80, 101), patch_size=(16, 16), in_chans=1,
                 embed_dim=192, depth=12, num_heads=3):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        config = ViTConfig(
            hidden_size=embed_dim,
            num_hidden_layers=depth,
            num_attention_heads=num_heads,
            intermediate_size=embed_dim * 4,
            hidden_act="gelu",
        )
        self.encoder = ViTEncoder(config)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.encoder(x, output_attentions=False, return_dict=False)[0]
        return self.norm(x)


class SOTAViTBackbone(nn.Module):
    """AudioMAE++ backbone: Macaron-style blocks with SwiGLU FFNs."""
    def __init__(self, img_size=(80, 101), patch_size=(16, 16), in_chans=1,
                 embed_dim=192, depth=12, num_heads=3):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.blocks = nn.ModuleList([
            MacaronTransformerBlock(embed_dim, num_heads, hidden_dim=embed_dim * 4)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


if __name__ == "__main__":
    dummy = torch.randn(4, 1, 80, 101)
    for cls in (NormalViTBackbone, SOTAViTBackbone):
        out = cls()(dummy)
        print(f"{cls.__name__}: {out.shape}")