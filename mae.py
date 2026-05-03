import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import NormalViTBackbone, SOTAViTBackbone, get_2d_sincos_pos_embed
from mask_utils import MaskingUtility


class AudioMAE(nn.Module):
    def __init__(self, embed_dim=192, decoder_embed_dim=128, decoder_depth=4,
                 decoder_num_heads=4, patch_size=(16, 16), in_chans=1,
                 use_sota_backbone=False, mask_t_prob=0.75, mask_f_prob=0.2):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.patch_dim = in_chans * patch_size[0] * patch_size[1]
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        
        self.mask_t_prob = mask_t_prob
        self.mask_f_prob = mask_f_prob

        backbone_cls = SOTAViTBackbone if use_sota_backbone else NormalViTBackbone
        self.encoder = backbone_cls(patch_size=patch_size, embed_dim=embed_dim)
        self.masker = MaskingUtility()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim, nhead=decoder_num_heads,
            dim_feedforward=decoder_embed_dim * 4, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.decoder_blocks = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.pred_head = nn.Linear(decoder_embed_dim, self.patch_dim)

    def patchify(self, imgs):
        p_h, p_w = self.patch_size
        h = imgs.shape[2] // p_h
        w = imgs.shape[3] // p_w
        x = imgs.reshape(imgs.shape[0], self.in_chans, h, p_h, w, p_w)
        x = torch.einsum('nchpwq->nhwpqc', x)
        return x.reshape(imgs.shape[0], h * w, self.patch_dim)

    @torch.compiler.disable
    def _encode(self, x):
        if hasattr(self.encoder, 'encoder'):
            return self.encoder.norm(
                self.encoder.encoder(x, output_attentions=False, return_dict=False)[0]
            )
        for block in self.encoder.blocks:
            x = block(x)
        return self.encoder.norm(x)

    def forward(self, x):
        B, _, H, W = x.shape
        pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
        x = F.pad(x, (0, pad_w))
        
        grid_F = x.shape[2] // self.patch_size[0]
        grid_T = x.shape[3] // self.patch_size[1]

        x_embed = self.encoder.patch_embed(x).flatten(2).transpose(1, 2)
        pos_embed = get_2d_sincos_pos_embed(self.embed_dim, grid_F, grid_T, x.device)
        x_embed = x_embed + pos_embed

        mask, ids_keep, ids_restore = self.masker.generate_2d_mask(
            x_embed, T=grid_T, F=grid_F, 
            mask_t_prob=self.mask_t_prob, mask_f_prob=self.mask_f_prob
        )
        
        x_visible = self.masker.apply_mask_to_sequence(x_embed, ids_keep)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_visible = torch.cat((cls_tokens, x_visible), dim=1)

        x_encoded = self._encode(x_visible)

        x_decoded = self.decoder_embed(x_encoded)
        
        cls_token_dec = x_decoded[:, :1, :]
        x_visible_dec = x_decoded[:, 1:, :]

        n_mask = ids_restore.shape[1] - x_visible_dec.shape[1]
        mask_tokens = self.mask_token.expand(B, n_mask, -1)
        x_full = torch.cat([x_visible_dec, mask_tokens], dim=1)
        
        x_full = torch.gather(
            x_full, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x_full.shape[-1])
        )
        
        x_full = torch.cat([cls_token_dec, x_full], dim=1)
        
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_embed_dim, grid_F, grid_T, x.device)
        cls_pos = torch.zeros(1, 1, self.decoder_embed_dim, device=x.device)
        decoder_pos_embed = torch.cat([cls_pos, decoder_pos_embed], dim=1)
        
        x_full = x_full + decoder_pos_embed
        x_full = self.decoder_norm(self.decoder_blocks(x_full))
        
        pred = self.pred_head(x_full[:, 1:, :])

        target = self.patchify(x)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6)**.5
        
        loss = ((pred - target) ** 2).mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()

        return loss, pred, mask