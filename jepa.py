import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import NormalViTBackbone, get_2d_sincos_pos_embed
from mask_utils import MaskingUtility


class AudioJEPA(nn.Module):
    def __init__(self, embed_dim=192, predictor_embed_dim=96, predictor_depth=4, 
                 mask_t_prob=0.5, mask_f_prob=0.4, patch_size=(16, 16)):
        super().__init__()
        self.embed_dim = embed_dim
        self.predictor_embed_dim = predictor_embed_dim
        self.patch_size = patch_size
        
        self.mask_t_prob = mask_t_prob
        self.mask_f_prob = mask_f_prob

        self.context_encoder = NormalViTBackbone(patch_size=patch_size, embed_dim=embed_dim)
        self.target_encoder = NormalViTBackbone(patch_size=patch_size, embed_dim=embed_dim)

        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())

        self.masker = MaskingUtility()
        
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        predictor_layer = nn.TransformerEncoderLayer(
            d_model=predictor_embed_dim, nhead=4,
            dim_feedforward=predictor_embed_dim * 4, activation="gelu",
            batch_first=True, norm_first=True
        )
        self.predictor_blocks = nn.TransformerEncoder(predictor_layer, num_layers=predictor_depth)
        self.predictor_norm = nn.LayerNorm(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim)

    @torch.compiler.disable
    def _encode(self, encoder, x):
        if hasattr(encoder, 'encoder'):
            return encoder.norm(encoder.encoder(x)[0])
        for block in encoder.blocks:
            x = block(x)
        return encoder.norm(x)

    @torch.no_grad()
    def update_target_encoder(self, momentum=0.996):
        for p_c, p_t in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            p_t.data.mul_(momentum).add_(p_c.data, alpha=1.0 - momentum)

    def forward(self, x):
        B, D = x.shape[0], self.embed_dim
        _, _, H, W = x.shape
        
        pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
        x = F.pad(x, (0, pad_w))
        
        grid_F = x.shape[2] // self.patch_size[0]
        grid_T = x.shape[3] // self.patch_size[1]
        pos_embed = get_2d_sincos_pos_embed(self.embed_dim, grid_F, grid_T, x.device)

        with torch.no_grad():
            x_tgt = self.target_encoder.patch_embed(x).flatten(2).transpose(1, 2)
            x_tgt = x_tgt + pos_embed
            target_feats = self._encode(self.target_encoder, x_tgt)
            target_feats = F.layer_norm(target_feats, (target_feats.size(-1),))

        x_ctx = self.context_encoder.patch_embed(x).flatten(2).transpose(1, 2)
        x_ctx = x_ctx + pos_embed
        
        mask, ids_keep, ids_restore = self.masker.generate_2d_mask(
            x_ctx, T=grid_T, F=grid_F, 
            mask_t_prob=self.mask_t_prob, mask_f_prob=self.mask_f_prob
        )
        x_visible = self.masker.apply_mask_to_sequence(x_ctx, ids_keep)
        context_feats = self._encode(self.context_encoder, x_visible)

        N_keep = ids_keep.shape[1]
        N_mask = x_ctx.shape[1] - N_keep
        masked_ids = torch.argsort(ids_restore, dim=1)[:, N_keep:]
        
        target_pos_emb = torch.gather(
            pos_embed.expand(B, -1, -1),
            dim=1, index=masked_ids.unsqueeze(-1).expand(-1, -1, D)
        )

        context_feats_pred = self.predictor_embed(context_feats)
        target_pos_emb_pred = self.predictor_embed(target_pos_emb)
        
        mask_tokens = self.mask_token.expand(B, N_mask, self.predictor_embed_dim).clone() + target_pos_emb_pred
        
        pred_sequence = torch.cat([context_feats_pred, mask_tokens], dim=1)
        predictions = self.predictor_norm(self.predictor_blocks(pred_sequence))
        
        predictions = predictions[:, N_keep:, :]
        predictions = self.predictor_proj(predictions)

        target_masked = torch.gather(
            target_feats, dim=1,
            index=masked_ids.unsqueeze(-1).expand(-1, -1, D)
        )
        
        loss = F.smooth_l1_loss(predictions, target_masked, beta=1.0)
        std = target_feats.var(dim=0).mean().sqrt().item()

        return loss, std