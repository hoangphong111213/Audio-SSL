import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import NormalViTBackbone
from mask_utils import MaskingUtility


class AudioJEPA(nn.Module):
    """
    JEPA baseline: context encoder predicts target encoder's latent representations
    of masked blocks. Target encoder is updated via EMA.
    """
    def __init__(self, embed_dim=192, predictor_depth=4):
        super().__init__()
        self.embed_dim = embed_dim

        self.context_encoder = NormalViTBackbone(img_size=(80, 112), embed_dim=embed_dim)
        self.target_encoder = NormalViTBackbone(img_size=(80, 112), embed_dim=embed_dim)

        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())

        self.masker = MaskingUtility(mask_ratio=0.60)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        predictor_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4,
            dim_feedforward=embed_dim * 4, activation="gelu",
            batch_first=True, norm_first=True
        )
        self.predictor_blocks = nn.TransformerEncoder(predictor_layer, num_layers=predictor_depth)
        self.predictor_norm = nn.LayerNorm(embed_dim)

    def _encode(self, encoder, x):
        return encoder.norm(encoder.encoder(x)[0])

    @torch.no_grad()
    def update_target_encoder(self, momentum=0.996):
        for p_c, p_t in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            p_t.data.mul_(momentum).add_(p_c.data, alpha=1.0 - momentum)

    def forward(self, x):
        B, D = x.shape[0], self.embed_dim
        x = F.pad(x, (0, 11))

        with torch.no_grad():
            x_tgt = self.target_encoder.patch_embed(x).flatten(2).transpose(1, 2)
            x_tgt = x_tgt + self.target_encoder.pos_embed
            target_feats = self._encode(self.target_encoder, x_tgt)

        x_ctx = self.context_encoder.patch_embed(x).flatten(2).transpose(1, 2)
        x_ctx = x_ctx + self.context_encoder.pos_embed
        mask, ids_keep, ids_restore = self.masker.generate_block_mask(x_ctx)
        x_visible = self.masker.apply_mask_to_sequence(x_ctx, ids_keep)
        context_feats = self._encode(self.context_encoder, x_visible)

        N_keep = ids_keep.shape[1]
        N_mask = x_ctx.shape[1] - N_keep
        masked_ids = torch.argsort(ids_restore, dim=1)[:, N_keep:]
        target_pos_emb = torch.gather(
            self.context_encoder.pos_embed.expand(B, -1, -1),
            dim=1, index=masked_ids.unsqueeze(-1).expand(-1, -1, D)
        )
        mask_tokens = self.mask_token.expand(B, N_mask, D).clone() + target_pos_emb
        predictions = self.predictor_norm(
            self.predictor_blocks(torch.cat([context_feats, mask_tokens], dim=1))
        )[:, N_keep:, :]

        target_masked = torch.gather(
            target_feats, dim=1,
            index=masked_ids.unsqueeze(-1).expand(-1, -1, D)
        )
        loss = F.smooth_l1_loss(predictions, target_masked.detach())
        std = target_feats.var(dim=0).mean().sqrt().item()

        return loss, std


if __name__ == "__main__":
    jepa = AudioJEPA()
    dummy = torch.randn(16, 1, 80, 101)
    loss, std = jepa(dummy)
    loss.backward()
    jepa.update_target_encoder()
    print(f"AudioJEPA — loss: {loss.item():.4f}, target std: {std:.4f}")