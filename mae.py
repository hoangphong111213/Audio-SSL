import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import NormalViTBackbone, SOTAViTBackbone
from mask_utils import MaskingUtility


class AudioMAE(nn.Module):
    """
    Vanilla MAE: random patch masking, pixel reconstruction via MSE.
    use_sota_backbone swaps the encoder to AudioMAE++ (Macaron + SwiGLU).
    """
    def __init__(self, embed_dim=192, decoder_embed_dim=128, decoder_depth=4,
                 decoder_num_heads=4, patch_size=(16, 16), in_chans=1,
                 use_sota_backbone=False):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.patch_dim = in_chans * patch_size[0] * patch_size[1]

        backbone_cls = SOTAViTBackbone if use_sota_backbone else NormalViTBackbone
        self.encoder = backbone_cls(img_size=(80, 112), embed_dim=embed_dim)
        self.masker = MaskingUtility(mask_ratio=0.75)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.encoder.num_patches, decoder_embed_dim)
        )
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

    def _encode(self, x):
        if hasattr(self.encoder, 'encoder'):
            return self.encoder.norm(
                self.encoder.encoder(x, output_attentions=False, return_dict=False)[0]
            )
        for block in self.encoder.blocks:
            x = block(x)
        return self.encoder.norm(x)

    def forward(self, x):
        x = F.pad(x, (0, 11))

        x_embed = self.encoder.patch_embed(x).flatten(2).transpose(1, 2)
        x_embed = x_embed + self.encoder.pos_embed
        mask, ids_keep, ids_restore = self.masker.generate_random_mask(x_embed)

        x_visible = self.masker.apply_mask_to_sequence(x_embed, ids_keep)
        x_encoded = self._encode(x_visible)

        x_decoded = self.decoder_embed(x_encoded)
        n_mask = ids_restore.shape[1] - x_decoded.shape[1]
        mask_tokens = self.mask_token.repeat(x_decoded.shape[0], n_mask, 1)
        x_full = torch.cat([x_decoded, mask_tokens], dim=1)
        x_full = torch.gather(
            x_full, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x_full.shape[2])
        )
        x_full = self.decoder_norm(self.decoder_blocks(x_full + self.decoder_pos_embed))
        pred = self.pred_head(x_full)

        target = self.patchify(x)
        loss = ((pred - target) ** 2).mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()

        return loss, pred, mask


if __name__ == "__main__":
    dummy = torch.randn(16, 1, 80, 101)
    for sota in (False, True):
        mae = AudioMAE(use_sota_backbone=sota)
        loss, pred, mask = mae(dummy)
        loss.backward()
        label = "SOTA" if sota else "Normal"
        print(f"{label} MAE — loss: {loss.item():.4f}, pred: {pred.shape}")