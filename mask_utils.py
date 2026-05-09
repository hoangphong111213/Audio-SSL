import torch
import math


class MaskingUtility:
    def __init__(self, mask_ratio=0.75):
        self.mask_ratio = mask_ratio

    def generate_random_mask(self, x):
        B, N, _ = x.shape
        len_keep = int(N * (1 - self.mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask, ids_keep, ids_restore

    def generate_jepa_block_mask(self, x, grid_h, grid_w,
                                 min_t=4, max_t=8, min_f=2, max_f=4,
                                 n_target_blocks=4, context_mask_ratio=0.15):
        B, N, _ = x.shape
        device = x.device
        F, T = grid_h, grid_w

        target_mask = torch.zeros(B, F, T, dtype=torch.bool, device=device)
        for b in range(B):
            for _ in range(n_target_blocks):
                bh = torch.randint(min_f, max_f + 1, (1,), device=device).item()
                bw = torch.randint(min_t, max_t + 1, (1,), device=device).item()
                f0 = torch.randint(0, max(1, F - bh + 1), (1,), device=device).item()
                t0 = torch.randint(0, max(1, T - bw + 1), (1,), device=device).item()
                target_mask[b, f0:f0 + bh, t0:t0 + bw] = True

        target_flat = target_mask.view(B, N)

        rand = torch.rand(B, N, device=device).masked_fill(target_flat, 1.0)
        context_visible = ~target_flat & (rand >= context_mask_ratio)

        ids_sorted  = torch.argsort((~context_visible).long(), dim=1, stable=True)
        ids_restore = torch.argsort(ids_sorted, dim=1)
        N_keep      = int(context_visible.sum(dim=1).min().item())
        ids_keep    = ids_sorted[:, :N_keep]

        target_ids_list = [target_flat[b].nonzero(as_tuple=False).squeeze(-1) for b in range(B)]
        max_tgt    = max(t.numel() for t in target_ids_list)
        target_ids = torch.stack([
            torch.cat([t, t[-1:].expand(max_tgt - t.numel())]) for t in target_ids_list
        ])

        return ids_keep, ids_restore, target_ids

    def generate_2d_mask(self, x, T, F, mask_t_prob=0.6, mask_f_prob=0.5):
        B, N, D = x.shape
        len_keep_t = int(T * (1 - mask_t_prob))
        len_keep_f = int(F * (1 - mask_f_prob))

        noise_t = torch.rand(B, T, device=x.device)
        ids_restore_t = torch.argsort(torch.argsort(noise_t, dim=1), dim=1)
        
        noise_f = torch.rand(B, F, device=x.device)
        ids_restore_f = torch.argsort(torch.argsort(noise_f, dim=1), dim=1)

        mask_f = torch.ones(B, F, device=x.device)
        mask_f[:, :len_keep_f] = 0
        mask_f = torch.gather(mask_f, dim=1, index=ids_restore_f).unsqueeze(1).repeat(1, T, 1)

        mask_t = torch.ones(B, T, device=x.device)
        mask_t[:, :len_keep_t] = 0
        mask_t = torch.gather(mask_t, dim=1, index=ids_restore_t).unsqueeze(1).repeat(1, F, 1).permute(0, 2, 1)

        mask = 1 - (1 - mask_t) * (1 - mask_f)
        
        id2res = torch.arange(B * T * F, device=x.device).reshape(B, T, F)
        id2res = id2res + 999 * mask
        
        ids_shuffle = torch.argsort(id2res.flatten(start_dim=1), dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep_f * len_keep_t]

        return mask.flatten(start_dim=1), ids_keep, ids_restore

    def apply_mask_to_sequence(self, x, ids_keep):
        idx = ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        return torch.gather(x, dim=1, index=idx)

    @staticmethod
    def _infer_grid(seq_len):
        for h in range(int(math.isqrt(seq_len)), 0, -1):
            if seq_len % h == 0:
                return h, seq_len // h
        return 1, seq_len


if __name__ == "__main__":
    dummy = torch.randn(4, 35, 192)
    masker = MaskingUtility(mask_ratio=0.75)

    mask, ids_keep, ids_restore = masker.generate_random_mask(dummy)
    print(f"Random — mask: {mask.shape}, ids_keep: {ids_keep.shape}")

    mask, ids_keep, ids_restore = masker.generate_block_mask(dummy)
    print(f"Block  — mask: {mask.shape}, ids_keep: {ids_keep.shape}")

    mask, ids_keep, ids_restore = masker.generate_2d_mask(dummy, T=7, F=5)
    print(f"2D Env — mask: {mask.shape}, ids_keep: {ids_keep.shape}")