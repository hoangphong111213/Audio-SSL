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

    def generate_block_mask(self, x, min_aspect=0.3, max_aspect=3.0):
        B, N, _ = x.shape
        device = x.device
        grid_h, grid_w = self._infer_grid(N)
        target_n_mask = int(N * self.mask_ratio)

        all_masks, all_ids_keep, all_ids_restore = [], [], []

        for b in range(B):
            for _ in range(10): 
                aspect  = math.exp(torch.empty(1).uniform_(math.log(min_aspect), math.log(max_aspect)).item())
                block_h = max(1, min(int(round(math.sqrt(target_n_mask * aspect))), grid_h))
                block_w = max(1, min(int(round(math.sqrt(target_n_mask / aspect))), grid_w))
                top  = torch.randint(0, grid_h - block_h + 1, (1,)).item()
                left = torch.randint(0, grid_w - block_w + 1, (1,)).item()
                mask_2d = torch.zeros(grid_h, grid_w, device=device)
                mask_2d[top:top + block_h, left:left + block_w] = 1
                if mask_2d.sum() > 0:
                    break

            mask_1d    = mask_2d.flatten()
            kept_ids   = (mask_1d == 0).nonzero(as_tuple=False).squeeze(1)
            masked_ids = (mask_1d == 1).nonzero(as_tuple=False).squeeze(1)
            kept_ids   = kept_ids[torch.randperm(len(kept_ids), device=device)]
            masked_ids = masked_ids[torch.randperm(len(masked_ids), device=device)]

            ids_shuffle = torch.cat([kept_ids, masked_ids])
            ids_restore = torch.argsort(ids_shuffle)

            all_masks.append(mask_1d)
            all_ids_keep.append(kept_ids)
            all_ids_restore.append(ids_restore)

        min_keep    = min(t.shape[0] for t in all_ids_keep)
        mask        = torch.stack(all_masks)
        ids_keep    = torch.stack([t[:min_keep] for t in all_ids_keep])
        ids_restore = torch.stack(all_ids_restore)

        return mask, ids_keep, ids_restore

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