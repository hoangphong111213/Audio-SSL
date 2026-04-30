import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import NormalViTBackbone


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=2048, hidden_dim=512, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return self.last_layer(x)


class AudioDINO(nn.Module):
    """
    DINO for audio mel-spectrograms.
    Student sees global + local time crops; teacher sees only global crops.
    Teacher updated via EMA; collapse prevented by centering + temperature sharpening.
    """
    def __init__(self, embed_dim=192, out_dim=2048, n_global=2, n_local=4,
                 global_crop_frac=0.88, local_crop_frac=0.50,
                 teacher_temp=0.04, student_temp=0.10, center_momentum=0.9):
        super().__init__()
        self.n_global = n_global
        self.n_local = n_local
        self.global_crop_size = int(101 * global_crop_frac)
        self.local_crop_size = int(101 * local_crop_frac)
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum

        self.student = NormalViTBackbone(img_size=(80, 112), embed_dim=embed_dim)
        self.teacher = NormalViTBackbone(img_size=(80, 112), embed_dim=embed_dim)
        self.student_head = DINOHead(embed_dim, out_dim)
        self.teacher_head = DINOHead(embed_dim, out_dim)

        for p in self.teacher.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

        self.teacher.load_state_dict(self.student.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        self.register_buffer("center", torch.zeros(1, out_dim))

    def _crop_batch(self, mels, size):
        B = mels.shape[0]
        starts = torch.randint(0, 101 - size + 1, (B,), device=mels.device)
        crops = torch.stack([mels[b, :, :, starts[b]: starts[b] + size] for b in range(B)])
        return F.pad(crops, (0, 112 - size))

    def _embed(self, backbone, x):
        tokens = backbone.patch_embed(x).flatten(2).transpose(1, 2)
        tokens = tokens + backbone.pos_embed
        tokens = backbone.norm(
            backbone.encoder(tokens, output_attentions=False, return_dict=False)[0]
        )
        return tokens.mean(dim=1)

    @torch.no_grad()
    def update_target_encoder(self, momentum=0.996):
        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1 - momentum)
        for ps, pt in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1 - momentum)

    @torch.no_grad()
    def _update_center(self, teacher_logits):
        batch_center = torch.cat(teacher_logits, dim=0).mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1 - self.center_momentum)

    def _dino_loss(self, student_logits, teacher_logits):
        t_probs = [
            F.softmax((t - self.center) / self.teacher_temp, dim=-1).detach()
            for t in teacher_logits
        ]
        total, n = 0.0, 0
        for i, s in enumerate(student_logits):
            s_log = F.log_softmax(s / self.student_temp, dim=-1)
            for j, tp in enumerate(t_probs):
                if i == j:
                    continue
                total += -(tp * s_log).sum(dim=-1).mean()
                n += 1
        return total / n

    def forward(self, mels):
        global_views = [self._crop_batch(mels, self.global_crop_size) for _ in range(self.n_global)]
        local_views = [self._crop_batch(mels, self.local_crop_size) for _ in range(self.n_local)]

        student_logits = [self.student_head(self._embed(self.student, v)) for v in global_views + local_views]

        with torch.no_grad():
            teacher_logits = [self.teacher_head(self._embed(self.teacher, v)) for v in global_views]

        self._update_center(teacher_logits)
        loss = self._dino_loss(student_logits, teacher_logits)

        with torch.no_grad():
            t_prob = F.softmax((teacher_logits[0] - self.center) / self.teacher_temp, dim=-1)
            entropy = -(t_prob * (t_prob + 1e-8).log()).sum(dim=-1).mean().item()

        return loss, entropy


if __name__ == "__main__":
    dino = AudioDINO()
    dummy = torch.randn(8, 1, 80, 101)
    loss, entropy = dino(dummy)
    loss.backward()
    dino.update_target_encoder()
    print(f"AudioDINO — loss: {loss.item():.4f}, teacher entropy: {entropy:.4f}")