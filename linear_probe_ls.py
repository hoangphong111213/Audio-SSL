"""
linear_probe_ls.py  —  Domain-shift linear probe for LS-only SSL checkpoints.

Loads a checkpoint from  checkpoints/<model>_ls/
Probes on GSC (keyword spotting) — an out-of-domain test vs. LibriSpeech SSL.
Saves results to  logs/<model>_ls/  so nothing in the original run is touched.

Usage:
    python linear_probe_ls.py --model mae      # probes mae_ls checkpoint
    python linear_probe_ls.py --model jepa     # probes jepa_ls checkpoint
    python linear_probe_ls.py --model mae_sota # probes mae_sota_ls checkpoint
"""

import os
import argparse
import json
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from dataset import CachedMelDataset
from mae import AudioMAE
from jepa import AudioJEPA

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model components (identical to linear_probe.py)
# ---------------------------------------------------------------------------

class AttentivePooling(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, x):
        q = self.query.expand(x.shape[0], -1, -1)
        out, _ = self.attn(q, x, x)
        return out.squeeze(1)


class LinearProbeWrapper(nn.Module):
    def __init__(self, encoder, num_classes, embed_dim=192):
        super().__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.pool = AttentivePooling(embed_dim, num_heads=4)
        self.bn   = nn.BatchNorm1d(embed_dim, affine=False)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()   # encoder stays frozen / eval at all times
        return self

    def forward(self, x):
        with torch.no_grad():
            pad_w = (16 - x.shape[3] % 16) % 16
            x = F.pad(x, (0, pad_w))
            features = self.encoder(x)
        pooled = self.pool(features)
        return self.head(self.bn(pooled))


# ---------------------------------------------------------------------------
# Dataset  —  GSC only (domain shift target)
# ---------------------------------------------------------------------------

def build_probe_datasets(args):
    """Probe dataset is GSC — deliberately out-of-domain vs. LS-only SSL."""
    full = CachedMelDataset(os.path.join(args.data_dir, "mel_cache", "gsc"))

    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(full), generator=g).tolist()
    train_size = int(0.9 * len(full))

    return Subset(full, perm[:train_size]), Subset(full, perm[train_size:])


def build_loader(dataset, args, shuffle):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=shuffle,
        persistent_workers=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # Derive run_name so all I/O goes to the _ls directories
    run_name = f"{args.model}_ls"

    print(f"Domain-Shift Linear Probe | checkpoint: {run_name} | probe target: GSC | Device: {DEVICE}")
    os.makedirs(f"logs/{run_name}",        exist_ok=True)
    os.makedirs(f"checkpoints/{run_name}", exist_ok=True)

    train_dataset, val_dataset = build_probe_datasets(args)
    train_loader = build_loader(train_dataset, args, shuffle=True)
    val_loader   = build_loader(val_dataset,   args, shuffle=False)

    # Infer number of classes from the probe split
    all_labels = set()
    for _, label in DataLoader(train_dataset, batch_size=1024, num_workers=4):
        all_labels.update(label.tolist())
    num_classes = len(all_labels)
    print(f"Detected {num_classes} GSC classes (domain-shift target)")

    # Build base model
    if args.model == "mae":
        base_model = AudioMAE(use_sota_backbone=False)
    elif args.model == "mae_sota":
        base_model = AudioMAE(use_sota_backbone=True)
    elif args.model == "jepa":
        base_model = AudioJEPA()
    else:
        raise ValueError("Unsupported model for probing")

    # Load LS-only SSL checkpoint
    ckpt_path = f"checkpoints/{run_name}/last.pt"
    if not os.path.isfile(ckpt_path):
        ckpt_path = f"checkpoints/{run_name}/best.pt"
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint found under checkpoints/{run_name}/\n"
            f"Run  python train_ls.py --model {args.model}  first."
        )

    ckpt = torch.load(ckpt_path, map_location="cpu")["model"]
    ckpt = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}
    base_model.load_state_dict(ckpt)
    print(f"Loaded LS-only checkpoint: {ckpt_path}")

    encoder = base_model.encoder if args.model in ("mae", "mae_sota") else base_model.context_encoder

    model     = LinearProbeWrapper(encoder, num_classes=num_classes, embed_dim=192).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    probe_params = list(model.pool.parameters()) + list(model.head.parameters())
    optimizer = optim.SGD(probe_params, lr=args.lr, momentum=0.9, weight_decay=0.0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_acc = 0.0
    history  = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for mels, batch_labels in train_loader:
            mels        = mels.to(DEVICE, non_blocking=True)
            batch_labels = batch_labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(mels), batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        model.eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for mels, batch_labels in val_loader:
                mels         = mels.to(DEVICE, non_blocking=True)
                batch_labels  = batch_labels.to(DEVICE, non_blocking=True)
                logits        = model(mels)
                val_loss     += criterion(logits, batch_labels).item()
                correct      += (logits.argmax(dim=1) == batch_labels).sum().item()
                total        += batch_labels.size(0)

        t_loss = train_loss / len(train_loader)
        v_loss = val_loss   / len(val_loader)
        v_acc  = correct / total * 100.0

        print(f"Epoch {epoch:3d} | lr {current_lr:.2e} | "
              f"Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.2f}%")

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        history["lr"].append(current_lr)

        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), f"checkpoints/{run_name}/probe_best.pt")
            print(f"  ** Best saved ({best_acc:.2f}%)")

    print(f"\nDomain-Shift Probing Finished. Best Val Accuracy on GSC: {best_acc:.2f}%")

    # ── Plots ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title("Loss (domain-shift probe)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["val_acc"], color="green")
    axes[1].set_title("GSC Val Accuracy (%) — domain shift")
    axes[1].set_xlabel("Epoch")

    axes[2].plot(history["lr"], color="orange")
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")

    plt.tight_layout()
    plot_path = f"logs/{run_name}/probe_domain_shift_plots.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved plots to {plot_path}")

    # ── JSON results ────────────────────────────────────────────────────────
    results = {
        "config":            vars(args),
        "run_name":          run_name,
        "ssl_data":          "LS100 + LS360 (no GSC)",
        "probe_data":        "GSC (domain shift)",
        "best_val_accuracy": best_acc,
        "history":           history,
    }
    json_path = f"logs/{run_name}/probe_domain_shift_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {json_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",       type=str,   default="mae",
                   choices=["mae", "mae_sota", "jepa"])
    p.add_argument("--data_dir",    type=str,   default="./data")
    p.add_argument("--batch_size",  type=int,   default=256)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--lr",          type=float, default=0.1)
    main(p.parse_args())