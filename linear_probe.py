import os
import argparse
import json
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset

from dataset import CachedMelDataset
from mae import AudioMAE
from jepa import AudioJEPA

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AttentivePooling(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
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
        self.bn = nn.BatchNorm1d(embed_dim, affine=False)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            pad_w = (16 - x.shape[3] % 16) % 16
            x = F.pad(x, (0, pad_w))
            features = self.encoder(x)
        pooled = self.pool(features)
        return self.head(self.bn(pooled))


def build_probe_datasets(args):
    # GSC cached mels only — labelled split used for probing
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


def main(args):
    print(f"Starting Linear Probe for {args.model.upper()} | Device: {DEVICE}")
    os.makedirs(f"logs/{args.model}", exist_ok=True)
    os.makedirs(f"checkpoints/{args.model}", exist_ok=True)

    train_dataset, val_dataset = build_probe_datasets(args)
    train_loader = build_loader(train_dataset, args, shuffle=True)
    val_loader = build_loader(val_dataset, args, shuffle=False)

    all_labels = set()
    for _, label in DataLoader(train_dataset, batch_size=1024, num_workers=4):
        all_labels.update(label.tolist())
    num_classes = len(all_labels)
    print(f"Detected {num_classes} classes")

    if args.model == "mae":
        base_model = AudioMAE(use_sota_backbone=False)
    elif args.model == "mae_sota":
        base_model = AudioMAE(use_sota_backbone=True)
    elif args.model == "jepa":
        base_model = AudioJEPA()
    else:
        raise ValueError("Unsupported model for probing")

    ckpt_path = f"checkpoints/{args.model}/last.pt"
    if not os.path.isfile(ckpt_path):
        ckpt_path = f"checkpoints/{args.model}/best.pt"
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found under checkpoints/{args.model}/")

    ckpt = torch.load(ckpt_path, map_location="cpu")["model"]
    ckpt = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}
    base_model.load_state_dict(ckpt)
    print(f"Loaded checkpoint: {ckpt_path}")

    if args.model in ("mae", "mae_sota"):
        encoder = base_model.encoder
    else:
        encoder = base_model.context_encoder

    model = LinearProbeWrapper(encoder, num_classes=num_classes, embed_dim=192).to(DEVICE)

    probe_params = list(model.pool.parameters()) + list(model.head.parameters())
    optimizer = optim.SGD(probe_params, lr=args.lr, momentum=0.9, weight_decay=0.0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for mels, batch_labels in train_loader:
            mels = mels.to(DEVICE, non_blocking=True)
            batch_labels = batch_labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(mels), batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for mels, batch_labels in val_loader:
                mels = mels.to(DEVICE, non_blocking=True)
                batch_labels = batch_labels.to(DEVICE, non_blocking=True)

                logits = model(mels)
                val_loss += criterion(logits, batch_labels).item()
                correct += (logits.argmax(dim=1) == batch_labels).sum().item()
                total += batch_labels.size(0)

        t_loss = train_loss / len(train_loader)
        v_loss = val_loss / len(val_loader)
        v_acc = correct / total * 100.0

        print(f"Epoch {epoch:3d} | lr {current_lr:.2e} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.2f}%")

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        history["lr"].append(current_lr)

        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), f"checkpoints/{args.model}/probe_best.pt")
            print(f"  ** Best saved ({best_acc:.2f}%)")

    print(f"\nProbing Finished. Best Val Accuracy: {best_acc:.2f}%")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["val_acc"], color="green")
    axes[1].set_title("Val Accuracy (%)")
    axes[1].set_xlabel("Epoch")

    axes[2].plot(history["lr"], color="orange")
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(f"logs/{args.model}/probe_plots.png", dpi=150)
    plt.close()
    print(f"Saved plots to logs/{args.model}/probe_plots.png")

    with open(f"logs/{args.model}/probe_results.json", "w") as f:
        json.dump({"config": vars(args), "best_val_accuracy": best_acc, "history": history}, f, indent=4)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="mae", choices=["mae", "mae_sota", "jepa"])
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.1)
    main(p.parse_args())