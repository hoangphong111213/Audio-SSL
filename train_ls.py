"""
train_ls.py  —  SSL pre-training on LibriSpeech ONLY (no GSC).
Outputs go to  checkpoints/<model>_ls/  and  logs/<model>_ls/
so they never collide with the original mixed-data runs.

Usage:
    python train_ls.py --model mae      # → mae_ls
    python train_ls.py --model jepa     # → jepa_ls
    python train_ls.py --model mae_sota # → mae_sota_ls
"""

import os
import math
import json
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
import torchaudio.transforms as T
import matplotlib.pyplot as plt

from dataset import CachedMelDataset
from mae import AudioMAE
from jepa import AudioJEPA

torch.set_float32_matmul_precision("high")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
JEPA_MIN_STD = 0.01

freq_masking = T.FrequencyMasking(freq_mask_param=15)
time_masking  = T.TimeMasking(time_mask_param=35)


# ---------------------------------------------------------------------------
# Helpers (identical to train.py)
# ---------------------------------------------------------------------------

def cosine_lr(optimizer, epoch, total_epochs, warmup_epochs, base_lr, min_lr=1e-6):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        t = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def ema_momentum(epoch, total_epochs, start=0.996, end=1.0):
    t = epoch / total_epochs
    return end - (end - start) * 0.5 * (1 + math.cos(math.pi * t))


def run_epoch(model, loader, model_type, optimizer=None, momentum=None, augment=False):
    model.train() if optimizer is not None else model.eval()

    total_loss = 0.0
    total_aux  = 0.0
    has_aux    = False

    with torch.set_grad_enabled(optimizer is not None):
        for step, (mels, _) in enumerate(loader):
            mels = mels.to(DEVICE, non_blocking=True)

            if augment:
                with torch.no_grad():
                    mels = freq_masking(mels)
                    mels = time_masking(mels)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if model_type in ("mae", "mae_sota"):
                    loss, _, _ = model(mels)
                    aux = None
                elif model_type == "jepa":
                    loss, std_val = model(mels)
                    aux = std_val
                    has_aux = True
                else:
                    raise ValueError("Unsupported model type")

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                if hasattr(model, "update_target_encoder"):
                    model.update_target_encoder(momentum=momentum)

            total_loss += loss.item()
            if aux is not None:
                total_aux += aux.item() if torch.is_tensor(aux) else aux

            if optimizer is not None and step % 50 == 0:
                msg = f"  step {step:4d}  loss {loss.item():.4f}"
                if model_type == "jepa":
                    msg += f"  std {(aux.item() if torch.is_tensor(aux) else aux):.4f}"
                print(msg)

    n = len(loader)
    return {
        "loss": total_loss / n,
        "aux":  (total_aux / n) if has_aux else None,
    }


# ---------------------------------------------------------------------------
# LS-ONLY dataset builder  (key difference from train.py)
# ---------------------------------------------------------------------------

def build_ssl_datasets(args):
    """Pre-train on LibriSpeech only — GSC is held out for domain-shift probing."""
    full = ConcatDataset([
        CachedMelDataset(os.path.join(args.data_dir, "mel_cache", "ls100")),
        CachedMelDataset(os.path.join(args.data_dir, "mel_cache", "ls360")),
    ])

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
        prefetch_factor=args.prefetch_factor,
        drop_last=shuffle,
        persistent_workers=True,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_plots(history, run_name):
    is_jepa = len(history["val_aux"]) > 0
    ncols = 3 if is_jepa else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))

    axes[0].plot(history["epoch"], history["train_loss"], label="Train")
    axes[0].plot(history["epoch"], history["val_loss"],   label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["lr"], color="orange")
    axes[1].set_title("Learning Rate")
    axes[1].set_xlabel("Epoch")

    if is_jepa:
        axes[2].plot(history["epoch"], history["train_aux"], label="Train std")
        axes[2].plot(history["epoch"], history["val_aux"],   label="Val std")
        axes[2].axhline(JEPA_MIN_STD, color="red", linestyle="--", linewidth=0.8, label="Collapse threshold")
        axes[2].set_title("JEPA Predictor Std")
        axes[2].set_xlabel("Epoch")
        axes[2].legend()

    plt.tight_layout()
    path = f"logs/{run_name}/training_plots.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved plots to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # Append '_ls' suffix so all outputs are isolated from original runs
    run_name = f"{args.model}_ls"

    print(f"Device: {DEVICE} | Model: {run_name.upper()} | Dataset: LS100 + LS360 (no GSC)")
    os.makedirs(args.data_dir,           exist_ok=True)
    os.makedirs(f"logs/{run_name}",      exist_ok=True)
    os.makedirs(f"checkpoints/{run_name}", exist_ok=True)

    train_dataset, val_dataset = build_ssl_datasets(args)
    train_loader = build_loader(train_dataset, args, shuffle=True)
    val_loader   = build_loader(val_dataset,   args, shuffle=False)

    if args.model == "mae":
        model = AudioMAE(use_sota_backbone=False)
    elif args.model == "mae_sota":
        model = AudioMAE(use_sota_backbone=True)
    elif args.model == "jepa":
        model = AudioJEPA()
    else:
        raise ValueError("Unsupported model.")

    model = model.to(DEVICE)
    model = torch.compile(model, fullgraph=False)

    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim == 1 or name.endswith(".bias") else decay).append(p)

    optimizer = optim.AdamW(
        [{"params": decay,    "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    best_loss  = float("inf")
    start_epoch = 0
    history = {
        "epoch": [], "train_loss": [], "val_loss": [], "lr": [],
        "ema_momentum": [], "train_aux": [], "val_aux": [],
    }

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_loss   = ckpt["best_loss"]
        if "history" in ckpt:
            history = ckpt["history"]
        print(f"Resumed from {args.resume} (epoch {start_epoch})")

    for epoch in range(start_epoch, args.epochs):
        lr       = cosine_lr(optimizer, epoch, args.epochs, args.warmup_epochs, args.lr)
        momentum = ema_momentum(epoch, args.epochs)
        print(f"\nEpoch {epoch:3d}  lr {lr:.2e}" +
              (f"  ema {momentum:.4f}" if args.model == "jepa" else ""))

        train = run_epoch(model, train_loader, args.model, optimizer, momentum, augment=args.augment)
        val   = run_epoch(model, val_loader,   args.model)

        summary = f"=> train {train['loss']:.4f}  val {val['loss']:.4f}"
        if args.model == "jepa":
            summary += f"  val_std {val['aux']:.4f}"
            if val["aux"] < JEPA_MIN_STD:
                summary += "  !! COLLAPSE"
        print(summary)

        history["epoch"].append(epoch)
        history["train_loss"].append(train["loss"])
        history["val_loss"].append(val["loss"])
        history["lr"].append(lr)

        if args.model == "jepa":
            history["ema_momentum"].append(momentum)
            history["train_aux"].append(train["aux"])
            history["val_aux"].append(val["aux"])

        with open(f"logs/{run_name}/history.json", "w") as f:
            json.dump(history, f, indent=4)

        save_plots(history, run_name)

        state = {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_loss": best_loss,
            "history":   history,
        }

        torch.save(state, f"checkpoints/{run_name}/last.pt")

        if val["loss"] < best_loss:
            best_loss = val["loss"]
            state["best_loss"] = best_loss
            torch.save(state, f"checkpoints/{run_name}/best.pt")
            print(f"  ** Best saved ({best_loss:.4f})")

    print(f"\nDone. Best val_loss: {best_loss:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",           type=str,   default="mae",
                   choices=["mae", "mae_sota", "jepa"])
    p.add_argument("--data_dir",        type=str,   default="./data")
    p.add_argument("--batch_size",      type=int,   default=1024)
    p.add_argument("--num_workers",     type=int,   default=64)
    p.add_argument("--prefetch_factor", type=int,   default=1)
    p.add_argument("--epochs",          type=int,   default=200)
    p.add_argument("--warmup_epochs",   type=int,   default=10)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--weight_decay",    type=float, default=0.05)
    p.add_argument("--resume",          type=str,   default=None)
    p.add_argument("--augment",         action="store_true", default=False)
    main(p.parse_args())