import os
import math
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from dataset import GSCv2MelDataset
from mae import AudioMAE
from jepa import AudioJEPA
from dino import AudioDINO

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
JEPA_MIN_STD = 0.01
DINO_MIN_ENTROPY = 0.50


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


def run_epoch(model, loader, model_type, optimizer=None, momentum=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = total_aux = 0.0

    with torch.set_grad_enabled(is_train):
        for step, (mels, _) in enumerate(loader):
            mels = mels.to(DEVICE)

            if model_type in ("mae", "mae_sota"):
                loss, _, _ = model(mels)
                aux = None
            elif model_type == "jepa":
                loss, std = model(mels)
                aux = std
            else:
                loss, entropy = model(mels)
                aux = entropy

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if hasattr(model, "update_target_encoder"):
                    model.update_target_encoder(momentum=momentum)

            total_loss += loss.item()
            if aux is not None:
                total_aux += aux

            if is_train and step % 50 == 0:
                msg = f"  step {step:4d}  loss {loss.item():.4f}"
                if model_type == "jepa":
                    msg += f"  std {aux:.4f}"
                elif model_type == "dino":
                    msg += f"  entropy {aux:.4f}"
                print(msg)

    n = len(loader)
    return {"loss": total_loss / n, "aux": total_aux / n if aux is not None else None}


def main(args):
    print(f"Device: {DEVICE} | Model: {args.model.upper()}")

    full_train = GSCv2MelDataset(root=args.data_dir, download=True, is_training=True)
    full_val = GSCv2MelDataset(root=args.data_dir, download=False, is_training=False)
    n_val = int(0.1 * len(full_train))
    idx = torch.randperm(len(full_train), generator=torch.Generator().manual_seed(42)).tolist()

    train_loader = DataLoader(
        Subset(full_train, idx[n_val:]), batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        Subset(full_val, idx[:n_val]), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    if args.model == "mae":
        model = AudioMAE(use_sota_backbone=False)
    elif args.model == "mae_sota":
        model = AudioMAE(use_sota_backbone=True)
    elif args.model == "jepa":
        model = AudioJEPA()
    else:
        model = AudioDINO()
    model = model.to(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim == 1 or name.endswith(".bias") else decay).append(p)
    optimizer = optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )

    writer = SummaryWriter(log_dir=f"runs/{args.model}")
    best_loss = float("inf")
    start_epoch = 0

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, best_loss = ckpt["epoch"] + 1, ckpt["best_loss"]
        print(f"Resumed from {args.resume} (epoch {start_epoch})")

    for epoch in range(start_epoch, args.epochs):
        lr = cosine_lr(optimizer, epoch, args.epochs, args.warmup_epochs, args.lr)
        momentum = ema_momentum(epoch, args.epochs)
        print(f"\nEpoch {epoch:3d}  lr {lr:.2e}  ema {momentum:.4f}")

        train = run_epoch(model, train_loader, args.model, optimizer, momentum)
        val = run_epoch(model, val_loader, args.model)

        summary = f"=> train {train['loss']:.4f}  val {val['loss']:.4f}"
        if args.model == "jepa":
            summary += f"  val_std {val['aux']:.4f}"
            if val["aux"] < JEPA_MIN_STD:
                summary += "  !! COLLAPSE"
        elif args.model == "dino":
            summary += f"  val_entropy {val['aux']:.4f}"
            if val["aux"] < DINO_MIN_ENTROPY:
                summary += "  !! COLLAPSE"
        print(summary)

        writer.add_scalars("Loss", {"train": train["loss"], "val": val["loss"]}, epoch)
        writer.add_scalar("LR", lr, epoch)
        writer.add_scalar("EMA_momentum", momentum, epoch)
        if args.model == "jepa":
            writer.add_scalars("JEPA_std", {"train": train["aux"], "val": val["aux"]}, epoch)
        elif args.model == "dino":
            writer.add_scalars("DINO_entropy", {"train": train["aux"], "val": val["aux"]}, epoch)

        state = {"epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "best_loss": best_loss}
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(state, f"checkpoints/{args.model}_epoch_{epoch:03d}.pt")
        if val["loss"] < best_loss:
            best_loss = val["loss"]
            torch.save(state, f"checkpoints/{args.model}_best.pt")
            print(f"  ** Best saved ({best_loss:.4f})")

    writer.close()
    print(f"\nDone. Best val_loss: {best_loss:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="mae", choices=["mae", "mae_sota", "jepa", "dino"])
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()
    args.lr = args.lr * args.batch_size / 256
    main(args)