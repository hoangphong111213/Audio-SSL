import os
import math
import json
import argparse
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from dataset import LibriSpeechMelDataset
from mae import AudioMAE
from jepa import AudioJEPA

torch.set_float32_matmul_precision('high')
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
JEPA_MIN_STD = 0.01


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
                print(msg)

    n = len(loader)
    return {"loss": total_loss / n, "aux": total_aux / n if aux is not None else None}


def main(args):
    print(f"Device: {DEVICE} | Model: {args.model.upper()} | Dataset: LibriSpeech")
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(f"logs/{args.model}", exist_ok=True)
    os.makedirs(f"checkpoints/{args.model}", exist_ok=True)

    full_dataset = LibriSpeechMelDataset(root=args.data_dir, url="train-clean-100", download=True, is_training=True)
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    val_dataset.dataset.is_training = False

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    if args.model == "mae":
        model = AudioMAE(use_sota_backbone=False)
    elif args.model == "mae_sota":
        model = AudioMAE(use_sota_backbone=True)
    elif args.model == "jepa":
        model = AudioJEPA()
    else:
        raise ValueError("Unsupported model.")
        
    model = model.to(DEVICE)
    model = torch.compile(model)
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

    best_loss = float("inf")
    start_epoch = 0
    history = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "lr": [], "ema_momentum": [], "train_aux": [], "val_aux": []
    }

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, best_loss = ckpt["epoch"] + 1, ckpt["best_loss"]
        if "history" in ckpt:
            history = ckpt["history"]
        print(f"Resumed from {args.resume} (epoch {start_epoch})")

    for epoch in range(start_epoch, args.epochs):
        lr = cosine_lr(optimizer, epoch, args.epochs, args.warmup_epochs, args.lr)
        momentum = ema_momentum(epoch, args.epochs)
        print(f"\nEpoch {epoch:3d}  lr {lr:.2e}" + (f"  ema {momentum:.4f}" if args.model == "jepa" else ""))

        train = run_epoch(model, train_loader, args.model, optimizer, momentum)
        val = run_epoch(model, val_loader, args.model)

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

        with open(f"logs/{args.model}/history.json", "w") as f:
            json.dump(history, f, indent=4)

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_loss": best_loss,
            "history": history,
        }
        
        torch.save(state, f"checkpoints/{args.model}/last.pt")

        if val["loss"] < best_loss:
            best_loss = val["loss"]
            state["best_loss"] = best_loss
            torch.save(state, f"checkpoints/{args.model}/best.pt")
            print(f"  ** Best saved ({best_loss:.4f})")

    print(f"\nDone. Best val_loss: {best_loss:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="mae", choices=["mae", "mae_sota", "jepa"])
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()
    main(args)