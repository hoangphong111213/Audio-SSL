# baseline_random_probe.py
import os, argparse, json, torch
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from dataset import CachedMelDataset
from mae import AudioMAE

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Reuse the exact probe head from linear_probe.py so the comparison is apples-to-apples
from linear_probe import AttentivePooling, LinearProbeWrapper, build_probe_datasets, build_loader

def main(args):
    print(f"Random-init frozen-encoder probe | Device: {DEVICE}")
    os.makedirs("logs/random", exist_ok=True)
    os.makedirs("checkpoints/random", exist_ok=True)

    train_dataset, val_dataset = build_probe_datasets(args)
    train_loader = build_loader(train_dataset, args, shuffle=True)
    val_loader   = build_loader(val_dataset, args, shuffle=False)

    all_labels = set()
    for _, label in DataLoader(train_dataset, batch_size=1024, num_workers=4):
        all_labels.update(label.tolist())
    num_classes = len(all_labels)
    print(f"Detected {num_classes} classes")

    # >>> The whole point: NO checkpoint load. Fresh random encoder. <
    torch.manual_seed(args.seed)
    base_model = AudioMAE(use_sota_backbone=False)
    encoder = base_model.encoder
    print(f"Using RANDOM-INITIALIZED encoder (seed={args.seed}) — no pretraining.")

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
        for mels, y in train_loader:
            mels, y = mels.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(mels), y)
            loss.backward(); optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        cur_lr = scheduler.get_last_lr()[0]

        model.eval(); val_loss = correct = total = 0
        with torch.no_grad():
            for mels, y in val_loader:
                mels, y = mels.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                logits = model(mels)
                val_loss += criterion(logits, y).item()
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)

        t, v, a = train_loss/len(train_loader), val_loss/len(val_loader), correct/total*100
        print(f"Epoch {epoch:3d} | lr {cur_lr:.2e} | Train {t:.4f} | Val {v:.4f} | Acc {a:.2f}%")
        for k, val in zip(history, [t, v, a, cur_lr]): history[k].append(val)
        if a > best_acc:
            best_acc = a
            torch.save(model.state_dict(), "checkpoints/random/probe_best.pt")

    print(f"\nRandom-init probe finished. Best Val Acc: {best_acc:.2f}%")
    with open("logs/random/probe_results.json", "w") as f:
        json.dump({"config": vars(args), "best_val_accuracy": best_acc, "history": history}, f, indent=4)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())