import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from dataset import GSCv2MelDataset
from mae import AudioMAE
from jepa import AudioJEPA

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LinearProbeWrapper(nn.Module):
    def __init__(self, encoder, num_classes, embed_dim=192):
        super().__init__()
        self.encoder = encoder
        
        for param in self.encoder.parameters():
            param.requires_grad = False
            
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
            features = features.mean(dim=1)
            
        return self.head(self.bn(features))


def main(args):
    print(f"Starting Linear Probe for {args.model.upper()} | Device: {DEVICE}")
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    full_train = GSCv2MelDataset(root=args.data_dir, download=False, is_training=False)
    full_val = GSCv2MelDataset(root=args.data_dir, download=False, is_training=False)
    
    dynamic_num_classes = len(full_train.classes)
    
    labels = [os.path.basename(os.path.dirname(p)) for p in full_train.dataset._walker]
    train_idx, val_idx = train_test_split(
        range(len(full_train)), 
        test_size=0.1, 
        stratify=labels, 
        random_state=42
    )

    train_loader = DataLoader(
        Subset(full_train, train_idx), batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        Subset(full_val, val_idx), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

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
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found!")
    
    ckpt = torch.load(ckpt_path, map_location="cpu")["model"]
    new_ckpt = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}

    base_model.load_state_dict(new_ckpt)
    encoder = base_model.encoder if args.model.startswith("mae") else base_model.context_encoder
    
    model = LinearProbeWrapper(encoder, num_classes=dynamic_num_classes, embed_dim=192).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for mels, batch_labels in train_loader:
            mels, batch_labels = mels.to(DEVICE), batch_labels.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(mels)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for mels, batch_labels in val_loader:
                mels, batch_labels = mels.to(DEVICE), batch_labels.to(DEVICE)
                
                logits = model(mels)
                loss = criterion(logits, batch_labels)
                
                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == batch_labels).sum().item()
                total += batch_labels.size(0)
                
        t_loss = train_loss / len(train_loader)
        v_loss = val_loss / len(val_loader)
        v_acc = correct / total * 100.0
        
        print(f"Epoch {epoch:3d} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.2f}%")
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), f"checkpoints/{args.model}/probe_best.pt")

    print(f"\nProbing Finished. Best Val Accuracy: {best_acc:.2f}%")
    
    results_dict = {
        "config": vars(args),
        "best_val_accuracy": best_acc,
        "history": history
    }
    
    json_path = f"logs/{args.model}/probe_results.json"
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=4)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="mae", choices=["mae", "mae_sota", "jepa"])
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight_decay", type=float, default=0.0)
    main(p.parse_args())