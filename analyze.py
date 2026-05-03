import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset

from dataset import GSCv2MelDataset
from mae import AudioMAE
from jepa import AudioJEPA
from linear_probe import LinearProbeWrapper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    print(f"Starting Analysis for {args.model.upper()}...")
    
    full_val = GSCv2MelDataset(root=args.data_dir, download=False, is_training=False)
    dynamic_num_classes = len(full_val.classes)
    
    subset_idx = torch.randperm(len(full_val))[:2000].tolist()
    loader = DataLoader(Subset(full_val, subset_idx), batch_size=args.batch_size, shuffle=False)
    
    if args.model == "mae":
        base_model = AudioMAE(use_sota_backbone=False)
    elif args.model == "mae_sota":
        base_model = AudioMAE(use_sota_backbone=True)
    elif args.model == "jepa":
        base_model = AudioJEPA()
        
    encoder = base_model.encoder if args.model.startswith("mae") else base_model.context_encoder
    model = LinearProbeWrapper(encoder, num_classes=dynamic_num_classes, embed_dim=192).to(DEVICE)
    
    ckpt_path = f"checkpoints/{args.model}/probe_best.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    
    all_features, all_labels, all_preds = [], [], []
    
    print("Extracting features and predictions...")
    with torch.no_grad():
        for mels, labels in loader:
            mels = mels.to(DEVICE)
            pad_w = (16 - mels.shape[3] % 16) % 16
            mels = torch.nn.functional.pad(mels, (0, pad_w))
            
            features = model.encoder(mels).mean(dim=1)
            preds = model.head(features).argmax(dim=1)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
            all_preds.append(preds.cpu().numpy())
            
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    
    class_correct = np.zeros(dynamic_num_classes)
    class_total = np.zeros(dynamic_num_classes)
    for l, p in zip(labels, preds):
        class_total[l] += 1
        if l == p:
            class_correct[l] += 1
            
    accuracies = np.divide(class_correct, class_total, out=np.zeros_like(class_correct), where=class_total!=0)
    
    print("\n--- Top 5 Best Performing Classes ---")
    for c in accuracies.argsort()[-5:][::-1]:
        print(f"Class '{full_val.classes[c]}' (ID {c}): {accuracies[c]*100:.1f}%")
        
    print("\n--- Top 5 Worst Performing Classes ---")
    for c in accuracies.argsort()[:5]:
        print(f"Class '{full_val.classes[c]}' (ID {c}): {accuracies[c]*100:.1f}%")

    print("\nRunning t-SNE (this might take a minute)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # Map label IDs to string names for the legend
    label_names = [full_val.classes[l] for l in labels]
    
    plt.figure(figsize=(14, 10))
    sns.scatterplot(
        x=features_2d[:, 0], y=features_2d[:, 1],
        hue=label_names, palette="tab20", legend="full", s=30, alpha=0.8
    )
    plt.title(f"t-SNE Projection of {args.model.upper()} Latent Space")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2, ncol=2)
    plt.tight_layout()
    plt.savefig(f"logs/{args.model}/tsne.png", dpi=300)
    print(f"Saved t-SNE plot to logs/{args.model}/tsne.png")

    top_5_indices = [int(c) for c in accuracies.argsort()[-5:][::-1]]
    worst_5_indices = [int(c) for c in accuracies.argsort()[:5]]

    class_stats = {}
    for c in range(dynamic_num_classes):
        class_stats[full_val.classes[c]] = {
            "class_id": int(c),
            "correct": int(class_correct[c]),
            "total": int(class_total[c]),
            "accuracy": float(accuracies[c])
        }

    analysis_dict = {
        "config": vars(args),
        "top_5_classes": [
            {"name": full_val.classes[c], "accuracy": float(accuracies[c])} 
            for c in top_5_indices
        ],
        "worst_5_classes": [
            {"name": full_val.classes[c], "accuracy": float(accuracies[c])} 
            for c in worst_5_indices
        ],
        "all_class_statistics": class_stats
    }

    json_path = f"logs/{args.model}/analysis_results.json"
    with open(json_path, "w") as f:
        json.dump(analysis_dict, f, indent=4)
    print(f"Saved analysis results to {json_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="mae", choices=["mae", "mae_sota", "jepa"])
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--batch_size", type=int, default=128)
    main(p.parse_args())