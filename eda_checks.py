import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from dataset import GSCv2MelDataset

def run_eda():
    os.makedirs("logs", exist_ok=True)
    
    print("Loading dataset for EDA (Augmentation Disabled)...")
    dataset = GSCv2MelDataset(download=True, is_training=False)

    print("Auditing class distribution... (This may take a minute to scan the dataset)")
    labels = [dataset.dataset[i][2] for i in range(len(dataset))]
    label_counts = Counter(labels)
    
    plt.figure(figsize=(15, 6))
    classes, counts = zip(*label_counts.most_common())
    plt.bar(classes, counts, color='skyblue', edgecolor='black')
    plt.xticks(rotation=45, ha='right')
    plt.title("GSCv2 Class Distribution (35 Classes)")
    plt.xlabel("Speech Commands")
    plt.ylabel("Number of Utterances")
    plt.tight_layout()
    plt.savefig("logs/class_distribution.png", dpi=300)
    plt.close()
    
    print("Generating spectrogram grid...")
    fig, axes = plt.subplots(4, 5, figsize=(15, 10))
    fig.suptitle("Visual Verification of Normalized Mel-Spectrograms", fontsize=16)
    
    indices = np.random.choice(len(dataset), 20, replace=False)
    
    for i, ax in enumerate(axes.flatten()):
        mel_spec, label = dataset[indices[i]]
        mel_img = mel_spec.squeeze(0).numpy()
        im = ax.imshow(mel_img, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(f"Label: {label}")
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig("logs/spectrogram_grid.png", dpi=300)
    plt.close()
    
    print("EDA plots successfully saved to the 'logs' directory.")

if __name__ == "__main__":
    run_eda()