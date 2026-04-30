import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from dataset import GSCv2MelDataset

def run_eda():
    print("Loading dataset for EDA (Augmentation Disabled)...")
    dataset = GSCv2MelDataset(download=True, is_training=False)

    # 1. Audit Class Distribution
    print("Auditing class distribution... (This may take a minute to scan the dataset)")
    labels = [dataset.dataset[i][2] for i in range(len(dataset))]
    label_counts = Counter(labels)
    
    # Plotting the Histogram
    plt.figure(figsize=(15, 6))
    classes, counts = zip(*label_counts.most_common())
    plt.bar(classes, counts, color='skyblue', edgecolor='black')
    plt.xticks(rotation=45, ha='right')
    plt.title("GSCv2 Class Distribution (35 Classes)")
    plt.xlabel("Speech Commands")
    plt.ylabel("Number of Utterances")
    plt.tight_layout()
    plt.show()
    
    # 2. Visual Verification of Spectrograms
    print("Generating spectrogram grid...")
    fig, axes = plt.subplots(4, 5, figsize=(15, 10))
    fig.suptitle("Visual Verification of Normalized Mel-Spectrograms", fontsize=16)
    
    # Randomly sample 20 indices
    indices = np.random.choice(len(dataset), 20, replace=False)
    
    for i, ax in enumerate(axes.flatten()):
        mel_spec, label = dataset[indices[i]]
        
        # Squeeze out the channel dimension: (1, 80, 101) -> (80, 101)
        mel_img = mel_spec.squeeze(0).numpy()
        
        # Use origin='lower' so low frequencies are at the bottom
        im = ax.imshow(mel_img, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(f"Label: {label}")
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_eda()