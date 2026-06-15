import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from dataset import CachedMelDataset

random.seed(42)

ds = CachedMelDataset("./data/mel_cache/gsc")
mel, _ = ds[0]  # [1, 80, T]
mel = mel.squeeze()[:, :96]  # [80, 96] — 5×6 patch grid

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax in axes:
    ax.imshow(mel, aspect='auto', origin='lower', cmap='viridis')
    ax.set_xticks([i * 16 for i in range(7)])
    ax.set_yticks([i * 16 for i in range(6)])
    ax.set_xticklabels(range(7))
    ax.set_yticklabels(range(6))
    ax.set_xlabel("Time patches")
    ax.set_ylabel("Frequency patches")
    ax.grid(True, color='white', linewidth=0.5, alpha=0.3)

# MAE: 22 random patches masked
masked = random.sample(range(30), 22)
for idx in masked:
    row, col = idx // 6, idx % 6
    axes[0].add_patch(mpatches.Rectangle(
        (col * 16, row * 16), 16, 16,
        linewidth=0, facecolor='gray', alpha=0.85
    ))
axes[0].set_title("MAE: Random Masking (75%)", fontsize=12)

# JEPA: one contiguous 2×3 block
# Block 1: rows 1-2, cols 2-4
for row in range(1, 3):
    for col in range(2, 5):
        axes[1].add_patch(mpatches.Rectangle(
            (col * 16, row * 16), 16, 16,
            linewidth=0, facecolor='gray', alpha=0.85
        ))

# Block 2: rows 3-4, cols 0-2
for row in range(3, 5):
    for col in range(0, 3):
        axes[1].add_patch(mpatches.Rectangle(
            (col * 16, row * 16), 16, 16,
            linewidth=0, facecolor='gray', alpha=0.85
        ))
axes[1].set_title("JEPA: Block Masking", fontsize=12)

plt.suptitle("Masking Strategy Comparison (5×6 Patch Grid)", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("logs/masking_comparison.png", dpi=150, bbox_inches='tight')
print("Saved to logs/masking_comparison.png")