import os
import torch
import torchaudio.transforms as T
from torch.utils.data import DataLoader

from dataset import LibriSpeechWaveformDataset, GSCv2Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data"
CACHE_DIR = os.path.join(DATA_DIR, "mel_cache")

mel_transform = T.MelSpectrogram(
    sample_rate=16000, n_fft=400, hop_length=160, n_mels=80
).to(DEVICE)

datasets = {
    "ls100": LibriSpeechWaveformDataset(root=DATA_DIR, url="train-clean-100", download=True),
    "ls360": LibriSpeechWaveformDataset(root=DATA_DIR, url="train-clean-360", download=True),
    "gsc": GSCv2Dataset(root=DATA_DIR, download=True),
}

for name, ds in datasets.items():
    out_dir = os.path.join(CACHE_DIR, name)
    
    if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
        print(f"Skipping {name} (already cached: {len(os.listdir(out_dir))} files)")
        continue

    os.makedirs(out_dir, exist_ok=True)
    loader = DataLoader(ds, batch_size=512, num_workers=8, pin_memory=True)

    print(f"Caching {name} ({len(ds)} samples)...")
    idx = 0
    for waveforms, labels in loader:
        waveforms = waveforms.to(DEVICE)
        with torch.no_grad():
            mels = mel_transform(waveforms)
            mels = 10.0 * torch.log10(torch.clamp(mels, min=1e-10))
            mean = mels.mean(dim=[-2, -1], keepdim=True)
            std = mels.std(dim=[-2, -1], keepdim=True)
            mels = (mels - mean) / (std + 1e-6)

        for mel, label in zip(mels, labels):
            if name == "gsc":
                torch.save({"mel": mel.half().cpu(), "label": label.item()}, f"{out_dir}/{idx:07d}.pt")
            else:
                torch.save(mel.half().cpu(), f"{out_dir}/{idx:07d}.pt")
            idx += 1

    print(f"  Done. {idx} files written to {out_dir}")