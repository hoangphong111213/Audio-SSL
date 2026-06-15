import os
import torch
import torchaudio.transforms as T
from torch.utils.data import DataLoader
import json

from dataset import LibriSpeechWaveformDataset, GSCv2Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data"
CACHE_DIR = os.path.join(DATA_DIR, "mel_cache")

mel_transform = T.MelSpectrogram(
    sample_rate=16000, n_fft=400, hop_length=160, n_mels=80
).to(DEVICE)

datasets = {
    #"ls100": LibriSpeechWaveformDataset(root=DATA_DIR, url="train-clean-100", download=True),
    #"ls360": LibriSpeechWaveformDataset(root=DATA_DIR, url="train-clean-360", download=True),
    "gsc": GSCv2Dataset(root=DATA_DIR, download=True),
}

for name, ds in datasets.items():
    out_dir = os.path.join(CACHE_DIR, name)

    if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
        print(f"Skipping {name} (already cached: {len(os.listdir(out_dir))} files)")
        continue

    os.makedirs(out_dir, exist_ok=True)
    loader = DataLoader(ds, batch_size=2048, num_workers=os.cpu_count()//4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    print(f"Caching {name} ({len(ds)} samples)...")

    all_mels, all_labels = [], []

    for waveforms, labels in loader:
        waveforms = waveforms.to(DEVICE, non_blocking=True)
        with torch.no_grad():
            mels = mel_transform(waveforms)
            mels = 10.0 * torch.log10(torch.clamp(mels, min=1e-10))
            mean = mels.mean(dim=[-2, -1], keepdim=True)
            std = mels.std(dim=[-2, -1], keepdim=True)
            mels = (mels - mean) / (std + 1e-6)

        all_mels.append(mels.half().cpu())
        all_labels.append(labels.cpu())
        del mels, waveforms
        torch.cuda.empty_cache()

    all_mels = torch.cat(all_mels)
    all_labels = torch.cat(all_labels)

    if name == "gsc":
        torch.save({"mels": all_mels, "labels": all_labels}, f"{out_dir}/gsc.pt")
        with open(f"{out_dir}/classes.json", "w") as f:
            json.dump({k: v for k, v in {v: k for k, v in ds.label_map.items()}.items()}, f)
        print(f"  Done. Saved {len(all_mels)} samples to {out_dir}/gsc.pt")
    else:
        for idx, mel in enumerate(all_mels):
            torch.save(mel, f"{out_dir}/{idx:07d}.pt")
        print(f"  Done. {len(all_mels)} files written to {out_dir}")