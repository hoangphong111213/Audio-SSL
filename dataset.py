import os
import random
import torch
from torch.utils.data import Dataset
from torchaudio.datasets import SPEECHCOMMANDS, LIBRISPEECH
import json

def _crop_or_pad(waveform, target, training):
    n = waveform.shape[1]
    if n < target:
        return torch.nn.functional.pad(waveform, (0, target - n))
    if n > target:
        start = random.randint(0, n - target) if training else 0
        return waveform[:, start:start + target]
    return waveform


class LibriSpeechWaveformDataset(Dataset):
    def __init__(self, root="./data", url="train-clean-100", download=True, is_training=True):
        self.dataset = LIBRISPEECH(root=root, url=url, download=download)
        self.is_training = is_training
        self.target_length = 16000

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        waveform, *_ = self.dataset[index]
        waveform = _crop_or_pad(waveform, self.target_length, self.is_training)
        return waveform, 0


class GSCv2Dataset(Dataset):
    def __init__(self, root="./data", download=True, is_training=True):
        self.dataset = SPEECHCOMMANDS(root=root, download=download)
        self.target_length = 16000
        self.is_training = is_training
        all_labels = sorted({self.dataset[i][2] for i in range(len(self.dataset))})
        self.label_map = {word: i for i, word in enumerate(all_labels)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        waveform, _, label_str, *_ = self.dataset[index]
        waveform = _crop_or_pad(waveform, self.target_length, self.is_training)
        return waveform, self.label_map[label_str]

class CachedMelDataset(Dataset):
    def __init__(self, cache_dir):
        single_file = os.path.join(cache_dir, "gsc.pt")

        if os.path.exists(single_file):
            data = torch.load(single_file, weights_only=True)
            self.mels = data["mels"]
            self.labels = data["labels"]
            self.paths = None
        else:
            self.paths = sorted(
                [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith(".pt")]
            )
            self.mels = None
            self.labels = None
        
        classes_file = os.path.join(cache_dir, "classes.json")
        if os.path.exists(classes_file):
            with open(classes_file) as f:
                self.classes = json.load(f)

    def __len__(self):
        return len(self.mels) if self.mels is not None else len(self.paths)

    def __getitem__(self, idx):
        if self.mels is not None:
            return self.mels[idx].float(), self.labels[idx].item()
        data = torch.load(self.paths[idx], weights_only=True)
        if isinstance(data, dict):
            return data["mel"].float(), data["label"]
        return data.float(), 0