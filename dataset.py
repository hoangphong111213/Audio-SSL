import os
import random
import torch
import torchaudio.transforms as T
from torchaudio.datasets import SPEECHCOMMANDS, LIBRISPEECH
from torch.utils.data import Dataset, DataLoader

class LibriSpeechMelDataset(Dataset):
    def __init__(self, root="./data", url="train-clean-100", download=True, is_training=True):
        self.dataset = LIBRISPEECH(root=root, url=url, download=download)
        self.is_training = is_training
        self.sample_rate = 16000
        self.target_length = self.sample_rate * 3

        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=80
        )
        self.freq_masking = T.FrequencyMasking(freq_mask_param=15)
        self.time_masking = T.TimeMasking(time_mask_param=35)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        waveform, _, _, _, _, _ = self.dataset[index]

        if waveform.shape[1] < self.target_length:
            pad_amount = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        elif waveform.shape[1] > self.target_length:
            start = random.randint(0, waveform.shape[1] - self.target_length) if self.is_training else 0
            waveform = waveform[:, start:start + self.target_length]

        mel_spec = self.mel_transform(waveform)
        mel_spec = 10.0 * torch.log10(torch.clamp(mel_spec, min=1e-10))
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)

        if self.is_training:
            mel_spec = self.freq_masking(mel_spec)
            mel_spec = self.time_masking(mel_spec)

        return mel_spec, 0


class GSCv2MelDataset(Dataset):
    def __init__(self, root="./data", download=True, is_training=True):
        self.dataset = SPEECHCOMMANDS(root=root, download=download)
        self.is_training = is_training
        
        self.classes = sorted(list(set(
            os.path.basename(os.path.dirname(p)) for p in self.dataset._walker
        )))
        
        self.sample_rate = 16000
        self.target_length = self.sample_rate * 1
        
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=400,          
            hop_length=160,     
            n_mels=80           
        )
        
        self.freq_masking = T.FrequencyMasking(freq_mask_param=15)
        self.time_masking = T.TimeMasking(time_mask_param=35)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        waveform, _, label, _, _ = self.dataset[index]
        
        if waveform.shape[1] < self.target_length:
            pad_amount = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        elif waveform.shape[1] > self.target_length:
            waveform = waveform[:, :self.target_length]

        mel_spec = self.mel_transform(waveform)
        mel_spec = 10.0 * torch.log10(torch.clamp(mel_spec, min=1e-10))
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)
        
        if self.is_training:
            mel_spec = self.freq_masking(mel_spec)
            mel_spec = self.time_masking(mel_spec)
            
        label_idx = self.classes.index(label)
        return mel_spec, label_idx