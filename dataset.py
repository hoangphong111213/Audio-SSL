import torch
import torchaudio
import torchaudio.transforms as T
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import Dataset, DataLoader

class GSCv2MelDataset(Dataset):
    """
    Google Speech Commands v2 Dataset with Mel-Spectrogram Pipeline and SpecAugment.
    """
    def __init__(self, root="./data", download=True, is_training=True):
        self.dataset = SPEECHCOMMANDS(root=root, download=download)
        self.is_training = is_training
        
        self.sample_rate = 16000
        self.target_length = self.sample_rate * 1 # Enforces strictly 1-second clips at 16kHz
        
        # Mel-spectrogram: Outputs spatial representation of audio. Shape: (n_mels, time_steps) -> (80, 101)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=400,          
            hop_length=160,     
            n_mels=80           
        )
        
        # SpecAugment: Randomly dropping continuous blocks of time/frequency
        self.freq_masking = T.FrequencyMasking(freq_mask_param=15)
        self.time_masking = T.TimeMasking(time_mask_param=35)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[index]
        
        # Enforce exact target length via zero-padding or truncation. Shape: (1, 16000)
        if waveform.shape[1] < self.target_length:
            pad_amount = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        elif waveform.shape[1] > self.target_length:
            waveform = waveform[:, :self.target_length]

        mel_spec = self.mel_transform(waveform) # Shape: (1, 80, 101)
        
        # Log scaling
        mel_spec = 10.0 * torch.log10(torch.clamp(mel_spec, min=1e-10))
        # Normalization
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)
        
        if self.is_training:
            mel_spec = self.freq_masking(mel_spec)
            mel_spec = self.time_masking(mel_spec)
            
        return mel_spec, label


if __name__ == "__main__":
    train_dataset = GSCv2MelDataset(download=True, is_training=True)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=256, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    mels, labels = next(iter(train_loader))
    
    # Verify batched tensor dimensions. 
    print(f"Batch shape: {mels.shape}") # [Batch_Size, Channels, Mels, Time_Steps]
    print(f"Mean: {mels.mean().item():.4f}, Std: {mels.std().item():.4f}")