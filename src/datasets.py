from typing import List
from torch.utils.data import Dataset

from .configs import AudioConfig
from .utils import process_audio

class AudioDataset(Dataset):
    def __init__(self, audio_files: List[str], sample_rate: float, channels: int):
        self.audio_files = audio_files
        self.sample_rate = sample_rate
        self.channels = channels

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        waveform = process_audio(audio_path, self.sample_rate)
        audio_config = AudioConfig(
            file_name=audio_path,
            length_seconds=waveform.shape[-1] / self.sample_rate,
            length_samples=waveform.shape[-1]
        )

        return waveform, audio_config
