import os
import torch
from typing import List
from datasets import load_dataset
from torch.utils.data import Dataset
from encodec.utils import convert_audio

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
            length_samples=waveform.shape[-1],
            length_tokens=50
        )

        return waveform, audio_config


class GigaSpeechDataset(Dataset):
    def __init__(self, sample_rate: int, size: str = "m", split: str = "train"):
        assert os.environ.get("HF_TOKEN"), "Please set the huggingface API token in the environment (HF_TOKEN)"

        self.dataset = load_dataset(
            "speechcolab/gigaspeech",
            size,
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN"),
            # streaming=True
        )[split]
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        ds_idx = self.dataset[idx]
        audio_input = torch.Tensor(ds_idx["audio"]["array"].reshape(1, -1))
        sr = ds_idx["audio"]["sampling_rate"]

        waveform = convert_audio(audio_input, sr, self.sample_rate, 1)

        audio_config = AudioConfig(
            file_name=ds_idx["audio"]["path"],
            length_seconds=waveform.shape[-1] / self.sample_rate,
            length_samples=waveform.shape[-1]
        )

        return waveform, audio_config
