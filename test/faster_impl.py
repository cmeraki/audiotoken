import torch
import torch.nn.functional as F
from typing import Optional
import math
import pdb

device = 'cuda'

# Helper functions
def hertz_to_mel(freq):
    return 1127.0 * torch.log(1.0 + (freq / 700.0))


def mel_to_hertz(mels):
    return 700.0 * (torch.exp(mels / 1127.0) - 1.0)


def _create_triangular_filter_bank(fft_freqs: torch.Tensor, filter_freqs: torch.Tensor) -> torch.Tensor:
    filter_diff = torch.diff(filter_freqs)
    slopes = filter_freqs.unsqueeze(0) - fft_freqs.unsqueeze(1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    return torch.maximum(torch.zeros(1), torch.minimum(down_slopes, up_slopes))


def mel_filter_bank(
    num_frequency_bins: int,
    num_mel_filters: int,
    min_frequency: float,
    max_frequency: float,
    sampling_rate: int
) -> torch.Tensor:
    mel_min = hertz_to_mel(torch.tensor(min_frequency))
    mel_max = hertz_to_mel(torch.tensor(max_frequency))
    mel_freqs = torch.linspace(mel_min, mel_max, num_mel_filters + 2)
    filter_freqs = mel_to_hertz(mel_freqs)

    filter_freqs = mel_freqs

    fft_bin_width = sampling_rate / (num_frequency_bins * 2)
    fft_freqs = hertz_to_mel(fft_bin_width * torch.arange(num_frequency_bins))

    return _create_triangular_filter_bank(fft_freqs, filter_freqs)


def spectrogram(
    waveform: torch.Tensor,
    window: torch.Tensor,
    frame_length: int,
    hop_length: int,
    fft_length: int = 512,
    power: float = 2.0,
    center: bool = False,
    preemphasis: float = 0.97,
    mel_filters: Optional[torch.Tensor] = None,
    log_mel: str = "log",
    mel_floor: float = 1.192092955078125e-07,
    remove_dc_offset: bool = True,
) -> torch.Tensor:
    device = waveform.device
    dtype = torch.float32

    waveform = waveform.squeeze(0)

    num_frames = int(1 + math.floor((waveform.shape[-1] - frame_length) / hop_length))
    num_frequency_bins = (fft_length // 2) + 1
    spectrogram = torch.empty((num_frames, num_frequency_bins), dtype=torch.cfloat, device=device)

    buffer = torch.zeros(fft_length, device=device)

    timestep = 0
    for frame_idx in range(num_frames):
        buffer[:frame_length] = waveform[timestep : timestep + frame_length]

        if remove_dc_offset:
            buffer[:frame_length] = buffer[:frame_length] - buffer[:frame_length].mean()

        if preemphasis is not None:
            buffer[1:frame_length] -= preemphasis * buffer[: frame_length - 1]
            buffer[0] *= 1 - preemphasis

        buffer[:frame_length] *= window

        spectrogram[frame_idx] = torch.fft.rfft(buffer)
        timestep += hop_length

    # Compute power spectrogram
    spectrogram = spectrogram.abs().pow(power)

    if mel_filters is not None:
        # Apply mel filterbank
        spectrogram = torch.matmul(spectrogram, mel_filters)
        spectrogram = torch.maximum(spectrogram, torch.tensor(mel_floor, device=device, dtype=dtype))

        # Apply log
        if log_mel == "log":
            spectrogram = torch.log(spectrogram)

    return spectrogram.to(dtype)


class FasterSeamlessM4TFeatureExtractor(torch.nn.Module):
    def __init__(
        self,
        feature_size=80,
        sampling_rate=16000,
        num_mel_bins=80,
        padding_value=0.0,
        stride=2,
        device: str = 'cpu',
        **kwargs,
    ):
        super().__init__()
        self.num_mel_bins = num_mel_bins
        self.return_attention_mask = True
        self.stride = stride
        self.device = device

        mel_filters = mel_filter_bank(
            num_frequency_bins=256,
            num_mel_filters=self.num_mel_bins,
            min_frequency=20,
            max_frequency=sampling_rate // 2,
            sampling_rate=sampling_rate,
        ).to(device)

        self.register_buffer('mel_filters', F.pad(mel_filters, (0, 0, 0, 1)))
        window = torch.pow(torch.hann_window(400, periodic=False, device=device), 0.85)
        self.register_buffer('window', window)

    def _extract_fbank_features(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform * (2**15)  # Kaldi compliance: 16-bit signed integers
        features = spectrogram(
            waveform,
            self.window,
            frame_length=400,
            hop_length=160,
            fft_length=512,
            power=2.0,
            center=False,
            preemphasis=0.97,
            mel_filters=self.mel_filters,
            log_mel="log",
            mel_floor=1.192092955078125e-07,
            remove_dc_offset=True,
        )

        return features

    def forward(self, raw_speech: torch.Tensor):
        features = self._extract_fbank_features(raw_speech)

        # Normalize per mel bin
        mean = features.mean(dim=0, keepdim=True)
        var = features.var(dim=0, keepdim=True, unbiased=True)
        features = (features - mean) / torch.sqrt(var + 1e-7)

        return features


import torch
from src.utils import read_audio

def faster_impl(a):
    feature_extractor = FasterSeamlessM4TFeatureExtractor(
        feature_size=80,
        num_mel_bins=80,
        padding_value=1,
        return_attention_mask=True,
        sampling_rate=16000,
        stride=2
    )

    out = feature_extractor(a.to(device))

    return out

if __name__ == '__main__':
    audio = read_audio('data/test-clean/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac', 16_000) # type: ignore
    o = faster_impl(audio)
