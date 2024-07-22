import torch
import torch.nn.functional as F
from typing import List, Optional, Union

def spectrogram(
    waveform: torch.Tensor,
    window: torch.Tensor,
    frame_length: int,
    hop_length: int,
    fft_length: Optional[int] = None,
    power: Optional[float] = 1.0,
    center: bool = True,
    pad_mode: str = "reflect",
    onesided: bool = True,
    preemphasis: Optional[float] = None,
    mel_filters: Optional[torch.Tensor] = None,
    mel_floor: float = 1e-10,
    log_mel: Optional[str] = None,
    remove_dc_offset: Optional[bool] = None,
) -> torch.Tensor:
    if fft_length is None:
        fft_length = frame_length

    # Pad the input waveform
    if center:
        padding = (frame_length - hop_length) // 2
        waveform = F.pad(waveform, (padding, padding), mode=pad_mode)

    # Preemphasis
    if preemphasis is not None:
        waveform = torch.cat([waveform[:, :1], waveform[:, 1:] - preemphasis * waveform[:, :-1]], dim=1)

    # Remove DC offset
    if remove_dc_offset:
        waveform = waveform - waveform.mean(dim=1, keepdim=True)

    # Compute STFT
    stft = torch.stft(
        waveform,
        n_fft=fft_length,
        hop_length=hop_length,
        win_length=frame_length,
        window=window,
        center=False,
        return_complex=True,
    )

    # Compute power spectrogram
    if power is not None:
        spectrogram = stft.abs().pow(power)
    else:
        spectrogram = stft

    print(waveform, waveform.shape)
    print(mel_filters)

    # Apply mel filterbank
    if mel_filters is not None:
        spectrogram = torch.matmul(mel_filters.T, spectrogram)
        spectrogram = torch.max(spectrogram, torch.tensor(mel_floor, device=spectrogram.device))

        if log_mel:
            spectrogram = torch.log(spectrogram)

    return spectrogram

class FasterSeamlessM4TFeatureExtractor(torch.nn.Module):
    def __init__(
        self,
        feature_size=80,
        sampling_rate=16000,
        num_mel_bins=80,
        padding_value=0.0,
        stride=2,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
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
        print(waveform)
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
        return features.transpose(-1, -2)

    def forward(self, raw_speech: torch.Tensor):
        features = self._extract_fbank_features(raw_speech)

        # Normalize per mel bin
        mean = features.mean(dim=1, keepdim=True)
        var = features.var(dim=1, keepdim=True, unbiased=True)
        features = (features - mean) / torch.sqrt(var + 1e-7)

        return features

# Helper functions (unmodified)
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
