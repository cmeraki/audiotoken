import pdb
import math
import torch
from typing import Optional
import torch.nn.functional as F

# Helper functions
def hertz_to_mel(freq):
    return 1127.0 * torch.log(1.0 + (freq / 700.0))

def mel_to_hertz(mels):
    return 700.0 * (torch.exp(mels / 1127.0) - 1.0)

def create_triangular_filter_bank(fft_freqs: torch.Tensor, filter_freqs: torch.Tensor) -> torch.Tensor:
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

    return create_triangular_filter_bank(fft_freqs, filter_freqs)


class W2VBert2Processor(torch.nn.Module):
    def __init__(
        self,
        feature_size=80,
        sampling_rate=16000,
        num_mel_bins=80,
        stride=2,
        padding_value: float = 1.0,
        pad_to_multiple_of: int = 2,
        device: str = 'cpu',
    ):
        super().__init__()
        self.num_mel_bins = num_mel_bins
        self.stride = stride
        self.padding_value = padding_value
        self.device = device
        self.pad_to_multiple_of = pad_to_multiple_of

        mel_filters = mel_filter_bank(
            num_frequency_bins=256,
            num_mel_filters=self.num_mel_bins,
            min_frequency=20,
            max_frequency=sampling_rate // 2,
            sampling_rate=sampling_rate,
        ).to(device)
        window = torch.pow(torch.hann_window(400, periodic=False, device=device), 0.85)

        self.register_buffer('mel_filters', F.pad(mel_filters, (0, 0, 0, 1)))
        self.register_buffer('window', window)

    def _create_spectrogram(
        self,
        waveform: torch.Tensor,
        window: torch.Tensor,
        frame_length: int,
        hop_length: int,
        fft_length: int,
        power: float,
        preemphasis: float,
        mel_filters: torch.Tensor,
        mel_floor: float = 1.192092955078125e-07,
        remove_dc_offset: bool = True,
    ) -> torch.Tensor:

        waveform = waveform.squeeze(0)

        num_frames = int(1 + math.floor((waveform.shape[-1] - frame_length) / hop_length))
        num_frequency_bins = (fft_length // 2) + 1

        spectrogram = torch.empty((num_frames, num_frequency_bins), dtype=torch.cfloat, device=self.device)
        buffer = torch.zeros(fft_length, device=self.device)

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

        # Apply mel filterbank
        spectrogram = torch.matmul(spectrogram, mel_filters)
        spectrogram = torch.maximum(spectrogram, torch.tensor(mel_floor, dtype=torch.float32))

        # Apply log
        spectrogram = torch.log(spectrogram)

        return spectrogram

    def _extract_fbank_features(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform * (2**15)  # Kaldi compliance: 16-bit signed integers

        features = self._create_spectrogram(
            waveform,
            self.window,
            frame_length=400,
            hop_length=160,
            fft_length=512,
            power=2.0,
            preemphasis=0.97,
            mel_filters=self.mel_filters,
            mel_floor=1.192092955078125e-07,
            remove_dc_offset=True,
        )

        return features

    def _pad(self, arr: torch.Tensor):
        N, D = arr.shape

        P = 0
        if self.pad_to_multiple_of > 0:
            P = self.pad_to_multiple_of - (N % self.pad_to_multiple_of) if N % self.pad_to_multiple_of > 0 else 0

        # Create the padded array
        padded_array = F.pad(arr, (0, 0, 0, P), mode='constant', value=self.padding_value)

        # Create the attention mask
        attention_mask = torch.ones(N + P, dtype=torch.int32, device=arr.device)
        attention_mask[N:] = 0

        return padded_array, attention_mask

    def forward(self, raw_speech: torch.Tensor):

        if len(raw_speech.shape) > 1:
            speech = raw_speech
        else:
            speech = raw_speech.unsqueeze(0)

        padded_inputs = { # type: ignore
            "input_features": [],
            "attention_mask": []
        }

        for s in speech:
            features = self._extract_fbank_features(s)

            # Normalize per mel bin
            mean = features.mean(dim=0, keepdim=True)
            var = features.var(dim=0, keepdim=True, unbiased=True)
            features = (features - mean) / torch.sqrt(var + 1e-7)

            padded_features = self._pad(features)
            input_features = padded_features[0]
            attention_mask = padded_features[1]

            num_frames, num_channels = input_features.shape

            remainder = num_frames % self.stride

            if remainder != 0:
                input_features = input_features[:num_frames, :]
                attention_mask = attention_mask[:num_frames]

            input_features = input_features.reshape(
                num_frames // self.stride, num_channels * self.stride
            )

            attention_mask = attention_mask[torch.arange(0, num_frames, device=attention_mask.device) % self.stride == 1]

            padded_inputs['input_features'].append(input_features)
            padded_inputs['attention_mask'].append(attention_mask)

        padded_inputs['input_features'] = torch.stack(padded_inputs['input_features'], dim=0) # type: ignore
        padded_inputs['attention_mask'] = torch.stack(padded_inputs['attention_mask'], dim=0) # type: ignore

        return padded_inputs

if __name__ == '__main__':
    import torch
    from .utils import read_audio

    device = 'cuda'
    audio = read_audio('data/test-clean/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac', 16_000) # type: ignore

    feature_extractor = W2VBert2Processor(
        feature_size=80,
        num_mel_bins=80,
        sampling_rate=16000,
        stride=2,
        padding_value=1,
        device=device,
    )

    o = feature_extractor(
        audio.to(device)
    )

    print(f'Shape: {o["input_features"].shape}, {o["attention_mask"].shape}')
