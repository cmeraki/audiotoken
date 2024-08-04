import numpy as np
from typing import List, Optional, Union
import pdb

def window_function(
    window_length: int,
    name: str = "hann",
    periodic: bool = True,
    frame_length: Optional[int] = None,
    center: bool = True,
) -> np.ndarray:
    """
    Returns an array containing the specified window. This window is intended to be used with `stft`.
    """
    length = window_length + 1 if periodic else window_length

    if name == "boxcar":
        window = np.ones(length)
    elif name in ["hamming", "hamming_window"]:
        window = np.hamming(length)
    elif name in ["hann", "hann_window"]:
        window = np.hanning(length)
    elif name in ["povey"]:
        window = np.power(np.hanning(length), 0.85)
    else:
        raise ValueError(f"Unknown window function '{name}'")

    if periodic:
        window = window[:-1]

    if frame_length is None:
        return window

    if window_length > frame_length:
        raise ValueError(
            f"Length of the window ({window_length}) may not be larger than frame_length ({frame_length})"
        )

    padded_window = np.zeros(frame_length)
    offset = (frame_length - window_length) // 2 if center else 0
    padded_window[offset : offset + window_length] = window
    return padded_window

def spectrogram(
    waveform: np.ndarray,
    window: np.ndarray,
    frame_length: int,
    hop_length: int,
    fft_length: Optional[int] = None,
    power: Optional[float] = 1.0,
    center: bool = True,
    pad_mode: str = "reflect",
    onesided: bool = True,
    preemphasis: Optional[float] = None,
    mel_filters: Optional[np.ndarray] = None,
    mel_floor: float = 1e-10,
    log_mel: Optional[str] = None,
    reference: float = 1.0,
    min_value: float = 1e-10,
    db_range: Optional[float] = None,
    remove_dc_offset: Optional[bool] = None,
    dtype = np.float32,
) -> np.ndarray:
    window_length = len(window)

    if fft_length is None:
        fft_length = frame_length

    if frame_length > fft_length:
        raise ValueError(f"frame_length ({frame_length}) may not be larger than fft_length ({fft_length})")

    if window_length != frame_length:
        raise ValueError(f"Length of the window ({window_length}) must equal frame_length ({frame_length})")

    if hop_length <= 0:
        raise ValueError("hop_length must be greater than zero")

    if waveform.ndim != 1:
        raise ValueError(f"Input waveform must have only one dimension, shape is {waveform.shape}")

    if np.iscomplexobj(waveform):
        raise ValueError("Complex-valued input waveforms are not currently supported")

    if power is None and mel_filters is not None:
        raise ValueError(
            "You have provided `mel_filters` but `power` is `None`. Mel spectrogram computation is not yet supported for complex-valued spectrogram."
            "Specify `power` to fix this issue."
        )

    # promote to float64, since np.fft uses float64 internally
    waveform = waveform.astype(np.float64)
    window = window.astype(np.float64)
    # split waveform into frames of frame_length size
    num_frames = int(1 + np.floor((waveform.size - frame_length) / hop_length))

    num_frequency_bins = (fft_length // 2) + 1 if onesided else fft_length
    spectrogram = np.empty((num_frames, num_frequency_bins), dtype=np.complex64)

    # rfft is faster than fft
    fft_func = np.fft.rfft if onesided else np.fft.fft
    buffer = np.zeros(fft_length)

    timestep = 0
    for frame_idx in range(num_frames):
        buffer[:frame_length] = waveform[timestep : timestep + frame_length]

        if remove_dc_offset:
            buffer[:frame_length] = buffer[:frame_length] - buffer[:frame_length].mean()

        if preemphasis is not None:
            buffer[1:frame_length] -= preemphasis * buffer[: frame_length - 1]
            buffer[0] *= 1 - preemphasis

        buffer[:frame_length] *= window

        spectrogram[frame_idx] = fft_func(buffer)
        timestep += hop_length

    # note: ** is much faster than np.power
    if power is not None:
        spectrogram = np.abs(spectrogram, dtype=np.float64) ** power #type: ignore

    spectrogram = spectrogram.T

    if mel_filters is not None:
        spectrogram = np.maximum(mel_floor, np.dot(mel_filters.T, spectrogram))

        spectrogram = np.log(spectrogram)
        spectrogram = np.asarray(spectrogram, dtype)

    return spectrogram


def hertz_to_mel(freq):
    """
    Convert frequency from hertz to mels.
    mel_scale: kaldi
    """

    return 1127.0 * np.log(1.0 + (freq / 700.0))


def mel_to_hertz(mels):
    """
    Convert frequency from mels to hertz.
    mel_scale: kaldi
    """

    return 700.0 * (np.exp(mels / 1127.0) - 1.0)


def _create_triangular_filter_bank(fft_freqs: np.ndarray, filter_freqs: np.ndarray) -> np.ndarray:
    """
    Creates a triangular filter bank.

    Adapted from *torchaudio* and *librosa*.
    """
    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    return np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))


def mel_filter_bank(
    num_frequency_bins: int,
    num_mel_filters: int,
    min_frequency: float,
    max_frequency: float,
    sampling_rate: int
) -> np.ndarray:
    # center points of the triangular mel filters
    mel_min = hertz_to_mel(min_frequency)
    mel_max = hertz_to_mel(max_frequency)
    mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
    filter_freqs = mel_to_hertz(mel_freqs)

    fft_bin_width = sampling_rate / (num_frequency_bins * 2)
    fft_freqs = hertz_to_mel(fft_bin_width * np.arange(num_frequency_bins))
    filter_freqs = mel_freqs

    mel_filters = _create_triangular_filter_bank(fft_freqs, filter_freqs)

    return mel_filters


class OptimizedSeamlessM4TFeatureExtractor():
    def __init__(
        self,
        feature_size=80,
        sampling_rate=16000,
        num_mel_bins=80,
        padding_value=1.0,
        pad_to_multiple_of=2,
        stride=2,
        **kwargs,
    ):
        """
        SeamlessM4TFeatureExtractor {
            "feature_extractor_type": "SeamlessM4TFeatureExtractor",
            "feature_size": 80,
            "num_mel_bins": 80,
            "padding_side": "right",
            "padding_value": 1,
            "processor_class": "Wav2Vec2BertProcessor",
            "return_attention_mask": true,
            "sampling_rate": 16000,
            "stride": 2
        }
        """
        self.num_mel_bins = num_mel_bins
        self.return_attention_mask = True
        self.stride = stride
        self.padding_value = padding_value
        self.pad_to_multiple_of = pad_to_multiple_of

        mel_filters = mel_filter_bank(
            num_frequency_bins=256,
            num_mel_filters=self.num_mel_bins,
            min_frequency=20,
            max_frequency=sampling_rate // 2,
            sampling_rate=sampling_rate,
        )

        self.mel_filters = np.pad(mel_filters, ((0, 1), (0, 0)))
        self.window = window_function(400, "povey", periodic=False)

    def _extract_fbank_features(
        self,
        waveform: np.ndarray,
    ) -> np.ndarray:
        """
        Get mel-filter bank features using TorchAudio. Note that TorchAudio requires 16-bit signed integers as inputs
        and hence the waveform should not be normalized before feature extraction.
        """
        waveform = np.squeeze(waveform) * (2**15)  # Kaldi compliance: 16-bit signed integers
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
        features = features.T
        return features

    def pad(self, arr: np.ndarray):
        N, D = arr.shape

        P = 0
        if self.pad_to_multiple_of > 0:
            P = self.pad_to_multiple_of - (N % self.pad_to_multiple_of) if N % self.pad_to_multiple_of > 0 else 0

        # Create the padded array
        padded_array = np.pad(arr, ((0, P), (0, 0)), mode='constant', constant_values=self.padding_value)

        # Create the attention mask
        attention_mask = np.ones(N + P, dtype=np.int32)
        attention_mask[N:] = 0

        return padded_array, attention_mask

    def __call__(
        self,
        raw_speech: Union[np.ndarray],
    ):
        padded_inputs = {  # type: ignore
            "input_features": [],
            "attention_mask": []
        }

        if len(raw_speech.shape) > 1:
            speech = raw_speech
        else:
            speech = np.expand_dims(raw_speech, 0)

        for w in speech:
            features = self._extract_fbank_features(w)

            # Normalize per mel bin
            features = (features - np.expand_dims(features.mean(0), 0)) / np.sqrt(np.expand_dims(features.var(0, ddof=1), 0) + 1e-7)

            padded_features = self.pad(features)
            input_features = padded_features[0]
            attention_mask = padded_features[1]

            num_frames, num_channels = input_features.shape

            remainder = num_frames % self.stride

            if remainder != 0:
                input_features = input_features[:num_frames, :]
                attention_mask = attention_mask[:num_frames]

            input_features = np.reshape(
                input_features, (num_frames // self.stride, num_channels * self.stride)
            )

            indices = np.arange(0, num_frames)
            attention_mask = attention_mask[indices % self.stride == 1]

            padded_inputs["input_features"].append(input_features)
            padded_inputs["attention_mask"].append(attention_mask)

        padded_inputs['input_features'] = np.stack(padded_inputs['input_features'], axis=0) # type: ignore
        padded_inputs['attention_mask'] = np.stack(padded_inputs['attention_mask'], axis=0) # type: ignore

        return padded_inputs


import torch
from .utils import read_audio

def optim_impl(a):
    a = a.numpy()
    feature_extractor = OptimizedSeamlessM4TFeatureExtractor(
        feature_size=80,
        num_mel_bins=80,
        padding_value=1,
        return_attention_mask=True,
        sampling_rate=16000,
        stride=2
    )
    out = feature_extractor(a)

    return out

if __name__ == '__main__':
    audio = read_audio('./data/test-clean/LibriSpeech/test-clean/7021/85628/7021-85628-0005.flac', 16_000) # type: ignore
    o = optim_impl(audio)
    print(f'Shape: {o["input_features"].shape}, {o["attention_mask"].shape}')
