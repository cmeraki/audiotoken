import math
import torch
from typing import Dict
import torch.nn.functional as F

from .utils import hertz_to_mel, mel_to_hertz, create_triangular_filter_bank

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


class Wav2VecBertProcessor(torch.nn.Module):
    """
    Processor for Wav2Vec-BERT 2.0 model
    This class is inspired by: https://github.com/huggingface/transformers/blob/main/src/transformers/models/seamless_m4t/feature_extraction_seamless_m4t.py

    With the following changes:

        1. Supports batched inputs
        2. Supports torch.Tensor inputs
        3. Supports computation on GPU
        4. Pad to multiple of is applied at the end after striding
            4.1 This will give same results but the `pad_to_multiple_of` argument makes more sense now

    This implementation is optimized only for Wav2Vec-BERT 2.0 model
    and is at least 10x faster than the HF implementation and can run on GPUs :)
    """
    def __init__(
        self,
        feature_size=80,
        sampling_rate=16000,
        num_mel_bins=80,
        stride=2,
        padding_value: float = 1.0,
    ):
        super().__init__()

        self.stride = stride
        self.padding_value = padding_value

        self.num_frequency_bins = 256
        self.sampling_rate = sampling_rate
        self.num_mel_bins = num_mel_bins

        # Move device-specific operations to forward method
        self.register_buffer('window', None)
        self.register_buffer('mel_filters', None)

    def _initialize_buffers(self, device):
        if self.window is None or self.mel_filters is None:
            mel_filters = mel_filter_bank(
                num_frequency_bins=self.num_frequency_bins,
                num_mel_filters=self.num_mel_bins,
                min_frequency=20,
                max_frequency=self.sampling_rate // 2,
                sampling_rate=self.sampling_rate,
            ).to(device)
            window = torch.pow(torch.hann_window(400, periodic=False, device=device), 0.85)

            self.register_buffer('mel_filters', torch.nn.Parameter(F.pad(mel_filters, (0, 0, 0, 1))))
            self.register_buffer('window', torch.nn.Parameter(window))

    def _create_spectrogram_mask(
        self,
        mask: torch.Tensor,
        num_frames: int,
        num_frequency_bins: int,
        **kwargs
    ) -> torch.Tensor:
        """
        Create a mask that matches the time resolution of the spectrogram

        Args:
            mask (torch.Tensor)
            num_frames (int)
            num_frequency_bins (int)
        Returns:
            torch.Tensor (shape: [batch_size, num_frames, num_frequency_bins])
        """
        frame_length = kwargs.get('frame_length', 400)
        hop_length = kwargs.get('hop_length', 160)

        # Reshape mask to match spectrogram time resolution
        # The shape of the mask here will be: (batch_size, num_frames)
        mask_downsampled = F.avg_pool1d(
            mask.unsqueeze(1),
            kernel_size=frame_length,
            stride=hop_length,
            padding=0
        ).squeeze(1)[:, :num_frames]
        mask_downsampled = torch.where(mask_downsampled == 1, mask_downsampled, 0)

        # Expand the mask to cover all frequency bins
        # The expanded mask shape will be: (batch_size, num_frames, num_frequency_bins)
        # The mask will be expanded with the same value to all the frequency_bins
        mask_expanded = mask_downsampled.unsqueeze(-1).expand(-1, -1, num_frequency_bins)

        return mask_expanded

    def _compute_masked_mean_var(self, x: torch.Tensor, mask: torch.Tensor) -> tuple:
        """
        Compute the mean and variance of the input tensor x using the mask

        Args:
            x (torch.Tensor)
            mask (torch.Tensor)

        Returns:
            torch.Tensor
        """
        masked_x: torch.Tensor = x * mask
        sum_x = torch.sum(masked_x, dim=1, keepdims=True) # type: ignore
        count_valid = torch.sum(mask, dim=1, keepdims=True) # type: ignore

        mean_x = sum_x / count_valid.clamp(min=1)
        var_x = ((masked_x - mean_x) ** 2 * mask).sum(dim=1, keepdim=True) / count_valid.clamp(min=1)

        return mean_x, var_x

    def _create_spectrogram(
        self,
        waveform: torch.Tensor,
        window: torch.Tensor,
        mel_filters: torch.Tensor,
        device: torch.device,
        **kwargs
    ) -> torch.Tensor:

        frame_length = kwargs.get('frame_length', 400)
        hop_length = kwargs.get('hop_length', 160)
        fft_length = kwargs.get('fft_length', 512)
        power = kwargs.get('power', 2)
        preemphasis = kwargs.get('preemphasis')
        mel_floor = kwargs.get('mel_floor')
        remove_dc_offset = kwargs.get('remove_dc_offset')

        # Kaldi compliance: 16-bit signed integers
        waveform = waveform * (2**15)
        batch_size, num_samples = waveform.shape

        num_frames = int(1 + math.floor((num_samples - frame_length) / hop_length))
        num_frequency_bins = (fft_length // 2) + 1

        spectrogram = torch.empty((batch_size, num_frames, num_frequency_bins), dtype=torch.cfloat, device=device)
        buffer = torch.zeros(batch_size, fft_length, device=device)

        timestep = 0
        for frame_idx in range(num_frames):
            buffer[:, :frame_length] = waveform[:, timestep : timestep + frame_length]

            if remove_dc_offset:
                buffer[:, :frame_length] = buffer[:, :frame_length] - torch.mean(buffer[:, :frame_length], dim=1, keepdim=True)

            if preemphasis is not None:
                buffer[:, 1:frame_length] -= preemphasis * buffer[:, : frame_length - 1]
                buffer[:, 0] *= 1 - preemphasis

            buffer[:, :frame_length] *= window

            spectrogram[:, frame_idx] = torch.fft.rfft(buffer)
            timestep += hop_length

        # Compute power spectrogram
        spectrogram = spectrogram.abs().pow(power)

        # Apply mel filterbank
        spectrogram = torch.matmul(spectrogram, mel_filters)
        spectrogram = torch.maximum(spectrogram, torch.tensor(mel_floor, dtype=torch.float32))

        # Apply log
        spectrogram = torch.log(spectrogram)

        return spectrogram

    def _pad(self, arr: torch.Tensor, mask: torch.Tensor, pad_to_multiple_of: int) -> tuple:
        B, N, D = arr.shape

        P = 0
        if pad_to_multiple_of > 0:
            P = pad_to_multiple_of - (N % pad_to_multiple_of) if N % pad_to_multiple_of > 0 else 0

        # Create the padded array after applying the original mask provided
        padded_array = torch.where(mask == 0, self.padding_value, arr)
        padded_array = F.pad(padded_array, (0, 0, 0, P), mode='constant', value=self.padding_value)

        # Create the padded attention mask using the original mask provided
        attention_mask = F.pad(mask[:, :, 0], (0, P), mode='constant', value=0)
        attention_mask = torch.where(attention_mask == 1, attention_mask, 0)

        return padded_array, attention_mask

    def forward(self, raw_speech: torch.Tensor, mask: torch.Tensor, pad_to_multiple_of: int = 2) -> Dict[str, torch.Tensor]:

        device = raw_speech.device
        self._initialize_buffers(device)

        assert len(raw_speech.shape) == 2, "Input tensor must have shape [batch, time]"

        spectrogram_config = {
            "frame_length": 400,
            "hop_length": 160,
            "fft_length": 512,
            "power": 2.0,
            "preemphasis": 0.97,
            "mel_floor": 1.192092955078125e-07,
            "remove_dc_offset": True,
        }

        features = self._create_spectrogram(
            raw_speech,
            window=self.window,
            mel_filters=self.mel_filters,
            device=device,
            **spectrogram_config
        )

        spectrogram_mask = self._create_spectrogram_mask(
            mask=mask,
            num_frames=features.shape[1],
            num_frequency_bins=features.shape[2],
            **spectrogram_config
        )

        mean, var = self._compute_masked_mean_var(features, spectrogram_mask)
        features = (features - mean) / torch.sqrt(var + 1e-7)

        batch_size, num_frames, num_channels = features.shape

        remainder = num_frames % self.stride

        if remainder != 0:
            features = features[:, :num_frames-remainder, :]
            spectrogram_mask = spectrogram_mask[:, :num_frames-remainder, :]

        features = features.reshape(
            batch_size, (num_frames-remainder) // self.stride, num_channels * self.stride
        )
        spectrogram_mask = spectrogram_mask.reshape(
            batch_size, (num_frames-remainder) // self.stride, num_channels * self.stride
        )

        input_features, attention_mask = self._pad(features, spectrogram_mask, pad_to_multiple_of)

        padded_inputs = {
            "input_features": input_features,
            "attention_mask": attention_mask
        }

        return padded_inputs


if __name__ == '__main__':
    """
    python -m src.processors --indir ./data/test-clean/ --device cuda:0
    """
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    torch.set_float32_matmul_precision("high") # set matmul precision to use either bfloat16 or tf32

    import pdb
    import time
    import torch
    import numpy as np
    from tqdm import tqdm
    from argparse import ArgumentParser
    from transformers import SeamlessM4TFeatureExtractor

    from .utils import read_audio, find_audio_files
    from .configs import Wav2VecBertConfig

    parser = ArgumentParser()

    parser.add_argument('--indir', type=str, default='./data/test-clean/')
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    device = args.device
    audio_files = find_audio_files(args.indir)

    batched_audio_files = []
    batch_attention_masks = []
    np_audio_files = []

    # Clipping to 10 seconds
    for a in audio_files:
        audio = read_audio(a, 16_000) # type: ignore
        audio = audio.squeeze(0)[:160_000]
        attention_mask = torch.ones_like(audio)

        np_audio_files.append(audio.numpy())

        if audio.shape[0] < 160_000:
            audio = torch.cat([audio, torch.zeros(160_000 - audio.shape[0])])
            attention_mask = torch.cat([attention_mask, torch.zeros(160_000 - attention_mask.shape[0])])

        batched_audio_files.append(audio)
        batch_attention_masks.append(attention_mask)

    tensor_audio_files = torch.stack(batched_audio_files).to(device)
    tensor_attention_masks = torch.stack(batch_attention_masks).to(device)

    print(f'Batched audio files shape: {tensor_audio_files.shape}')

    # Testing custom implementation
    print('Testing custom implementation')
    batched_feature_extractor = Wav2VecBertProcessor(
        feature_size=80,
        num_mel_bins=80,
        sampling_rate=16000,
        stride=2,
        padding_value=1
    )

    start_time = time.time()
    batched_out = batched_feature_extractor(
        tensor_audio_files, tensor_attention_masks, pad_to_multiple_of=500
    )
    torch.cuda.synchronize()

    print(f'Batched feature extraction time: {time.time() - start_time:.2f}s')
    print(f'Batched input features shape: {batched_out["input_features"].shape}')

    # Testing HF implementation
    print('Testing HF implementation')
    processor = SeamlessM4TFeatureExtractor.from_pretrained(
        Wav2VecBertConfig.model_id)

    start_time = time.time()
    hf_out = processor(
        np_audio_files,
        sampling_rate=Wav2VecBertConfig.model_sample_rate,
        return_attention_masks=True,
        padding=True,
        truncation=False,
        pad_to_multiple_of=1000,
        return_tensors='pt'
    )

    print(f'HF feature extraction time: {time.time() - start_time:.2f}s')
    print(f'HF features shape: {hf_out["input_features"].shape}')

    diffs = []

    for idx in tqdm(range(len(batched_audio_files))):

        i1, a1 = batched_out['input_features'][idx].cpu(), batched_out['attention_mask'][idx].cpu()
        i2, a2 = hf_out['input_features'][idx], hf_out['attention_mask'][idx]

        try:
            assert torch.allclose(i1, i2, rtol=0, atol=1e-5), f'Input features are not equal for audio file {idx}'
            assert torch.allclose(a1, a2, rtol=0, atol=0), f'Attention mask are not equal for audio file {idx}'

        except Exception as e:
            # print(f'Error: {e}')
            # pdb.set_trace()
            diffs.append(
                (i1-i2).abs().max()
            )

    print(f'Diffs: {len(diffs)}, mean: {np.mean(diffs)}, max: {np.max(diffs)}')
