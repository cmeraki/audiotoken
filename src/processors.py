import math
import torch
from typing import Optional, Dict
import torch.nn.functional as F

from src.utils import hertz_to_mel, mel_to_hertz, create_triangular_filter_bank

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
        pad_to_multiple_of: int = 2,
        device: str = 'cpu',
    ):
        super().__init__()

        self.device = device

        self.stride = stride
        self.padding_value = padding_value
        self.pad_to_multiple_of = pad_to_multiple_of

        mel_filters = mel_filter_bank(
            num_frequency_bins=256,
            num_mel_filters=num_mel_bins,
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
        mask: torch.Tensor,
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

        # Kaldi compliance: 16-bit signed integers
        waveform = waveform * (2**15)
        batch_size, num_samples = waveform.shape

        num_frames = int(1 + math.floor((num_samples - frame_length) / hop_length))
        num_frequency_bins = (fft_length // 2) + 1

        spectrogram = torch.empty((batch_size, num_frames, num_frequency_bins), dtype=torch.cfloat, device=self.device)
        buffer = torch.zeros(batch_size, fft_length, device=self.device)
        buffer_mask = torch.ones(batch_size, fft_length, device=self.device)

        print(f'Batch size: {batch_size}, num frames: {num_frames}, num frequency bins: {num_frequency_bins}')

        timestep = 0
        for frame_idx in range(num_frames):
            buffer[:, :frame_length] = waveform[:, timestep : timestep + frame_length]
            buffer_mask[:, :frame_length] = mask[:, timestep : timestep + frame_length]

            if remove_dc_offset:
                buffer[:, :frame_length] = buffer[:, :frame_length] - torch.mean(buffer[:, :frame_length], dim=1, keepdim=True)

            if preemphasis is not None:
                buffer[:, 1:frame_length] -= preemphasis * buffer[:, : frame_length - 1]
                buffer[:, 0] *= 1 - preemphasis

            buffer[:, :frame_length] *= window
            buffer[:, :frame_length] *= buffer_mask[:, :frame_length]

            spectrogram[:, frame_idx] = torch.fft.rfft(buffer)
            timestep += hop_length

        # Compute power spectrogram
        spectrogram = spectrogram.abs().pow(power)

        # Apply mel filterbank
        spectrogram = torch.matmul(spectrogram, mel_filters)
        # TODO: Handle mask appropriately here
        spectrogram = torch.maximum(spectrogram, torch.tensor(mel_floor, dtype=torch.float32))

        # Apply log
        spectrogram = torch.log(spectrogram)

        return spectrogram

    def _pad(self, arr: torch.Tensor):
        B, N, D = arr.shape

        P = 0
        if self.pad_to_multiple_of > 0:
            P = self.pad_to_multiple_of - (N % self.pad_to_multiple_of) if N % self.pad_to_multiple_of > 0 else 0

        # Create the padded array
        padded_array = F.pad(arr, (0, 0, 0, P), mode='constant', value=self.padding_value)

        # Create the attention mask
        attention_mask = torch.ones(B, N + P, dtype=torch.int32, device=arr.device)
        attention_mask[:, N:] = 0

        return padded_array, attention_mask

    def forward(self, raw_speech: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:

        assert len(raw_speech.shape) == 2, "Input tensor must have shape [batch, time]"

        features = self._create_spectrogram(
            raw_speech,
            mask,
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

        # Normalize per mel bin
        mean = features.mean(dim=1, keepdim=True)
        var = features.var(dim=1, keepdim=True, unbiased=True)
        features = (features - mean) / torch.sqrt(var + 1e-7)

        batch_size, num_frames, num_channels = features.shape

        remainder = num_frames % self.stride

        if remainder != 0:
            features = features[:, :num_frames-remainder, :]

        features = features.reshape(
            batch_size, (num_frames-remainder) // self.stride, num_channels * self.stride
        )

        input_features, attention_mask = self._pad(features)

        padded_inputs = {
            "input_features": input_features,
            "attention_mask": attention_mask
        }

        return padded_inputs


if __name__ == '__main__':
    """
    python -m src.processors --indir ./data/test-clean/ --device cuda:0
    """
    import pdb
    import time
    import torch
    import numpy as np
    from tqdm import tqdm
    from argparse import ArgumentParser
    from transformers import SeamlessM4TFeatureExtractor

    from src.utils import read_audio, find_audio_files
    from src.configs import Wav2VecBertConfig

    parser = ArgumentParser()

    parser.add_argument('--indir', type=str, default='./data/test-clean/')
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    device = args.device
    audio_files = find_audio_files(args.indir)

    batched_audio_files = []
    np_audio_files = []

    # Clipping to 1 seconds
    for a in audio_files:
        audio = read_audio(a, 16_000) # type: ignore
        audio = audio.squeeze(0)[:16_000]

        if audio.shape[0] < 16_000:
            audio = torch.cat([audio, torch.zeros(16_000 - audio.shape[0])])

        batched_audio_files.append(audio)
        np_audio_files.append(audio.numpy())

    np_audio_files = np.array(np_audio_files) # type: ignore
    tensor_audio_files = torch.stack(batched_audio_files).to(device)
    print(f'Batched audio files shape: {tensor_audio_files.shape}')

    # Testing custom implementation
    print('Testing custom implementation')
    batched_feature_extractor = Wav2VecBertProcessor(
        feature_size=80,
        num_mel_bins=80,
        sampling_rate=16000,
        stride=2,
        padding_value=1,
        pad_to_multiple_of=500,
        device=device,
    )

    start_time = time.time()
    batched_out = batched_feature_extractor(
        tensor_audio_files, torch.ones_like(tensor_audio_files, device=device)
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
