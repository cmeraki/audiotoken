import pdb
import math
import torch
import torch.nn.functional as F

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)

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

        batch_size, num_samples = waveform.shape

        num_frames = int(1 + math.floor((num_samples - frame_length) / hop_length))
        num_frequency_bins = (fft_length // 2) + 1

        spectrogram = torch.empty((batch_size, num_frames, num_frequency_bins), dtype=torch.cfloat, device=self.device)
        buffer = torch.zeros(batch_size, fft_length, device=self.device)

        # print(waveform[0].mean())

        timestep = 0
        for frame_idx in range(num_frames):
            buffer[:, :frame_length] = waveform[:, timestep : timestep + frame_length]

            # print(frame_idx, 0, buffer[0, :frame_length].mean())
            if remove_dc_offset:
                buffer[:, :frame_length] = buffer[:, :frame_length] - torch.mean(buffer[:, :frame_length], dim=1, keepdim=True)

            # print(frame_idx, 1, buffer[0, :frame_length].mean())
            if preemphasis is not None:
                buffer[:, 1:frame_length] -= preemphasis * buffer[:, : frame_length - 1]
                buffer[:, 0] *= 1 - preemphasis

            # print(frame_idx, 2, buffer[0, :frame_length].mean())
            buffer[:, :frame_length] *= window

            # print(frame_idx, 3, buffer[0, :frame_length].mean())
            spectrogram[:, frame_idx] = torch.fft.rfft(buffer)

            # print(frame_idx, 4, buffer[0, :frame_length].mean())
            timestep += hop_length

        # print(spectrogram[0])

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

    def forward(self, raw_speech: torch.Tensor):

        assert len(raw_speech.shape) == 2, "Input tensor must have shape [batch, time]"

        features = self._extract_fbank_features(raw_speech)

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
    import time
    import torch
    from tqdm import tqdm
    from .utils import read_audio, find_audio_files

    device = 'cuda'
    audio_files = find_audio_files('./data/test-clean/LibriSpeech/test-clean/')

    # print(audio_files[0])
    batched_audio_files = []
    for a in audio_files:
        audio = read_audio(a, 16_000) # type: ignore
        audio = audio.squeeze(0)[:16_000]

        if audio.shape[0] < 16_000:
            audio = torch.cat([audio, torch.zeros(16_000 - audio.shape[0])])

        batched_audio_files.append(audio)

    tensor_audio_files = torch.stack(batched_audio_files).to(device)
    print(f'Batched audio files shape: {tensor_audio_files.shape}')

    batched_feature_extractor = W2VBert2Processor(
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
        tensor_audio_files
    )
    torch.cuda.synchronize()
    print(f'Batched feature extraction time: {time.time() - start_time:.2f}s')

    print(f'Batched input features shape: {batched_out["input_features"].shape}')

    from .faster_impl import W2VBert2Processor as SimpleW2VBert2Processor

    feature_extractor = SimpleW2VBert2Processor(
        feature_size=80,
        num_mel_bins=80,
        sampling_rate=16000,
        stride=2,
        padding_value=1,
        pad_to_multiple_of=1000,
        device=device,
    )

    # start_time = time.time()
    # for idx, audio in enumerate(batched_audio_files):
    #     single_out = feature_extractor(audio.to(device))

    # torch.cuda.synchronize()
    # print(f'Single feature extraction time: {time.time() - start_time:.2f}s')

    for idx, audio in tqdm(enumerate(batched_audio_files)):
        single_out = feature_extractor(audio.to(device))

        i1, a1 = batched_out['input_features'][idx], batched_out['attention_mask'][idx]
        i2, a2 = single_out['input_features'], single_out['attention_mask']

        if (i1-i2).abs().max().item() > 1e-5:
            print(f'Diff: {(i1-i2).abs().mean(), (i1-i2).abs().max()}')

        try:
            assert torch.allclose(i1, i2, rtol=0, atol=1e-5), f'Input features are not equal for audio file {idx}'
            assert torch.allclose(a1, a2, rtol=0, atol=0), f'Attention mask are not equal for audio file {idx}'
        except Exception as e:
            print(f'Error: {e}')
            # pdb.set_trace()
