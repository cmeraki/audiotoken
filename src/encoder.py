import torch
import tiktoken
import torchaudio
from typing import List
from functools import partial
from encodec import EncodecModel
from encodec.utils import convert_audio
from concurrent.futures import ThreadPoolExecutor

class TextEncoder():
    """
    Simple wrapper around the TikToken encoder to encode a list of strings
    """

    def __init__(self, tokenizer_name: str = "cl100k_base"):
        self.encoder = tiktoken.get_encoding(tokenizer_name)

    def __call__(self, x: List[str]) -> List[List[int]]:
        return self.encoder.encode_batch(x)

class VoiceEncoder():
    """
    Wrapper over encodec model to encode a list of audio files.
    Much faster than the original implementation as it uses multi-threading
    and batch processing
    """
    def __init__(self, bandwidth):
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(bandwidth)
        # Params for batch processing
        self.segment_length = self.model.sample_rate
        self.stride

    @staticmethod
    def preprocessing(x: str, model_sample_rate: int) -> torch.Tensor:
        """
        Given a audio file, this function reads the audio file and returns the audio tensor
        suitable for processing by the model
        """
        audio, sr = torchaudio.load(x)
        audio = convert_audio(audio, sr, model_sample_rate, 1)
        assert audio.channels == 1, f"Audio needs to be mono, provided {audio.channels} for {x}"
        assert audio.sample_rate == model_sample_rate, f"Audio needs to be {model_sample_rate}Hz, provided {audio.sample_rate}Hz for {x}"
        assert audio.dim() == 2, f"Audio needs to be 2D tensor, provided {audio.dim()}D for {x}"

        return audio.unsqueeze(0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, length = x.size(0), x.size(2)

        encoded_frames = []
        for offset in range(0, length, stride):
            frame = x[:, :, offset: offset + segment_length]
            encoded_frames.append(self._encode_frame(frame))
        return encoded_frames

    def __call__(self, x: List[str], batch_size: int, num_threads: int = 1) -> List[List[int]]:
        """
        Implements forward pass of the encodec model

        The input x is a list of audio files and the output is a list of list of tensors
        representing the encoded audio files
        """
        with ThreadPoolExecutor(num_threads) as executor:
            # Read the files in different threads and execute the forward pass
            futures = []
            for audio_file in x:
                futures.append(
                    executor.submit(partial(self.preprocessing, audio_file, self.model.sample_rate))
                )

        return None