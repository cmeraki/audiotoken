import os
import sys
import torch
import tiktoken
import torchaudio
from loguru import logger
from typing import List
from queue import Queue
from encodec import EncodecModel
from encodec.utils import convert_audio

logger.remove()
logger.add(sys.stdout, format="[{time: YYYY-MM-DD HH:mm:ss} {level}] {message}", level="DEBUG")


def process_audio(x: os.PathLike, model_sample_rate: int) -> torch.Tensor:
    """
        Given an audio file, this function reads the audio file and returns the audio tensor
        suitable for processing by the model
        """
    audio, sr = torchaudio.load(x)
    audio = convert_audio(audio, sr, model_sample_rate, 1)
    assert audio.shape[0] == 1, f"Audio needs to be mono, provided {audio.shape[0]} channels for {x}"
    assert sr == model_sample_rate, f"Audio needs to be {model_sample_rate}Hz, provided {sr}Hz for {x}"
    assert audio.dim() == 2, f"Audio needs to be 2D tensor, provided {audio.dim()}D for {x}"

    logger.debug(f"Processed audio file {x}, shape {audio.shape}")

    return audio


class TextEncoder:
    """
    Simple wrapper around the TikToken encoder to encode a list of strings
    """

    def __init__(self, tokenizer_name: str = "cl100k_base"):
        self.encoder = tiktoken.get_encoding(tokenizer_name)

    def __call__(self, x: List[str]) -> List[List[int]]:
        return self.encoder.encode_batch(x)

class VoiceEncoder:
    """
    Wrapper over Encodec model to encode a list of audio files.
    Much faster than the original implementation as it uses multi-threading
    and batch processing
    """

    def __init__(self, bandwidth: int, single_segment_duration: int):
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(bandwidth)
        self.device = torch.device('cpu')
        self.pad_token = 0

        # Params for batch processing
        self.overlap = 0.1
        self.segment_length = self.model.sample_rate * single_segment_duration
        self.stride = int(self.segment_length - self.overlap * self.model.sample_rate)  # 10 ms overlap

    def prepare_batch(self, audio: torch.Tensor, batch_size: int):
        logger.info('Preparing batch process started')

        local_batch = torch.zeros(batch_size, 1, self.segment_length, device=self.device)
        local_batch_idx = 0

        _, length = audio.shape

        for i in range(0, length, self.stride):
            logger.debug(f'Processing segment {i} to {i+self.segment_length} and batch index {local_batch_idx}')

            # If we reach the end of the audio, pad the audio and yield the batch
            if i + self.segment_length > length:
                local_batch[local_batch_idx, 0, :length-i] = audio[:, i:]
                local_batch[local_batch_idx, 0, length-i:] = self.pad_token
                yield local_batch[:local_batch_idx].clone()
                local_batch_idx = 0
                break

            local_batch[local_batch_idx, 0, :] = audio[:, i:i+self.segment_length]
            local_batch_idx += 1

            # If we reach the max batch size, yield the batch
            if local_batch_idx == batch_size:
                yield local_batch.clone()
                local_batch.zero_()
                local_batch_idx = 0

        # If we have some remaining audio, yield the batch
        if local_batch_idx > 0:
            yield local_batch[:local_batch_idx].clone()

    def __call__(self, read_q: Queue[torch.Tensor], batch_size: int):
        """
        Implements forward pass of the Encodec model

        The input x is a list of audio files and the output is a list of list of tensors
        representing the encoded audio files
        """

        # Have a global batch size
        while not read_q.empty():
            local_sample = read_q.get()
            logger.debug(f'Found an audio file to encode')

            for local_batch in self.prepare_batch(local_sample, batch_size):
                with torch.no_grad():
                    emb = self.model.encoder(local_batch)
                    codes = self.model.quantizer.encode(
                        emb, self.model.frame_rate, self.model.bandwidth
                    )
                    codes = codes.transpose(0, 1)  # [B, K, T]
                    yield codes

    def combine_batches(self, batches: List[torch.Tensor]):
        """
        Combines the batches to a single tensor
        """

        for b in batches:
            B, K, T = b.shape
            out = b.transpose(0, 1)
            out = out[:, :, :self.stride].reshape(K, -1)

            yield out

if __name__ == '__main__':
    import pdb
    from time import time
    from pathlib import Path

    model_sr = 24_000
    audio_file_paths = ['~/Desktop/meraki/encodec/test_24k.wav'] * 100
    audio_files: Queue[torch.Tensor] = Queue()

    for p in audio_file_paths:
        audio_files.put(process_audio(Path(p).expanduser(), model_sr))

    voice_encoder = VoiceEncoder(bandwidth=3, single_segment_duration=20)
    start_time = time()
    encoded_audio = voice_encoder(
        read_q=audio_files,
        batch_size=100
    )

    result = []
    for idx, batch in enumerate(encoded_audio):
        print(idx, batch.shape)
        result.append(batch)

    logger.info(f'Encoding took {time() - start_time:.2f}s')

    temp = []
    for a in voice_encoder.combine_batches(result):
        temp.append(a)

    # pdb.set_trace()

    print('Done')