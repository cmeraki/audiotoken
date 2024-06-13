import sys
import torch
import tiktoken
from loguru import logger
from typing import List
from queue import Queue
from encodec import EncodecModel

from .utils import process_audio

logger.remove()
logger.add(sys.stdout, format="[{time: YYYY-MM-DD HH:mm:ss} {level}] {message}", level="ERROR")

class TextEncoder:
    """
    Simple wrapper around the TikToken encoder to encode a list of strings
    """

    def __init__(self, tokenizer_name: str = "cl100k_base", num_threads: int = 12):
        self.encoder = tiktoken.get_encoding(tokenizer_name)
        self.num_threads = num_threads

    def __call__(self, x: List[str]) -> List[List[int]]:
        return self.encoder.encode_batch(
            x,
            num_threads=self.num_threads
        )

class VoiceEncoder:
    """
    Wrapper over Encodec model to encode a list of audio files.

    The input is a list of tensors which are batched in two ways:
    1. First the audio is batched to a local batch
    2. Then it is added to a global batch
    3. We finally process the global batch once it is full
    4. This way, we have uniform batch sizes for the model
    """

    def __init__(
            self,
            bandwidth: float,
            single_segment_duration: int,
            overlap: float = 0.1,
            global_batch_size: int = 100
        ):
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(bandwidth)
        self.device = torch.device('cpu')
        self.pad_token = 0
        self.eos_token = -1

        # Params for batch processing
        self.overlap = overlap
        self.segment_length = self.model.sample_rate * single_segment_duration
        self.stride = int(self.segment_length - self.overlap * self.model.sample_rate)
        self.global_batch_size = global_batch_size

        self.global_batch = torch.zeros(self.global_batch_size, 1, self.segment_length, device=self.device)

    def prepare_batch(self, audio: torch.Tensor, local_batch_size: int):
        logger.info('Preparing batch process started')

        local_batch = torch.zeros(local_batch_size, 1, self.segment_length, device=self.device)
        local_batch_idx = 0

        _, length = audio.shape

        for i in range(0, length, self.stride):
            logger.debug(f'Processing segment {i} to {i+self.segment_length} and batch index {local_batch_idx}')

            # If we reach the end of the audio, pad the audio and yield the batch
            if i + self.segment_length > length:
                local_batch[local_batch_idx, 0, :length-i] = audio[:, i:]
                local_batch[local_batch_idx, 0, length-i:] = self.pad_token
                local_batch_idx += 1
                yield local_batch[:local_batch_idx].clone(), local_batch_idx
                local_batch_idx = 0
                break

            local_batch[local_batch_idx, 0, :] = audio[:, i:i+self.segment_length]
            local_batch_idx += 1

            # If we reach the max batch size, yield the batch
            if local_batch_idx == local_batch_size:
                yield local_batch.clone(), local_batch_idx
                local_batch.zero_()
                local_batch_idx = 0

        # If we have some remaining audio, yield the batch
        if local_batch_idx > 0:
            yield local_batch[:local_batch_idx].clone(), local_batch_idx

    def encode_global_batch(self, global_batch_idx: int):
        with torch.no_grad():
            emb = self.model.encoder(self.global_batch[:global_batch_idx])
            codes = self.model.quantizer.encode(
                emb, self.model.frame_rate, self.model.bandwidth
            )
            codes = codes.transpose(0, 1)  # [B, K, T]
            self.global_batch.zero_()
            yield codes

    def __call__(self, read_q: Queue[torch.Tensor], local_batch_size: int):
        """
        Implements forward pass of the Encodec model

        The input x is a list of audio files and the output is a list of list of tensors
        representing the encoded audio files
        """
        assert local_batch_size <= self.global_batch_size, f'Local batch size {local_batch_size} should be less than global batch size {self.global_batch_size}'
        global_batch_idx = 0

        # Have a global batch size
        while not read_q.empty():
            local_sample = read_q.get()
            logger.debug(f'Found an audio file to encode')

            for local_batch, local_batch_idx in self.prepare_batch(local_sample, local_batch_size):
                # If we get a local batch that is larger than the global batch size, we need to split it
                # process one part that fits in the global batch now and the rest later
                if local_batch_idx + global_batch_idx > self.global_batch_size:
                    logger.debug(f'Global batch is overflowing, yielding. Local batch index {local_batch_idx} and global batch index {global_batch_idx}')
                    self.global_batch[global_batch_idx:] = local_batch[:self.global_batch_size-global_batch_idx]
                    yield from self.encode_global_batch(self.global_batch_size)
                    # Flush the reamining local batch to the global batch
                    self.global_batch[:local_batch_idx - (self.global_batch_size-global_batch_idx)] = local_batch[self.global_batch_size-global_batch_idx:]
                    global_batch_idx = local_batch_idx - (self.global_batch_size-global_batch_idx)
                    continue

                # If we get a local batch that does not fill the global batch, we can add it to the global batch
                if local_batch_idx + global_batch_idx < self.global_batch_size:
                    logger.debug(f'Adding to global batch. Local batch index {local_batch_idx} and global batch index {global_batch_idx}')
                    self.global_batch[global_batch_idx:global_batch_idx+local_batch_idx] = local_batch
                    global_batch_idx += local_batch_idx
                    continue

                # If the local batch fills the global batch, we can yield the encoding of the global batch
                if local_batch_idx + global_batch_idx == self.global_batch_size:
                    logger.debug(f'Global batch is full, yielding, Local batch index {local_batch_idx} and global batch index {global_batch_idx}')
                    self.global_batch[global_batch_idx:global_batch_idx+local_batch_idx] = local_batch
                    yield from self.encode_global_batch(self.global_batch_size)
                    global_batch_idx = 0

        if global_batch_idx > 0:
            logger.debug(f'Global batch is not full, yielding, Local batch index {local_batch_idx} and global batch index {global_batch_idx}')
            yield from self.encode_global_batch(global_batch_idx)

if __name__ == '__main__':
    import os
    import pdb
    from time import time
    from pathlib import Path
    from .configs import VoiceEncoderConfig

    audio_file_paths = ['~/Desktop/meraki/encodec/test_24k.wav']# * 10
    audio_files: Queue[torch.Tensor] = Queue()
    save_path = './data/tokens_0.pt'

    for p in audio_file_paths:
        audio_files.put(process_audio(Path(p).expanduser(), VoiceEncoderConfig.model_sample_rate))

    voice_encoder = VoiceEncoder(
        bandwidth=VoiceEncoderConfig.bandwidth,
        single_segment_duration=VoiceEncoderConfig.single_segment_duration,
        global_batch_size=VoiceEncoderConfig.global_batch_size,
        overlap=VoiceEncoderConfig.overlap,
    )

    start_time = time()
    encoded_audio = voice_encoder(
        read_q=audio_files,
        local_batch_size=VoiceEncoderConfig.local_batch_size
    )

    result = []
    for idx, batch in enumerate(encoded_audio):
        print(idx, batch.shape)
        result.append(batch)

    logger.info(f'Encoding took {time() - start_time:.2f}s')

    torch.save(
        result,
        os.path.abspath(save_path)
    )

    audio_files_n = []
    for p in audio_file_paths:
        audio_files_n.append(process_audio(Path(p).expanduser(), VoiceEncoderConfig.model_sample_rate).unsqueeze(0))

    start_time = time()

    batches = []
    for idx, batch in enumerate(audio_files_n):
        batches.append(batch)

    batches = torch.cat(batches)
    op = voice_encoder.model.encode(batches)
    pdb.set_trace()

    logger.info(f'Encoding took {time() - start_time:.2f}s')
