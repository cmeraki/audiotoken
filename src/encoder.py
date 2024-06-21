import sys
import torch
import tiktoken
from loguru import logger
from typing import List
from queue import Queue
from encodec import EncodecModel

from .utils import process_audio
from .configs import AudioConfig

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

    >>> from src.encoder import VoiceEncoder
    >>> voice_encoder = VoiceEncoder(
    >>>    bandwidth=6.0,
    >>>    single_segment_duration=2,
    >>>    batch_size=100,
    >>>    overlap=0.1,
    >>>    device='cuda'
    >>> )
    >>> audio_files = Queue()
    >>> ... # Add audio files to the queue
    >>> encoded_audio = voice_encoder(read_q=audio_files)
    >>> for idx, batch in enumerate(encoded_audio):
    >>>     print(idx, batch.shape)
    """

    def __init__(
            self,
            bandwidth: float,
            single_segment_duration: int,
            overlap: float = 0.1,
            batch_size: int = 100,
            device: str = 'cpu'
        ):

        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(bandwidth)
        self.device = torch.device(device)
        self.pad_token = 0
        self.eos_token = -1

        # Params for batch processing
        self.overlap = overlap
        self.segment_length = self.model.sample_rate * single_segment_duration
        self.stride = int(self.segment_length - self.overlap * self.model.sample_rate)
        self.batch_size = batch_size

        # Overlap introduced in the tokens
        self.cutoff = int(75 * self.overlap)

        self.global_batch = torch.zeros(self.batch_size, 1, self.segment_length, device=self.device)

        self.model.eval()

        if device != 'cpu':
            self.model = self.model.to(device)

            torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
            torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
            torch.set_float32_matmul_precision("medium") # set matmul precision to use either bfloat16 or tf32
            # torch.backends.cudnn.benchmark = True  # Selects the best conv algo

            self.model = torch.compile(self.model, mode="reduce-overhead")

            # warmup the model
            input = torch.randn(1, 1, self.segment_length, device=device)
            for _ in range(5):
                self.model(input)

    def prepare_batch(self, audio: torch.Tensor):
        _, length = audio.shape
        segments = []

        for i in range(0, length, self.stride):
            segment = audio[:, i:i+self.segment_length]
            if segment.shape[1] < self.segment_length:
                segment = torch.nn.functional.pad(segment, (0, self.segment_length - segment.shape[1]), value=0)

            segments.append(segment)

        return torch.vstack(segments).unsqueeze(1), len(segments)

    def encode_global_batch(self, global_batch_idx: int):
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                emb = self.model.encoder(self.global_batch[:global_batch_idx])
                codes = self.model.quantizer.encode(
                    emb, self.model.frame_rate, self.model.bandwidth
                )
                # TODO: Add cutoff support
                codes = codes.transpose(0, 1)  # [B, K, T]
                self.global_batch.zero_()
                return codes

    def __call__(self, read_q: Queue[torch.Tensor]):
        """
        Implements forward pass of the Encodec model

        The input x is a list of audio files and the output is a list of list of tensors
        representing the encoded audio files
        """
        global_batch_idx = 0
        file_pointers = []

        # Have a global batch size
        while not read_q.empty():
            local_sample, local_config = read_q.get()
            local_batch, local_batch_idx = self.prepare_batch(local_sample)
            # local_config: AudioConfig

            logger.debug(f'Local batch size {local_batch_idx} and local batch shape {local_batch.shape}')

            # If we get a local batch that is larger than the global batch size, we need to split it
            # process one part that fits in the global batch now and the rest later
            if local_batch_idx + global_batch_idx > self.batch_size:
                logger.debug(f'Global batch is overflowing, yielding. Local batch index {local_batch_idx} and global batch index {global_batch_idx}')

                local_config.start_idx = global_batch_idx
                local_config.end_idx = self.batch_size
                file_pointers.append(local_config)

                # logger.info(f'Start idx : {start_idx} and end idx : {end_idx}')
                self.global_batch[global_batch_idx:] = local_batch[:self.batch_size-global_batch_idx]

                yield (self.encode_global_batch(self.batch_size), file_pointers)

                file_pointers = []
                local_config.start_idx = 0
                local_config.end_idx = local_batch_idx - (self.batch_size-global_batch_idx)
                file_pointers.append(local_config)

                # Flush the reamining local batch to the global batch
                self.global_batch[:local_batch_idx - (self.batch_size-global_batch_idx)] = local_batch[self.batch_size-global_batch_idx:]
                global_batch_idx = local_batch_idx - (self.batch_size-global_batch_idx)

                continue

            # If we get a local batch that does not fill the global batch, we can add it to the global batch
            if local_batch_idx + global_batch_idx < self.batch_size:
                logger.debug(f'Adding to global batch. Local batch index {local_batch_idx} and global batch index {global_batch_idx}')

                local_config.start_idx = global_batch_idx
                local_config.end_idx = global_batch_idx + local_batch_idx
                file_pointers.append(local_config)

                self.global_batch[global_batch_idx:global_batch_idx+local_batch_idx] = local_batch
                global_batch_idx += local_batch_idx

                continue

            # If the local batch fills the global batch, we can yield the encoding of the global batch
            if local_batch_idx + global_batch_idx == self.batch_size:
                logger.debug(f'Global batch is full, yielding, Local batch index {local_batch_idx} and global batch index {global_batch_idx}')

                local_config.start_idx = global_batch_idx
                local_config.end_idx = global_batch_idx + local_batch_idx
                file_pointers.append(local_config)

                self.global_batch[global_batch_idx:global_batch_idx+local_batch_idx] = local_batch

                yield (self.encode_global_batch(self.batch_size), file_pointers)

                file_pointers = []
                global_batch_idx = 0

        if global_batch_idx > 0:
            logger.debug(f'Global batch is not full, yielding, Local batch index {local_batch_idx} and global batch index {global_batch_idx}')
            yield (self.encode_global_batch(global_batch_idx), file_pointers)

if __name__ == '__main__':
    import os
    import pdb
    from time import time
    from pathlib import Path
    from .configs import VoiceEncoderConfig

    audio_file_paths = ['~/Desktop/meraki/encodec/test_24k.wav']# * 10
    audio_files: Queue[torch.Tensor] = Queue()
    save_path = './data/tokens_0.pt'
    device = 'cuda:0'

    for p in audio_file_paths:
        audio_files.put(process_audio(Path(p).expanduser(), VoiceEncoderConfig.model_sample_rate))

    voice_encoder = VoiceEncoder(
        bandwidth=VoiceEncoderConfig.bandwidth,
        single_segment_duration=VoiceEncoderConfig.single_segment_duration,
        batch_size=VoiceEncoderConfig.batch_size,
        overlap=VoiceEncoderConfig.overlap,
        device=device
    )

    start_time = time()
    encoded_audio = voice_encoder(read_q=audio_files)

    result = []
    for idx, batch in enumerate(encoded_audio):
        print(idx, batch)
        result.append(batch)

    print(f'Encoding took {time() - start_time:.2f}s')

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

    tensor_batches = torch.vstack(batches).to(device)
    op = voice_encoder.model.encode(tensor_batches)

    print(f'Encoding took {time() - start_time:.2f}s')
