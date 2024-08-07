import os
import torch
import time
import psutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Union, Optional
from torch.utils.data import DataLoader

from .logger import get_logger
from .configs import TOKENIZERS, EncoderConfig
from .datasets import AudioBatchDataset, collate_fn
from .utils import (
    read_audio,
    process_audio_chunks,
    save_audio_tokens,
    sanitize_path
)

logger = get_logger(__name__, log_file=None, level="DEBUG")

class AudioToken:
    def __init__(
            self,
            tokenizer: TOKENIZERS,
            device: str = "cpu",
            compile: bool = False,
            **kwargs
        ):
        """
        Initialize the AudioToken class. One can choose between two tokenizers:
        - Acoustic Encoder: Enocdes audio to acoustic tokens
        - Semantic Encoder: Encodes audio to semantic tokens

        Args:
            tokenizer (TOKENIZERS): Tokenizer to use, either ACOUSTIC or SEMANTIC_M
            device (str, optional): Device to use for the tokenizer. Defaults to "cpu".
            compile (bool, optional): Weather to compile the model or not. Defaults to False.
                Note: When you use compile, the first encoding will be slow but subsequent encodings will be faster

        Raises:
            ValueError: If the tokenizer is not supported

        Usage:
            ```python
            from audiotoken import AudioToken
            encoder = AudioToken(tokenizer='ACOUSTIC', device='cuda:0')
            ```
        """

        self.tokenizer: torch.nn.Module
        self.model_config: EncoderConfig

        self.device = device

        if tokenizer == TOKENIZERS.ACOUSTIC.value:
            from .encoder import AcousticEncoder
            from .configs import AcousticEncoderConfig

            self.tokenizer = AcousticEncoder(device=device)
            self.model_config = AcousticEncoderConfig()

        elif tokenizer == TOKENIZERS.SEMANTIC_M.value:
            from .encoder import Wav2VecBertEncoder
            from .configs import Wav2VecBertConfig

            self.tokenizer = Wav2VecBertEncoder(device=device, quantize=True)
            self.model_config = Wav2VecBertConfig

        else:
            raise ValueError(f"Tokenizer {tokenizer} not supported")

        self.tokenizer.eval()
        if compile:
            self.tokenizer = torch.compile(self.tokenizer)  # type: ignore

        logger.info(f"Initialized {tokenizer} tokenizer on {device} with compile={compile}")

    def encode(
            self,
            audio: Union[torch.Tensor, np.ndarray, os.PathLike, bytes, Path],
            chunk_size: Optional[int] = None
        ) -> np.ndarray:
        """
        Encode the audio file to tokens. The audio can be provided as a numpy array, path to the audio file, or bytes.

        Args:
            audio (Union[np.ndarray, os.PathLike, bytes, Path]): Audio to encode
            chunk_size (Optional[int], optional): Chunk size (in seconds) to read the audio in memory.
             This is only used when `audio` is provided as `os.PathLike` or `Path`.
             This is specially usefule for large audio files. Defaults to None.

        Raises:
            NotImplementedError: If the provided `audio` is bytes
            ValueError: If the provided `audio` is not one of [np.ndarray, os.PathLike, bytes, Path]

        Returns:
            np.ndarray: Encoded tokens in the shape of (1, 16, num_tokens) for a single audio file

        Usage:
            ```python
            from audiotoken import AudioToken
            encoder = AudioToken(tokenizer='ACOUSTIC', device='cuda:0')
            encoded_audio = encoder.encode('path/to/audio.wav')
            ```
        """

        if isinstance(audio, np.ndarray):
            return self._encode_single(torch.from_numpy(audio))

        elif isinstance(audio, torch.Tensor):
            return self._encode_single(audio)

        elif isinstance(audio, os.PathLike) or isinstance(audio, Path):
            if chunk_size is None:
                logger.debug(f"Encoding single audio file: {audio} with no chunking")
                logger.warning(f"Chunking not provided. Encoding the complete audio file at once. May run out of memory for larger audio files.")

                audio_sample = read_audio(audio, self.model_config.model_sample_rate) # type: ignore
                logger.debug(audio_sample.shape)
                return self._encode_single(audio_sample)

            else:
                logger.debug(f"Encoding audio file: {audio} with chunking of size {chunk_size}")

                with open(audio, "rb") as file_stream:
                    processed_chunks = [self._encode_single(chunk)[0] for chunk, _ in process_audio_chunks(
                        audio, file_stream, self.model_config.model_sample_rate, chunk_size)]

                return np.concatenate(processed_chunks, axis=-1).reshape(1, 16, -1)

        elif isinstance(audio, bytes):
            raise NotImplementedError("Encoding bytes not supported yet")

        else:
            raise ValueError(f"Unsupported input type {type(audio)}. Should be one of: {np.ndarray, os.PathLike, bytes, Path}")

    def _encode_single(self, audio: torch.Tensor) -> np.ndarray:
        input_batch = audio.to(self.device)
        attention_mask = torch.ones_like(input_batch, device=self.device)

        toks = self.tokenizer(input_batch, attention_mask)

        return toks.cpu().numpy()

    def encode_batch_files(
            self,
            audio_files: List[os.PathLike],
            batch_size: int,
            outdir: os.PathLike,
            chunk_size: int = 30,
            num_workers: int = 12,
            **dataloader_kwargs
        ):

        outdir = sanitize_path(outdir)

        num_logical_cores = psutil.cpu_count(logical=True)
        num_workers = min(num_workers, len(audio_files), num_logical_cores)

        logger.info(f"Encoding {len(audio_files)} audio files with {num_workers} workers")

        dataset = AudioBatchDataset(
            audio_files, # type: ignore
            chunk_size=chunk_size,
            sample_rate=self.model_config.model_sample_rate,
            model_token_rate=self.model_config.model_token_rate,
            pad_token=self.model_config.pad_token
        )
        dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=num_workers,
                prefetch_factor=dataloader_kwargs.get('prefetch_factor', 4),
                pin_memory=dataloader_kwargs.get('pin_memory', True)
            )

        start_time = time.time()

        for idx, (input_ids, attention_masks, file_pointers) in tqdm(enumerate(dataloader)):
            logger.info(f'Processing batch: {idx}')

            input_ids = input_ids.to(self.device)
            attention_masks = attention_masks.to(self.device)
            encoded_audio = self.tokenizer(input_ids, attention_masks)

            logger.info(f"Processed batch: {idx}")

            for jdx, (tokens_batch, file_pointer) in enumerate(zip(encoded_audio, file_pointers)):
                logger.debug(f"Submitted saving for iteration {jdx}, batch: {idx}")
                save_audio_tokens(tokens_batch, file_pointer, str(outdir))

        logger.info(f"Encoding batch files took: {time.time() - start_time:.2f}s")

    def encode_batch(
            self,
            audio: List[Union[np.ndarray, os.PathLike, bytes]],
            chunk_size: Optional[int] = None
        ):

        raise NotImplementedError("Batch encoding without files not supported yet. Please use `encode_batch_files` for now")


if __name__ == '__main__':
    from argparse import ArgumentParser

    from .utils import find_audio_files

    parser = ArgumentParser(description='Encode audio files to tokens.')

    parser.add_argument('--tokenizer', choices=TOKENIZERS._member_names_, type=str, required=True, help='Encoder to run.')
    parser.add_argument('--indir', type=str, required=True, help='Input filename for audio files.')
    parser.add_argument('--outdir', type=str, required=False, help='Output directory for encoded files.')

    args = parser.parse_args()

    audio_file_paths = find_audio_files(args.indir)
    device = 'cuda:0'

    print(f'Found {len(audio_file_paths)} audio files.')

    print('Running single encode func')
    encoder = AudioToken(tokenizer=args.tokenizer, device=device)
    encoded = [encoder.encode(Path(a), chunk_size=5) for a in audio_file_paths[:10]]
    print([e.shape for e in encoded])

    print('Running batch encode func')
    os.makedirs(args.outdir, exist_ok=True)
    encoder.encode_batch_files(
        audio_file_paths[:100],
        batch_size=12,
        chunk_size=10,
        outdir=args.outdir
    )
