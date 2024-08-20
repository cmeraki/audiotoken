import os
import torch
import time
import psutil
import numpy as np
from tqdm import tqdm
from pathlib import Path
from functools import partial
from typing import List, Union, Optional, Callable
from torch.utils.data import DataLoader

from .logger import get_logger
from .configs import Tokenizers, EncoderConfig, AUDIO_EXTS
from .datasets import AudioBatchDataset, collate_fn
from .utils import (
    read_audio,
    save_audio,
    process_audio_chunks,
    save_audio_tokens,
    sanitize_path,
    save_rel_audio_tokens
)

logger = get_logger(__name__, log_file=None, level="WARNING")

class AudioToken:
    def __init__(
            self,
            tokenizer: Tokenizers,
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
            from audiotoken import AudioToken, Tokenizers
            encoder = AudioToken(tokenizer=Tokenizers.semantic_m, device='cuda:0')
            ```
        """
        self.tokenizer_name = Tokenizers(tokenizer)

        self.encoder: Optional[torch.nn.Module] = None
        self.decoder: Optional[torch.nn.Module] = None
        self.model_config: EncoderConfig
        self.tranform_func: Optional[Callable] = None
        self.compile = compile
        self.kwargs = kwargs
        self.device = device

    def load_encoder(self):
        if self.encoder is None:
            if self.tokenizer_name == Tokenizers.acoustic:
                from .encoder import AcousticEncoder
                from .configs import AcousticEncoderConfig

                self.encoder = AcousticEncoder(device=self.device)
                self.model_config = AcousticEncoderConfig()

            elif self.tokenizer_name == Tokenizers.semantic_s:
                from transformers import Wav2Vec2FeatureExtractor
                from .encoder import HubertEncoder, hubert_processor
                from .configs import HubertEncoderConfig

                self.encoder = HubertEncoder(device=self.device)
                self.model_config = HubertEncoderConfig()

                processor = Wav2Vec2FeatureExtractor.from_pretrained(HubertEncoderConfig.model_id)
                self.tranform_func = partial(hubert_processor, processor=processor)

            elif self.tokenizer_name == Tokenizers.semantic_m:
                from .encoder import Wav2VecBertEncoder
                from .configs import Wav2VecBertConfig

                self.encoder = Wav2VecBertEncoder(device=self.device, quantize=True)
                self.model_config = Wav2VecBertConfig()

            else:
                raise ValueError(f"Tokenizer {self.tokenizer_name} not supported")

            self.encoder.eval()
            if self.compile:
                self.encoder = torch.compile(self.encoder)  # type: ignore

            logger.info(f"Initialized {self.tokenizer_name} encoder")

    def encode(
            self,
            audio: Union[torch.Tensor, np.ndarray, os.PathLike, bytes, Path],
            chunk_size: Optional[int] = None
        ) -> torch.Tensor:
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
            from audiotoken import AudioToken, Tokenizers
            encoder = AudioToken(tokenizer=Tokenizers.acoustic, device='cuda:0')
            encoded_audio = encoder.encode('path/to/audio.wav')
            ```
        """
        self.load_encoder()

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

                return torch.cat(processed_chunks, dim=-1)

        elif isinstance(audio, bytes):
            raise NotImplementedError("Encoding bytes not supported yet")

        else:
            raise ValueError(f"Unsupported input type {type(audio)}. Should be one of: {np.ndarray, os.PathLike, bytes, Path}")

    def _encode_single(self, audio: torch.Tensor) -> torch.Tensor:
        input_batch = audio.to(self.device)
        attention_mask = torch.ones_like(input_batch, device=self.device)

        toks = self.encoder(input_batch, attention_mask) # type: ignore

        return toks.cpu()

    def encode_batch_files(
            self,
            batch_size: int,
            outdir: os.PathLike,
            chunk_size: int = 30,
            num_workers: int = 12,
            audio_files: Optional[List[os.PathLike]] = None,
            audio_dir: Optional[Union[os.PathLike, Path]] = None,
            **dataloader_kwargs
        ) -> None:
        """
        Encode a batch of audio files to tokens. The audio files can be provided as a list of paths or a directory.
        **NOTE**: `encode_batch_files` is not safe to run multiple times on the same list of files as it can result in incorrect data.
        This will be fixed in a future release.

        Args:
            batch_size (int): Batch size for encoding
            outdir (os.PathLike): Output directory to save the encoded tokens
            chunk_size (int, optional): Chunk size in seconds for batching audio files.
                Each audio file is processed in chunks of `chunk_size` seconds. Defaults to 30.
            num_workers (int, optional): Number of workers to load data. Defaults to 12.
            audio_files (Optional[List[os.PathLike]], optional): List of path of audio files. The base name of every file should be unique. Defaults to None.
            audio_dir (Optional[Union[os.PathLike, Path]], optional): Path where audio files can be found. Defaults to None.

            Either `audio_files` or `audio_dir` must be provided.

        Usage:
            ```python
            from audiotoken import AudioToken, Tokenizers
            encoder = AudioToken(tokenizer=Tokenizers.acoustic, device='cuda:0')
            encoder.encode_batch_files(
                batch_size=12,
                chunk_size=10,
                num_workers=2,
                outdir='path/to/save',
                audio_files=['path/to/audio1.wav', 'path/to/audio2.wav'],
            )
            ```
        """
        self.load_encoder()

        assert audio_files or audio_dir, "Either audio_files or audio_dir must be provided"
        assert not (audio_files and audio_dir), "Provide either audio_files or audio_dir, not both"

        outdir = sanitize_path(outdir)

        num_logical_cores = psutil.cpu_count(logical=True)
        num_workers = min(num_workers, num_logical_cores)
        if audio_files is not None:
            num_workers = min(num_workers, len(audio_files), num_logical_cores)
            logger.info(f"Encoding {len(audio_files)} audio files with {num_workers} workers")

        dataset = AudioBatchDataset(
            audio_files=audio_files, # type: ignore
            audio_dir=audio_dir, # type: ignore
            transform=self.tranform_func,
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
            logger.debug(f'Processing batch: {idx}')

            input_ids = input_ids.to(self.device)
            attention_masks = attention_masks.to(self.device)
            encoded_audio = self.encoder(input_ids, attention_masks) # type: ignore

            logger.debug(f"Processed batch: {idx}")

            for jdx, (tokens_batch, file_pointer) in enumerate(zip(encoded_audio, file_pointers)):
                logger.debug(f"Submitted saving for iteration {jdx}, batch: {idx}")

                if audio_files is not None:
                    save_audio_tokens(tokens_batch, file_pointer, str(outdir))
                    continue

                save_rel_audio_tokens(tokens_batch, file_pointer, str(outdir), str(audio_dir))

        logger.debug(f"Encoding batch files took: {time.time() - start_time:.2f}s")

    def load_decoder(self):
        if self.decoder is None:
            if self.tokenizer_name == Tokenizers.acoustic:
                from .decoder import AcousticDecoder
                self.decoder = AcousticDecoder(device=self.device)

            elif self.tokenizer_name == Tokenizers.semantic_s:
                from .decoder import HubertDecoder
                self.decoder = HubertDecoder(device=self.device)

            elif self.tokenizer_name == Tokenizers.semantic_m:
                from .decoder import Wav2VecBertDecoder
                self.decoder = Wav2VecBertDecoder(device=self.device)

            else:
                raise ValueError(f"Tokenizer {self.tokenizer_name} not supported")

            self.decoder.eval()
            if self.compile:
                self.decoder = torch.compile(self.decoder)  # type: ignore

            logger.info(f"Initialized {self.tokenizer_name} decoder")

    def decode(
            self,
            tokens: Union[torch.Tensor, np.ndarray, os.PathLike, Path],
        ) -> torch.Tensor:
        """
        Decode the tokens back to audio

        Usage:
            ```python
            from audiotoken import AudioToken, Tokenizers
            encoder = AudioToken(tokenizer=Tokenizers.acoustic, device='cuda:0')
            encoded_audio = encoder.encode('path/to/audio.wav')
            decoded_audio = encoder.decode(encoded_audio)
            ```
        """
        self.load_decoder()

        if isinstance(tokens, np.ndarray):
            return self._decode_single(torch.from_numpy(tokens))

        elif isinstance(tokens, torch.Tensor):
            return self._decode_single(tokens)

        elif isinstance(tokens, os.PathLike) or isinstance(tokens, Path):
            tokens_mem = torch.load(tokens, map_location=self.device)
            logger.debug(f'Loaded tokens from path {tokens_mem.shape}')
            return self._decode_single(tokens_mem)
        else:
            raise ValueError(f"Unsupported input type {type(tokens)}. Should be one of: {np.ndarray, os.PathLike, Path}")

    def _decode_single(self, tokens: torch.Tensor) -> torch.Tensor:
        input_batch = tokens.to(dtype=torch.long)
        toks = self.decoder(input_batch) # type:ignore

        return toks.cpu()

if __name__ == '__main__':
    from argparse import ArgumentParser

    from .utils import find_audio_files

    parser = ArgumentParser(description='Encode audio files to tokens.')

    parser.add_argument('--tokenizer', choices=Tokenizers._member_names_, type=str, required=True, help='Encoder to run.')
    parser.add_argument('--indir', type=str, required=True, help='Input filename for audio files.')
    parser.add_argument('--outdir', type=str, required=False, help='Output directory for encoded files.')

    args = parser.parse_args()

    audio_file_paths = find_audio_files(args.indir)
    device = 'cuda:0'

    print(f'Found {len(audio_file_paths)} audio files.')

    print('Running single encode func')
    tokenizer = AudioToken(tokenizer=args.tokenizer, device=device)

    encoded = [tokenizer.encode(Path(a), chunk_size=5) for a in audio_file_paths[:10]]
    for p, e in zip(audio_file_paths[:10], encoded):
        print(p, e.shape)

    print('Running single decode func')
    decoded = [tokenizer.decode(e) for e in encoded]
    for p, d in zip(audio_file_paths[:10], decoded):
        print(p, d.shape)
        save_audio(
            d,
            path=os.path.join(args.outdir, os.path.basename(p)),
            sample_rate=24_000
        )

    # print('Running batch encode func with directory')
    # encoder.encode_batch_files(
    #     batch_size=12,
    #     chunk_size=10,
    #     num_workers=2,
    #     outdir=args.outdir,
    #     audio_files=audio_file_paths,
    #     # audio_dir=args.indir,
    # )
