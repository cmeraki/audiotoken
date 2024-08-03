import os
import torch
import numpy as np
from typing import List, Union, Optional

from .configs import TOKENIZERS
from .utils import read_audio, process_audio_chunks
from .datasets import AudioBatchDataset

class AudioToken:
    def __init__(
            self,
            tokenizer: TOKENIZERS,
            device: str = "cpu",
            compile: bool = False
        ):

        # supported_tokenzers = TOKENIZERS._member_names_
        # assert tokenizer in supported_tokenzers, f"Tokenizer {tokenizer} not supported. Should be one of {supported_tokenzers}"

        self.tokenizer: torch.nn.Module
        self.device = device

        if tokenizer == TOKENIZERS.ACOUSTIC:
            from .encoder import AcousticEncoder
            from .configs import AcousticEncoderConfig

            self.tokenizer = AcousticEncoder(device=device)
            self.config = AcousticEncoderConfig()

        elif tokenizer == TOKENIZERS.SEMANTIC_M:
            from .encoder import Wav2VecBertEncoder
            from .configs import Wav2VecBertConfig

            self.tokenizer = Wav2VecBertEncoder(device=device, quantize=True)
            self.config = Wav2VecBertConfig() # type: ignore

        self.tokenizer.eval()
        if compile:
            self.tokenizer = torch.compile(self.tokenizer)  # type: ignore

    def encode(
            self,
            audio: Union[np.ndarray, os.PathLike, bytes],
            chunk_size: Optional[int] = None
        ) -> np.ndarray:

        if isinstance(audio, np.ndarray):
            return self._encode_single(torch.from_numpy(audio))

        elif isinstance(audio, os.PathLike):
            if chunk_size is None:
                audio_sample = read_audio(audio, self.config.model_sample_rate) # type: ignore
                return self._encode_single(audio_sample)

            else:
                with open(audio, "rb") as file_stream:
                    processed_chunks = [self._encode_single(chunk)[0] for chunk, _ in process_audio_chunks(audio, file_stream, chunk_size, self.config.model_sample_rate)]

                return np.array(processed_chunks)

        elif isinstance(audio, bytes):
            raise NotImplementedError("Encoding bytes not supported yet")

        else:
            raise ValueError(f"Unsupported input type {type(audio)}")

    def _encode_single(self, audio: torch.Tensor) -> np.ndarray:
        input_batch = audio.unsqueeze(0).to(self.device)
        attention_mask = torch.ones_like(input_batch, device=self.device)

        toks = self.tokenizer(input_batch, attention_mask)

        return toks.cpu().numpy()

    def encode_batch(
            self,
            audio_files: List[os.PathLike]],
            chunk_size: Optional[int] = None
        ) -> List[np.array]:

        pass

    def _encode_batch(
            self,
            audio: List[Union[np.ndarray, os.PathLike, bytes]],
            chunk_size: Optional[int] = None
        ):

        pass