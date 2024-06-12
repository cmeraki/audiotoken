import os
import sys
import torch
import torchaudio
from encodec.utils import convert_audio
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="[{time: YYYY-MM-DD HH:mm:ss} {level}] {message}", level="INFO")

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