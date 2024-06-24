import os
import sys
import torch
import numpy as np
import torchaudio
import numpy as np
from encodec.utils import convert_audio
from loguru import logger

from .configs import AudioConfig

logger.add('utils.log', format="[{time: YYYY-MM-DD HH:mm:ss} {level}] {message}", level="ERROR")

def read_audio(x: os.PathLike, model_sample_rate: int) -> torch.Tensor:
    """
    Given an audio file, this function reads the audio file and returns the audio tensor
    suitable for processing by the model
    """
    audio, sr = torchaudio.load(x)
    audio = convert_audio(audio, sr, model_sample_rate, 1)
    assert audio.shape[0] == 1, f"Audio needs to be mono, provided {audio.shape[0]} channels for {x}"
    assert audio.dim() == 2, f"Audio needs to be 2D tensor, provided {audio.dim()}D for {x}"

    logger.debug(f"Processed audio file {x}, shape {audio.shape}, length in seconds {audio.shape[1] / model_sample_rate}")

    return audio


def find_audio_files(folder):
    audio_extensions = ('.mp3', '.flac', '.wav', '.ogg')
    audio_files = []

    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(audio_extensions):
                audio_files.append(os.path.join(root, file))

    logger.info(f'Found {len(audio_files)} audio files in {folder}')
    return audio_files

def find_files(folder, extensions):
    tokens_files = []

    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(extensions):
                tokens_files.append(os.path.join(root, file))

    logger.info(f'Found {len(tokens_files)} tokens files in {folder}')
    return tokens_files

def save_audio_tokens(tokens: torch.Tensor, audio_pointer: AudioConfig, root_dir: str):

    try:
        filename = audio_pointer.file_name.split('/')[-1].split('.')[0]
        save_path = os.path.join(root_dir, f'{filename}.npy')
        tokens_to_save = tokens[audio_pointer.start_idx:audio_pointer.end_idx]
        B, K, T = tokens_to_save.size()
        tokens_to_save = tokens_to_save.permute(1, 0, 2).reshape(K, B*T).cpu().numpy()
        tokens_len = audio_pointer.tokens_len # type: ignore

        logger.info(f'Saving file: {filename} with shape: {tokens_to_save.shape} to {save_path} and length: {tokens_len} and samples: {audio_pointer.length_samples}')

        if os.path.exists(save_path):
            prev_tokens = np.load(save_path)
            prev_tokens = np.hstack([prev_tokens, tokens_to_save])
            np.save(save_path, prev_tokens[:, :tokens_len])

        else:
            np.save(save_path, tokens_to_save[:, :tokens_len])

    except Exception as e:
        print(f'Error saving tokens for {audio_pointer.file_name} with error {e}')

def preprocess_audio(audio, sample_rate, processor):

    return processor(
        audio,
        sampling_rate=sample_rate,
        return_tensors='pt'
    ).input_values[0]
