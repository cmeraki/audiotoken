import os
import sys
import torch
import numpy as np
import torchaudio
import numpy as np
from encodec.utils import convert_audio

from .configs import AudioConfig
from .logger import logger

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

        # B, K, T = tokens.size()
        # tokens = tokens.permute(1, 0, 2).reshape(K, B*T).cpu().numpy()

        tokens = tokens.cpu().numpy()
        tokens_len = audio_pointer.tokens_len # type: ignore

        logger.info(f'Saving file: {filename} with shape: {tokens.shape} to {save_path}')

        if os.path.exists(save_path):
            prev_tokens = np.load(save_path)
            prev_tokens = np.hstack([prev_tokens, tokens])
            np.save(save_path, prev_tokens[:, :tokens_len])

        else:
            np.save(save_path, tokens[:, :tokens_len])

        logger.info(f"Saved tokens for {filename} to {save_path}")

    except Exception as e:
        logger.error(f'Error saving tokens for {audio_pointer.file_name} with error {e}')

def preprocess_audio(audio, sample_rate, processor):

    return processor(
        audio,
        sampling_rate=sample_rate,
        return_tensors='pt'
    ).input_values[0]
