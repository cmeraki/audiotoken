import io
import os
import torch
import psutil
import zipfile
import tarfile
import numpy as np
import torchaudio
from tqdm import tqdm
from encodec.utils import convert_audio
from datasets import load_dataset
from typing import IO, Generator

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


def process_audio_chunks(tar, member, chunk_length: int = 10, model_sample_rate: int = 16000) -> Generator:
    f = tar.extractfile(member)
    if f is None:
        return

    buffer = io.BytesIO(f.read())

    metadata = torchaudio.info(buffer)
    sample_rate = metadata.sample_rate
    num_frames = metadata.num_frames
    num_channels = metadata.num_channels
    total_seconds = num_frames/sample_rate

    logger.debug(f'Reading {member.name} with metadata: {metadata} and length: {total_seconds}')

    # Calculate chunk size in frames
    chunk_size = chunk_length * sample_rate

    for start_frame in range(0, num_frames, chunk_size):
        end_frame = min(start_frame + chunk_size, num_frames)
        buffer.seek(0)

        audio, sr = torchaudio.load(buffer, frame_offset=start_frame, num_frames=end_frame-start_frame)
        audio = convert_audio(audio, sr, model_sample_rate, 1)

        yield audio, member.name, start_frame, end_frame


def iterate_zip(x: os.PathLike) -> Generator[tuple[IO[bytes], str], None, None]:
    """
    Given a zip file, this function reads a single audio file
    at once and returns the raw bytes of the audio file

    Args:
        x (os.PathLike)

    Yields:
        Generator[tuple[IO[bytes], str], None, None]
    """
    with zipfile.ZipFile(x, 'r') as zip_file:
        for file_info in zip_file.infolist():
            if file_info.is_dir():
                continue

            file_content = zip_file.open(file_info.filename)
            file_name = file_info.filename

            if file_content is None:
                logger.error(f"Error extracting file {file_info.filename} from {x}")
                continue

            yield file_content, file_name


def iterate_tar(x: os.PathLike) -> Generator[tuple[IO[bytes], str], None, None]:
    """
    Given a tar file, this function reads a single audio file
    at once and returns the raw bytes of the audio file

    Args:
        x (os.PathLike)

    Yields:
        Generator[tuple[IO[bytes], str], None, None]
    """
    with tarfile.open(x, 'r') as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue

            file_content = tar.extractfile(member)
            file_name = member.name

            if file_content is None:
                logger.error(f"Error extracting file {file_name} from {x}")
                continue

            yield file_content, file_name


def find_audio_files(folder):
    audio_extensions = ('.mp3', '.flac', '.wav', '.ogg', '.opus')
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

    logger.info(f'Found {len(tokens_files)} files in {folder}')
    return tokens_files


def save_audio_tokens(tokens: torch.Tensor, audio_pointer: AudioConfig, root_dir: str):

    try:
        filename = audio_pointer.file_name.split('/')[-1].split('.')[0]
        save_path = os.path.join(root_dir, f'{filename}.npy')

        # B, K, T = tokens.size()
        # tokens = tokens.permute(1, 0, 2).reshape(K, B*T).cpu().numpy()

        tokens = tokens.cpu().numpy()
        tokens_len = audio_pointer.tokens_len  # type: ignore

        logger.debug(f'Saving file: {filename} with shape: {tokens.shape} to {save_path}')

        if os.path.exists(save_path):
            prev_tokens = np.load(save_path)
            prev_tokens = np.hstack([prev_tokens, tokens])
            np.save(save_path, prev_tokens[:, :tokens_len])

        else:
            np.save(save_path, tokens[:, :tokens_len])

        logger.debug(f"Saved tokens for {filename} to {save_path}")

    except Exception as e:
        logger.error(f'Error saving tokens for {audio_pointer.file_name} with error {e}')


def preprocess_audio(audio, sample_rate, processor):

    return processor(
        audio,
        sampling_rate=sample_rate,
        return_tensors='pt'
    ).input_values[0]


def get_dataset_files(indir: str, hf_dataset: str):
    assert indir or hf_dataset, "Either hf_dataset or indir must be provided"

    if indir and os.path.isdir(indir):
        return find_audio_files(indir)

    elif indir and not os.path.isdir(indir):
        return [indir]

    assert os.environ.get("HF_TOKEN"), "Please set the huggingface API token in the environment (HF_TOKEN)"

    files = []

    ds = load_dataset(
        hf_dataset,
        "xs",
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )["train"]

    for idx in tqdm(range(len(ds))):
        files.append(
            ds[idx]["audio"]["path"]
        )

    del (ds)

    return files


def set_process_affinity(process_id, cores):
    """
    Given a process id and a list of cores, this function sets the process affinity to the list of cores

    Args:
        process_id (int): The process id of the process to set affinity to
        cores (list): A list of cores to set the process affinity to

    How to use:
    ```python
    from src.utils import set_process_affinity
    # Set the process affinity to the first 4 cores
    set_process_affinity(os.getpid(), [0, 1, 2, 3])
    ```
    """
    p = psutil.Process(process_id)
    p.cpu_affinity(cores)
