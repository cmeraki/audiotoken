import pathlib
import io
import os
import torch
import psutil
import zipfile
import tarfile
import numpy as np
import torchaudio

from tqdm import tqdm
from torchaudio.io import StreamReader
from encodec.utils import convert_audio
from datasets import load_dataset
from typing import IO, Generator

from .configs import AudioConfig
from .logger import get_logger

logger = get_logger(__name__)

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


def process_audio_chunks(
    file_name,
    file_stream,
    target_sample_rate,
    chunk_size
):
    streamer = StreamReader(file_stream)
    metadata = streamer.get_src_stream_info(0)

    streamer.add_basic_audio_stream(
        frames_per_chunk=int(chunk_size*metadata.sample_rate),
        decoder_option={"threads": "0"}
    )

    for idx, (chunk,) in enumerate(streamer.stream()):
        assert chunk.shape[-1] == 1, f"Audio needs to be mono, provided {chunk.shape[-1]} channels for {file_name}"

        start_idx = idx * chunk_size
        end_idx = start_idx + chunk_size
        # base, ext = os.path.splitext(file_name)
        # updated_file_name = f"{base}__{start_idx}_{end_idx}{ext}"

        # Using basic audio stream resampling vs torch resampling results in different behaviors
        # To keep it consistent with `convert_audio`, we use torchaudio.transforms.Resample
        # https://stackoverflow.com/questions/77438128/how-to-resample-from-8k-to-16k-with-librosa-or-torchaudio-as-ffmpeg-do-it
        chunk = chunk.reshape(1, -1)
        chunk = torchaudio.transforms.Resample(metadata.sample_rate, target_sample_rate)(chunk)

        yield chunk, file_name


def iterate_zip(x: os.PathLike, model_sample_rate: int, chunk_size: int = 30) -> Generator[tuple[IO[bytes], str], None, None]:
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
            logger.info(f'Processing {file_name} in zip: {x}')

            if file_content is None:
                logger.error(f"Error extracting file {file_info.filename} from {x}")
                continue

            yield from process_audio_chunks(
                file_name=file_name,
                file_stream=file_content,
                target_sample_rate=model_sample_rate,
                chunk_size=chunk_size
            )

        logger.debug(f'Processed {file_name} in zip: {x}')


def iterate_tar(x: os.PathLike, model_sample_rate: int, chunk_size: int = 30) -> Generator[tuple[IO[bytes], str], None, None]:
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
            logger.debug(f'Processing {member.name} in tar: {x}')
            if not member.isfile():
                continue

            file_content = tar.extractfile(member)
            file_name = member.name

            if file_content is None:
                logger.error(f"Error extracting file {file_name} from {x}")
                continue

            yield from process_audio_chunks(
                file_name=file_name,
                file_stream=file_content,
                target_sample_rate=model_sample_rate,
                chunk_size=chunk_size
            )

            logger.debug(f'Processed {file_name} in tar: {x}')


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
        length_tokens = audio_pointer.length_tokens  # type: ignore
        tokens = tokens[:, :length_tokens]

        logger.debug(f'Saving file: {filename} with shape: {tokens.shape} to {save_path}')

        if os.path.exists(save_path):
            prev_tokens = np.load(save_path)
            prev_tokens = np.hstack([prev_tokens, tokens])
            np.save(save_path, prev_tokens)

        else:
            np.save(save_path, tokens[:, :length_tokens])

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
        "s",
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
    from .utils import set_process_affinity
    # Set the process affinity to the first 4 cores
    set_process_affinity(os.getpid(), [0, 1, 2, 3])
    ```
    """
    p = psutil.Process(process_id)
    p.cpu_affinity(cores)


def hertz_to_mel(freq: torch.Tensor) -> torch.Tensor:
    """
    Convert frequency in Hertz to Mel scale using Kaldi scale

    Args:
        freq (torch.Tensor)

    Returns:
        torch.Tensor
    """
    return 1127.0 * torch.log(1.0 + (freq / 700.0))


def mel_to_hertz(mels: torch.Tensor) -> torch.Tensor:
    """
    Convert frequency in Mel scale to Hertz using Kaldi scale

    Args:
        mels (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """
    return 700.0 * (torch.exp(mels / 1127.0) - 1.0)


def create_triangular_filter_bank(fft_freqs: torch.Tensor, filter_freqs: torch.Tensor) -> torch.Tensor:
    """
    Create a triangular filter bank given the fft frequencies and filter frequencies

    Args:
        fft_freqs (torch.Tensor): _description_
        filter_freqs (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """
    filter_diff = torch.diff(filter_freqs)
    slopes = filter_freqs.unsqueeze(0) - fft_freqs.unsqueeze(1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]

    return torch.maximum(torch.zeros(1), torch.minimum(down_slopes, up_slopes))


def load_vq_weights(model_weights, model):
    new_state_dict = {}

    for k, v in model_weights.items():
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict)

    return model


def sanitize_path(path):
    path = pathlib.Path(path).expanduser()

    if not path.is_absolute():
        path = path.absolute()

    path = path.resolve()

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    return str(path)


def collate_audio_tokens(prev_tokens: np.ndarray, new_tokens: torch.Tensor, audio_pointer: AudioConfig):

    new_tokens = new_tokens.cpu().numpy()
    length_tokens = audio_pointer.length_tokens  # type: ignore

    tokens = np.hstack([prev_tokens, new_tokens])
    tokens = tokens[:, :length_tokens]

    return tokens


def save_rel_audio_tokens(tokens: torch.Tensor, audio_pointer: AudioConfig, root_dir: str, rel_dir: str):

    try:
        tokens = tokens.cpu().numpy()
        length_tokens = audio_pointer.length_tokens  # type: ignore

        rel_path = os.path.relpath(audio_pointer.file_name, start=rel_dir)
        rel_path = os.path.dirname(rel_path)
        output_path = os.path.join(root_dir, rel_path)

        os.makedirs(output_path, exist_ok=True)

        filename = os.path.splitext(os.path.basename(audio_pointer.file_name))[0]
        save_path = os.path.join(output_path, f'{filename}.npy')

        logger.debug(f'Saving file: {filename} with shape: {length_tokens} to {save_path}')

        if os.path.exists(save_path):
            prev_tokens = np.load(save_path)
            prev_tokens = np.hstack([prev_tokens, tokens])
            np.save(save_path, prev_tokens[:, :length_tokens])

        else:
            np.save(save_path, tokens[:, :length_tokens])

        logger.debug(f"Saved tokens for {filename} to {save_path}")

    except Exception as e:
        logger.error(f'Error saving tokens for {audio_pointer.file_name} with error {e}')
