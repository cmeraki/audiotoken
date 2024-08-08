from math import ceil
from enum import StrEnum, auto
from typing import Optional
from dataclasses import dataclass

from huggingface_hub import hf_hub_download, snapshot_download

AUDIO_EXTS = ('.mp3', '.flac', '.wav', '.ogg', '.opus')
TAR_EXTS = ('.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz', '.tar.xz', '.txz')
ZIP_EXTS = ('.zip', '.ZIP')


class Tokenizers(StrEnum):
    acoustic=auto()
    semantic_s=auto()
    semantic_m=auto()


@dataclass
class EncoderConfig:
    model_id: str # Model ID for the encoder
    model_sample_rate: int # Sample rate of the audio file that the model expects
    model_token_rate: int # Number of tokens produced by the model per second
    pad_token: Optional[int] # Padding token for the model

@dataclass
class AcousticEncoderConfig(EncoderConfig):
    model_id: str = 'encodec'
    model_sample_rate: int = 24_000
    bandwidth: float = 12
    model_token_rate: int = 75
    pad_token: Optional[int] = 0

@dataclass
class AcousticDecoderConfig(AcousticEncoderConfig):
    pass

@dataclass
class HubertEncoderConfig(EncoderConfig):
    model_id: str = 'voidful/mhubert-base'
    model_sample_rate: int = 16_000
    output_layer: int = 11
    model_token_rate: int = 50
    quantizer_path: Optional[str] = hf_hub_download(
        repo_id=model_id,
        filename='mhubert_base_vp_en_es_fr_it3_L11_km1000.bin'
    )
    pad_token: Optional[int] = 0

@dataclass
class Wav2VecBertConfig(EncoderConfig):
    model_id: str = hf_hub_download(
        repo_id='cmeraki/w2vbert2_L19',
        repo_type='model',
        revision='c5fc4e6c2db24eec909ec678fe2b8debcc59ffed',
        filename='model.safetensors'
    ).replace('model.safetensors', '')
    model_hf_config: str = hf_hub_download(
        repo_id='cmeraki/w2vbert2_L19',
        repo_type='model',
        revision='c5fc4e6c2db24eec909ec678fe2b8debcc59ffed',
        filename='config.json'
    )
    model_sample_rate: int = 16_000
    model_token_rate: int = 50
    output_layer: int = 19
    quantizer_path: Optional[str] = hf_hub_download(
        repo_id='cmeraki/w2vbert2_vq_quantizer',
        repo_type='model',
        revision='dcaa88d656395c0a8eaf61350d2f358cff3328ee',
        filename='quantizer__L19_C2048_ckpt9000.pkl'
    )
    pad_token: Optional[int] = 0

@dataclass
class WhisperEncoderConfig:
    model_id: str = 'openai/whisper-tiny'
    model_sample_rate: int = 16_000
    single_segment_duration: int = 30 # Whisper converts the audio to images of mel spectrograms
    model_token_rate: int = 100 # Whisper has a fixed token rate
    output_layer: int = -1
    quantizer_path: Optional[str] = 'data/kmeans_whisper_xs/kmeans__L4_C1024_ckpt1450.pkl'
    pad_token: Optional[int] = None

@dataclass
class AudioConfig:
    """
    Metadata for audio files

    Args:
        - file_name: str: Path to the audio file
        - start_idx: Optional[int]: Start index of the audio in the complete file (in samples)
        - end_idx: Optional[int]: End index of the audio in the complete file (in samples)
        - length_seconds: Optional[float]: Length of the audio in seconds
        - length_samples: Optional[int]: Length of the audio in samples
        - model_token_rate: Optional[int]: Number of tokens produced by the model per second

    Properties:
        - length_tokens: int: Length of the audio file in tokens after the audio is tokenized
    """
    file_name: str
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None
    length_seconds: Optional[float] = None
    length_samples: Optional[int] = None
    model_token_rate: Optional[int] = None

    @property
    def length_tokens(self) -> int:
        if self.model_token_rate is None or self.length_seconds is None:
            raise ValueError("Model token rate or length of the audio file is not provided")

        return ceil(self.length_seconds * self.model_token_rate) # type: ignore

@dataclass
class KMeansClusterConfig:
    max_iter: int = 150
    batch_size: int = 64000
    max_no_improvement: int = 100
    n_init: int = 5
    reassignment_ratio: float = 0.5
