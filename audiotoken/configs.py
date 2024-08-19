from math import ceil
from enum import StrEnum, auto
from typing import Optional
from dataclasses import dataclass

from huggingface_hub import hf_hub_download, snapshot_download

AUDIO_EXTS = ('.mp3', '.flac', '.wav', '.ogg', '.opus')
TAR_EXTS = ('.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz', '.tar.xz', '.txz')
ZIP_EXTS = ('.zip', '.ZIP')

class COMMONS(StrEnum):
    SEMANTIC = auto()
    ACOUSTIC = auto()
    TEXT = auto()
    HI = auto()
    EN = auto()


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
    model_id: str = 'encodec'
    model_sample_rate: int = 24_000
    bandwidth: float = 6
    model_token_rate: int = 75
    pad_token: Optional[int] = 0

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
class HubertDecoderConfig:
    suported_languages: tuple = (COMMONS.EN, )

    en_model_id: str = hf_hub_download(
        repo_id='cmeraki/audiotoken',
        repo_type='model',
        revision='5d74db4ca565e348e9d15fb782f5589cd7d0f0c0',
        filename='semantic_detokenizer/semantic_s/hubert_semantic_acoustic_gpt_en.pt'
    )

    vocab_sizes = {
        COMMONS.TEXT: 50257,
        COMMONS.SEMANTIC: 1000,
        COMMONS.ACOUSTIC: 2048,
    }
    max_source_tokens = 256

    OFFSET = {
        COMMONS.TEXT: 0,
        COMMONS.SEMANTIC: vocab_sizes[COMMONS.TEXT],
        COMMONS.ACOUSTIC: vocab_sizes[COMMONS.TEXT] + vocab_sizes[COMMONS.SEMANTIC],
    }

    max_token_value = 0
    for i in OFFSET:
        max_token_value = max(OFFSET[i] + vocab_sizes[i], max_token_value)

    pad_token = {
        COMMONS.TEXT: 50256,
        COMMONS.SEMANTIC: max_token_value + 2,
        COMMONS.ACOUSTIC: max_token_value + 3,
    }  # type: ignore

    coarse_codebooks = 2
    per_codebook_size = 1024

    INFER_TOKEN = {
        COMMONS.TEXT: max_token_value + 4,
        COMMONS.SEMANTIC: max_token_value + 5,
        COMMONS.ACOUSTIC: max_token_value + 6
    }

    STOP_TOKEN = {
        COMMONS.TEXT: max_token_value + 7,
        COMMONS.SEMANTIC: max_token_value + 8,
        COMMONS.ACOUSTIC: max_token_value + 9,
    }

    VOCAB_SIZE = (max(STOP_TOKEN.values()) // 64 + 1)*64

@dataclass
class Wav2VecBertConfig(EncoderConfig):
    model_id: str = hf_hub_download(
        repo_id='cmeraki/audiotoken',
        repo_type='model',
        revision='5d74db4ca565e348e9d15fb782f5589cd7d0f0c0',
        filename='w2vbert2_l21/model.safetensors'
    ).replace('model.safetensors', '')
    model_hf_config: str = hf_hub_download(
        repo_id='cmeraki/audiotoken',
        repo_type='model',
        revision='5d74db4ca565e348e9d15fb782f5589cd7d0f0c0',
        filename='w2vbert2_l21/config.json'
    )
    model_sample_rate: int = 16_000
    model_token_rate: int = 50
    output_layer: int = 19
    quantizer_path: Optional[str] = hf_hub_download(
        repo_id='cmeraki/audiotoken',
        repo_type='model',
        revision='5d74db4ca565e348e9d15fb782f5589cd7d0f0c0',
        filename='semantic_detokenizer/semantic_m/vq_quantizer/run4__quantizer__L19_C2048_ckpt8000.pkl'
    )
    pad_token: Optional[int] = 0

@dataclass
class Wav2VecBertDecoderConfig:
    suported_languages: tuple = (COMMONS.HI, )

    en_model_id: str = ''
    hi_model_id: str = hf_hub_download(
        repo_id='cmeraki/audiotoken',
        repo_type='model',
        revision='5d74db4ca565e348e9d15fb782f5589cd7d0f0c0',
        filename='semantic_detokenizer/semantic_m/w2vbert2_semantic_acoustic_gpt_hi.pt'
    )

    vocab_sizes = {
        COMMONS.TEXT: 50257,
        COMMONS.SEMANTIC: 1000,
        COMMONS.ACOUSTIC: 2048,
    }
    max_source_tokens = 250

    OFFSET = {
        COMMONS.TEXT: 0,
        COMMONS.SEMANTIC: vocab_sizes[COMMONS.TEXT],
        COMMONS.ACOUSTIC: vocab_sizes[COMMONS.TEXT] + vocab_sizes[COMMONS.SEMANTIC],
    }

    max_token_value = 0
    for i in OFFSET:
        max_token_value = max(OFFSET[i] + vocab_sizes[i], max_token_value)

    pad_token = {
        COMMONS.TEXT: 50256,
        COMMONS.SEMANTIC: max_token_value + 2,
        COMMONS.ACOUSTIC: max_token_value + 3,
    }  # type: ignore

    coarse_codebooks = 2
    per_codebook_size = 1024

    INFER_TOKEN = {
        COMMONS.TEXT: max_token_value + 4,
        COMMONS.SEMANTIC: max_token_value + 5,
        COMMONS.ACOUSTIC: max_token_value + 6
    }

    STOP_TOKEN = {
        COMMONS.TEXT: max_token_value + 7,
        COMMONS.SEMANTIC: max_token_value + 8,
        COMMONS.ACOUSTIC: max_token_value + 9,
    }

    VOCAB_SIZE = (max(STOP_TOKEN.values()) // 64 + 1)*64


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
