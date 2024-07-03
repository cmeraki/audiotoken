from math import ceil
from typing import Optional
from dataclasses import dataclass

from huggingface_hub import hf_hub_download

@dataclass
class VoiceEncoderConfig:
    model_sample_rate: int = 24_000
    bandwidth: float = 12
    single_segment_duration: int = 10
    overlap: float = 0
    batch_size: int = 64
    model_token_rate: int = 75

@dataclass
class VoiceDecoderConfig(VoiceEncoderConfig):
    pass

@dataclass
class HubertEncoderConfig:
    model_id: str = 'voidful/mhubert-base'
    model_sample_rate: int = 16_000
    single_segment_duration: int = 10
    overlap: float = 0
    batch_size: int = 64
    model_token_rate: int = 50
    quantizer_path: Optional[str] = hf_hub_download(
        repo_id=model_id,
        filename='mhubert_base_vp_en_es_fr_it3_L11_km1000.bin'
    )

@dataclass
class Wav2VecBertConfig:
    model_id: str = 'facebook/w2v-bert-2.0'
    model_sample_rate: int = 16_000
    single_segment_duration: int = 10
    model_token_rate: int = 50
    output_layer: int = -1
    quantizer_path: Optional[str] = '../data/kmeans/kmeans__L-1_C1024_ckpt150.pkl'

@dataclass
class AudioConfig:
    """
    Metadata for audio files

    Args:
        - file_name: str: Path to the audio file
        - start_idx: Optional[int]: Start index of the audio file in the batch
        - end_idx: Optional[int]: End index of the audio file in the batch
        - length_seconds: Optional[float]: Length of the audio file in seconds
        - length_samples: Optional[int]: Length of the audio file in samples (according to the sample rate in which it is read)
        - length_tokens: Optional[int]: Length of the audio file in tokens

    Properties:
        - tokens_len: int: Length of the audio file in tokens after the audio is tokenized
    """
    file_name: str
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None
    length_seconds: Optional[float] = None
    length_samples: Optional[int] = None
    length_tokens: Optional[int] = None

    @property
    def tokens_len(self) -> int:
        if self.length_tokens is None or self.length_seconds is None:
            raise ValueError("Length of tokens not set")

        return ceil(self.length_seconds * self.length_tokens) # type: ignore

@dataclass
class KMeansClusterConfig:
    max_iter: int = 150
    batch_size: int = 10000
    max_no_improvement: int = 100
    n_init: int = 20
    reassignment_ratio: float = 0.5
