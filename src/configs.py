from math import ceil
from typing import Optional
from dataclasses import dataclass

@dataclass
class VoiceEncoderConfig:
    model_sample_rate: int = 24_000
    bandwidth: float = 12
    single_segment_duration: int = 10
    overlap: float = 0
    batch_size: int = 128

@dataclass
class VoiceDecoderConfig(VoiceEncoderConfig):
    pass

@dataclass
class HubertEncoderConfig:
    model_id: str = 'voidful/mhubert-base'
    audio_sample_rate: int = 16_000
    single_segment_duration: int = 10
    overlap: float = 0
    batch_size: int = 2


@dataclass
class AudioConfig:
    file_name: str
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None
    length_seconds: Optional[float] = -1
    length_samples: Optional[int] = -1
    length_tokens: Optional[int] = -1

    @property
    def tokens_len(self) -> int:
        return ceil(self.length_seconds * self.length_tokens) # type: ignore
