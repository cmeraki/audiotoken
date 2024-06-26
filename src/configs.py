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
    token_length: int = 75

@dataclass
class VoiceDecoderConfig(VoiceEncoderConfig):
    pass

@dataclass
class HubertEncoderConfig:
    model_id: str = 'voidful/mhubert-base'
    audio_sample_rate: int = 16_000
    single_segment_duration: int = 10
    overlap: float = 0
    batch_size: int = 32
    token_length: int = 50

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
        if self.length_tokens is None:
            raise ValueError("Length of tokens not set")

        return ceil(self.length_seconds * self.length_tokens) # type: ignore
