from dataclasses import dataclass

@dataclass
class VoiceEncoderConfig:
    model_sample_rate: int = 24_000
    bandwidth: float = 12
    single_segment_duration: int = 10
    overlap: float = 0
    batch_size: int = 64

@dataclass
class VoiceDecoderConfig(VoiceEncoderConfig):
    pass