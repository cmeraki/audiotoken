from dataclasses import dataclass

@dataclass
class VoiceEncoderConfig:
    model_sample_rate: int = 24_000
    bandwidth: float = 1.5
    single_segment_duration: int = 10
    overlap: float = 1
    global_batch_size: int = 16
    local_batch_size: int = 8

@dataclass
class VoiceDecoderConfig(VoiceEncoderConfig):
    pass