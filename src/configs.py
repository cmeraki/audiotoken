from dataclasses import dataclass

@dataclass
class VoiceEncoderConfig:
    model_sample_rate: int = 24_000
    bandwidth: float = 1.5
    single_segment_duration: int = 25
    overlap: float = 0
    global_batch_size: int = 20
    local_batch_size: int = 10
