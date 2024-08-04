import torch
import tiktoken
from typing import List
from queue import Queue
from encodec import EncodecModel

from .logger import get_logger

logger = get_logger(__name__)

class TextDecoder:
    """
    Simple wrapper around the TikToken to decode a list of strings
    """

    def __init__(self, tokenizer_name: str = "cl100k_base", num_threads: int = 12):
        self.decoder = tiktoken.get_encoding(tokenizer_name)
        self.num_threads = num_threads

    def __call__(self, x: List[List[int]]) -> List[str]:
        return self.decoder.decode_batch(
            x,
            num_threads=self.num_threads
        )

class VoiceDecoder:
    """
    Wrapper over Encodec model to decode a list of audio files

    >>> from .decoder import VoiceDecoder
    >>> voice_decoder = VoiceDecoder(
    >>>    bandwidth=6.0,
    >>>    single_segment_duration=2,
    >>>    overlap=0.1,
    >>>    device='cuda'
    >>> )
    >>> audio_files = Queue()
    >>> ... # Add audio files to the queue
    >>> decoded_audio = voice_decoder(read_q=audio_files)
    >>> for idx, batch in enumerate(decoded_audio):
    >>>     print(idx, batch.shape)
    """

    def __init__(
            self,
            bandwidth: float,
            single_segment_duration: int,
            overlap: float = 0.1,
            device: str = 'cpu'
        ):
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(bandwidth)
        self.device = torch.device(device)
        self.pad_token = 0
        self.eos_token = -1

        # Params for batch processing
        self.overlap = overlap
        self.segment_length = self.model.sample_rate * single_segment_duration
        self.cutoff = int(self.model.sample_rate * self.overlap)

        if device != 'cpu':
            self.model = self.model.to(device)

            torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
            torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
            # set matmul precision to use either bfloat16 or tf32
            torch.set_float32_matmul_precision("medium")
            # torch.backends.cudnn.benchmark = True  # Selects the best conv algo

            self.model = torch.compile(self.model, mode="reduce-overhead")

            # warmup the model
            input = torch.randn(1, 1, self.segment_length, device=device)
            for _ in range(5):
                self.model(input)

    def __call__(self, read_q: Queue[torch.Tensor]):
        """
        Implements forward pass of the Encodec model

        The input x is a list of audio files and the output is a list of list of tensors
        representing the encoded audio files
        """
        while not read_q.empty():
            tokens_batch = read_q.get() # B, K, T
            logger.info(f'Processing tensors of shape {tokens_batch.shape}')

            # Decode the complete batch and then join
            with torch.no_grad():
                batch_size = tokens_batch.shape[0]
                tokens_batch = tokens_batch.transpose(0, 1)
                out = self.model.quantizer.decode(tokens_batch)
                out = self.model.decoder(out) # B, 1, L
                logger.info(f'Output shape: {out.shape}')
                # Remove the overlap introduced by the encoder
                if self.cutoff > 0:
                    out = out[:, :, :-self.cutoff]
                yield out.reshape(-1, )
