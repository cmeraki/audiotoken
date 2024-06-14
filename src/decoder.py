import sys
import torch
import tiktoken
from loguru import logger
from typing import List
from queue import Queue
from encodec import EncodecModel

logger.remove()
logger.add(sys.stdout, format="[{time: YYYY-MM-DD HH:mm:ss} {level}] {message}", level="DEBUG")


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
            torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
            torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
            torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

            self.model = self.model.to(device)
            self.model = torch.compile(self.model)

            # warmup the model
            input = torch.randn(1, 1, self.segment_length, device=device)
            for _ in range(50):
                self.model(input)

    def __call__(self, read_q: Queue[torch.Tensor]):
        """
        Implements forward pass of the Encodec model

        The input x is a list of audio files and the output is a list of list of tensors
        representing the encoded audio files
        """
        while not read_q.empty():
            tokens_batch = read_q.get() #B, K, T
            logger.info(f'Processing tensors of shape {tokens_batch.shape}')

            # Decode the complete batch and then join
            with torch.no_grad():
                batch_size = tokens_batch.shape[0]
                tokens_batch = tokens_batch.transpose(0, 1)
                out = self.model.quantizer.decode(tokens_batch)
                out = self.model.decoder(out) # B, 1, L
                logger.info(f'Output shape: {out.shape}')
                # Remove the overlap introduced by the encoder
                out = out[:, :, :-self.cutoff]
                logger.info(f'Transformed output shape: {out.shape}')
                yield out.reshape(-1, )


if __name__ == '__main__':
    import pdb
    from time import time
    from pathlib import Path

    from .configs import VoiceDecoderConfig

    device='cuda:0'

    tokens_file_paths: List[str] = ['./data/tokens_0.pt']
    tokens: Queue[torch.Tensor] = Queue()
    tokens_n: Queue[torch.Tensor] = Queue()

    for p in tokens_file_paths:
        temp = torch.load(Path(p).expanduser())
        print(p, temp[0].shape)
        tokens.put(temp[0])
        tokens_n.put(temp[0])

    voice_decoder = VoiceDecoder(
        bandwidth=VoiceDecoderConfig.bandwidth,
        single_segment_duration=VoiceDecoderConfig.single_segment_duration,
        overlap=VoiceDecoderConfig.overlap,
        device=device
    )

    start_time = time()
    decoded_audio = voice_decoder(
        read_q=tokens
    )

    result = []
    for idx, batch in enumerate(decoded_audio):
        print(idx, batch.shape)
        result.append(batch)

    print(f'Decoding took {time() - start_time:.2f}s')

    # start_time = time()

    # batches = []
    # while not tokens_n.empty():
    #     batch = tokens_n.get()
    #     batches.append(batch)

    # batches = torch.cat(batches)
    # op = voice_decoder.model.decode(batches)
    # pdb.set_trace()

    # logger.info(f'Decoding took {time() - start_time:.2f}s')
