import torch
import tiktoken
import joblib
import numpy as np

from loguru import logger
from typing import List, Tuple
from encodec import EncodecModel
from huggingface_hub import hf_hub_download
from transformers import HubertModel, Wav2Vec2FeatureExtractor

from .utils import read_audio, preprocess_audio
from .configs import HubertEncoderConfig, AudioConfig, VoiceEncoderConfig

logger.add('encoder.log', format="[{time: YYYY-MM-DD HH:mm:ss} {level}] {message}", level="DEBUG")

class TextEncoder:
    """
    Simple wrapper around the TikToken encoder to encode a list of strings
    """

    def __init__(self, tokenizer_name: str = "cl100k_base", num_threads: int = 12):
        self.encoder = tiktoken.get_encoding(tokenizer_name)
        self.num_threads = num_threads

    def __call__(self, x: List[str]) -> List[List[int]]:
        return self.encoder.encode_batch(
            x,
            num_threads=self.num_threads
        )

class VoiceEncoder:
    """
    Wrapper over Encodec model to encode a list of audio files.

    >>> from src.encoder import VoiceEncoder
    >>> voice_encoder = VoiceEncoder(
    >>>    bandwidth=6.0,
    >>>    single_segment_duration=2,
    >>>    batch_size=100,
    >>>    overlap=0.1,
    >>>    device='cuda'
    >>> )
    >>> audio_files = []
    >>> ... # Add audio files to the list
    >>> encoded_audio = voice_encoder(read_q=audio_files)
    >>> for idx, batch in enumerate(encoded_audio):
    >>>     print(idx, batch.shape)
    """

    def __init__(
            self,
            bandwidth: float,
            single_segment_duration: int,
            overlap: float = 0.1,
            batch_size: int = 100,
            config: VoiceEncoderConfig = VoiceEncoderConfig(),
            device: str = 'cpu'
        ):

        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(bandwidth)
        self.device = torch.device(device)
        self.config = config
        self.pad_token = 0
        self.eos_token = -1

        # Params for batch processing
        self.overlap = overlap
        self.segment_length = self.model.sample_rate * single_segment_duration
        self.stride = int(self.segment_length - self.overlap * self.model.sample_rate)
        self.batch_size = batch_size

        # Overlap introduced in the tokens
        self.cutoff = int(75 * self.overlap)

        self.global_batch = torch.zeros(self.batch_size, 1, self.segment_length, device=self.device)

        self.model.eval()

        if device != 'cpu':
            self.model = self.model.to(device)

            torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
            torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
            torch.set_float32_matmul_precision("medium") # set matmul precision to use either bfloat16 or tf32
            # torch.backends.cudnn.benchmark = True  # Selects the best conv algo

            self.model = torch.compile(self.model, mode="reduce-overhead")

            # warmup the model
            input = torch.randn(1, 1, self.segment_length, device=device)
            for _ in range(5):
                self.model(input)

    def prepare_batch(self, audio: torch.Tensor):
        _, length = audio.shape
        segments = []

        for i in range(0, length, self.stride):
            segment = audio[:, i:i+self.segment_length]
            if segment.shape[1] < self.segment_length:
                segment = torch.nn.functional.pad(segment, (0, self.segment_length - segment.shape[1]), value=0)

            segments.append(segment)

        return torch.vstack(segments).unsqueeze(1), len(segments)

    def encode_global_batch(self, global_batch_idx: int):
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                emb = self.model.encoder(self.global_batch[:global_batch_idx])
                codes = self.model.quantizer.encode(
                    emb, self.model.frame_rate, self.model.bandwidth
                )
                # TODO: Add cutoff support
                codes = codes.transpose(0, 1)  # [B, K, T]
                self.global_batch.zero_()
                return codes

    def __call__(self, read_q: List[Tuple[torch.Tensor, AudioConfig]]):
        """
        Implements forward pass of the Encodec model

        The input x is a list of audio files and the output is a list of list of tensors
        representing the encoded audio files
        """
        global_batch_idx = 0
        file_pointers = []

        # Have a global batch size
        while read_q:
            local_sample, local_config = read_q.pop()
            local_batch, local_batch_idx = self.prepare_batch(local_sample)
            local_config.length_tokens = self.config.token_length
            # local_config: AudioConfig

            logger.debug(f'Local batch size {local_batch_idx} and local batch shape {local_batch.shape}')

            # If we get a local batch that is larger than the global batch size, we need to split it
            # process one part that fits in the global batch now and the rest later
            if local_batch_idx + global_batch_idx > self.batch_size:
                logger.debug(f'Global batch is overflowing, yielding. Local batch index {local_batch_idx} and global batch index {global_batch_idx}')

                local_config.start_idx = global_batch_idx
                local_config.end_idx = self.batch_size
                file_pointers.append(local_config)

                # logger.info(f'Start idx : {start_idx} and end idx : {end_idx}')
                self.global_batch[global_batch_idx:] = local_batch[:self.batch_size-global_batch_idx]

                yield (self.encode_global_batch(self.batch_size), file_pointers)

                file_pointers = []
                local_config.start_idx = 0
                local_config.end_idx = local_batch_idx - (self.batch_size-global_batch_idx)
                file_pointers.append(local_config)

                # Flush the reamining local batch to the global batch
                self.global_batch[:local_batch_idx - (self.batch_size-global_batch_idx)] = local_batch[self.batch_size-global_batch_idx:]
                global_batch_idx = local_batch_idx - (self.batch_size-global_batch_idx)

                continue

            # If we get a local batch that does not fill the global batch, we can add it to the global batch
            if local_batch_idx + global_batch_idx < self.batch_size:
                logger.debug(f'Adding to global batch. Local batch index {local_batch_idx} and global batch index {global_batch_idx}')

                local_config.start_idx = global_batch_idx
                local_config.end_idx = global_batch_idx + local_batch_idx
                file_pointers.append(local_config)

                self.global_batch[global_batch_idx:global_batch_idx+local_batch_idx] = local_batch
                global_batch_idx += local_batch_idx

                continue

            # If the local batch fills the global batch, we can yield the encoding of the global batch
            if local_batch_idx + global_batch_idx == self.batch_size:
                logger.debug(f'Global batch is full, yielding, Local batch index {local_batch_idx} and global batch index {global_batch_idx}')

                local_config.start_idx = global_batch_idx
                local_config.end_idx = global_batch_idx + local_batch_idx
                file_pointers.append(local_config)

                self.global_batch[global_batch_idx:global_batch_idx+local_batch_idx] = local_batch

                yield (self.encode_global_batch(self.batch_size), file_pointers)

                file_pointers = []
                global_batch_idx = 0

        if global_batch_idx > 0:
            logger.debug(f'Global batch is not full, yielding, Local batch index {local_batch_idx} and global batch index {global_batch_idx}')
            yield (self.encode_global_batch(global_batch_idx), file_pointers)


class HubertEncoder:
    def __init__(
        self,
        config: HubertEncoderConfig=HubertEncoderConfig(),
        device: str = 'cpu'
    ):

        model_id = config.model_id
        self.batch_size = config.batch_size
        self.audio_sample_rate = config.audio_sample_rate
        self.segment_length = config.audio_sample_rate * config.single_segment_duration
        self.overlap = config.overlap
        self.stride = int(self.segment_length - self.overlap * self.audio_sample_rate)
        self.device = torch.device(device)
        self.config = config

        self.pad_token = 0
        self.output_layer = 11

        self.model = HubertModel.from_pretrained(model_id)#,attn_implementation="flash_attention_2", torch_dtype=torch.float16)
        self.global_batch = torch.zeros(self.batch_size, self.segment_length, device=self.device)#, dtype=torch.float16)
        self.global_attention_mask = torch.ones(self.batch_size, self.segment_length, device=self.device)#, dtype=torch.float16)

        self.model.eval()

        if device != 'cpu':
            self.model.to(self.device)

            torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
            torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
            torch.set_float32_matmul_precision("medium") # set matmul precision to use either bfloat16 or tf32
            # torch.backends.cudnn.benchmark = True  # Selects the best conv algo

            self.model = torch.compile(self.model)#, mode="reduce-overhead")

            # warmup the model
            input = torch.randn((1, 16000), device=self.device)#, dtype=torch.float16)
            for _ in range(5):
                _ = self.model(input, output_hidden_states=True).hidden_states

        kmeans_path = hf_hub_download(repo_id=model_id, filename='mhubert_base_vp_en_es_fr_it3_L11_km1000.bin')
        self.km = joblib.load(kmeans_path)
        self.C_np = self.km.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0)

        self.C = torch.from_numpy(self.C_np).t().to(self.device)
        self.Cnorm = torch.from_numpy(self.Cnorm_np).to(self.device)

        self.C = self.C.unsqueeze(0).unsqueeze(0)  # (1, 1, K, D)

        del(self.C_np)
        del(self.Cnorm_np)


    def prepare_batch(self, audio: torch.Tensor):
        _, length = audio.shape
        segments, attention_mask = [], []

        logger.debug(f'Creating segments for audio of length {length}')
        for i in range(0, length, self.stride):
            segment = audio[0, i:i+self.segment_length]
            local_attention_mask = [1] * len(segment)

            if segment.shape[0] < self.segment_length:
                local_attention_mask += [0] * (self.segment_length - segment.shape[0])
                segment = torch.nn.functional.pad(segment, (0, self.segment_length - segment.shape[0]), value=0)

            segments.append(segment)
            attention_mask.append(torch.Tensor(local_attention_mask))

        return torch.vstack(segments), torch.vstack(attention_mask).to(self.device), len(segments)  # (B, T)

    def encode_global_batch(self, global_batch_idx: int):
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                waveforms = self.global_batch[:global_batch_idx]
                attention_mask = self.global_attention_mask[:global_batch_idx]

                embeddings = self.model.forward(waveforms, attention_mask=attention_mask, output_hidden_states=True).hidden_states[self.output_layer]
                # embeddings = embeddings # (B, T, D)
                embeddings = embeddings.unsqueeze(2)  # (B, T, 1, D)

                logger.debug(f'Embeddings size: {embeddings.shape}, C size: {self.C.shape}')

                # Compute squared distances
                distances = torch.sum((embeddings - self.C)**2, dim=-1)  # (B, T, K)

                # Use in-place square root
                distances.sqrt_()  # (B, T, K)

                min_dist = torch.argmin(distances, dim=-1, keepdim=True)  # (B, T, 1)

                logger.debug(f'Min dist size: {min_dist.shape}')
                self.global_batch.zero_()

                return min_dist.transpose(1, 2) # B, 1, T

    def __call__(self, read_q: List[Tuple[torch.Tensor, AudioConfig]]):
        """
        Implements forward pass of the Encodec model

        The input x is a list of audio files and the output is a list of list of tensors
        representing the encoded audio files
        """
        global_batch_idx = 0
        file_pointers = []

        # Have a global batch size
        while read_q:
            local_sample, local_config = read_q.pop()
            local_config.length_tokens = self.config.token_length
            local_batch, local_attention_mask, local_batch_idx = self.prepare_batch(local_sample)

            logger.debug(f'Local batch size {local_batch_idx} and local batch shape {local_batch.shape}')

            # If we get a local batch that is larger than the global batch size, we need to split it
            # process one part that fits in the global batch now and the rest later
            if local_batch_idx + global_batch_idx > self.batch_size:
                logger.debug(f'Global batch is overflowing, yielding. Local batch index {local_batch_idx} and global batch index {global_batch_idx}')

                local_config.start_idx = global_batch_idx
                local_config.end_idx = self.batch_size
                file_pointers.append(local_config)

                # logger.info(f'Start idx : {start_idx} and end idx : {end_idx}')
                self.global_batch[global_batch_idx:] = local_batch[:self.batch_size-global_batch_idx]
                self.global_attention_mask[global_batch_idx:] = local_attention_mask[:self.batch_size-global_batch_idx]

                yield (self.encode_global_batch(self.batch_size), file_pointers)

                file_pointers = []
                local_config.start_idx = 0
                local_config.end_idx = local_batch_idx - (self.batch_size-global_batch_idx)
                file_pointers.append(local_config)

                # Flush the reamining local batch to the global batch
                self.global_batch[:local_batch_idx - (self.batch_size-global_batch_idx)] = local_batch[self.batch_size-global_batch_idx:]
                self.global_attention_mask[:local_batch_idx - (self.batch_size-global_batch_idx)] = local_attention_mask[self.batch_size-global_batch_idx:]

                global_batch_idx = local_batch_idx - (self.batch_size-global_batch_idx)

                continue

            # If we get a local batch that does not fill the global batch, we can add it to the global batch
            if local_batch_idx + global_batch_idx < self.batch_size:
                logger.debug(f'Adding to global batch. Local batch index {local_batch_idx} and global batch index {global_batch_idx}')

                local_config.start_idx = global_batch_idx
                local_config.end_idx = global_batch_idx + local_batch_idx
                file_pointers.append(local_config)

                self.global_batch[global_batch_idx:global_batch_idx+local_batch_idx] = local_batch
                self.global_attention_mask[global_batch_idx:global_batch_idx+local_batch_idx] = local_attention_mask

                global_batch_idx += local_batch_idx

                continue

            # If the local batch fills the global batch, we can yield the encoding of the global batch
            if local_batch_idx + global_batch_idx == self.batch_size:
                logger.debug(f'Global batch is full, yielding, Local batch index {local_batch_idx} and global batch index {global_batch_idx}')

                local_config.start_idx = global_batch_idx
                local_config.end_idx = global_batch_idx + local_batch_idx
                file_pointers.append(local_config)

                self.global_batch[global_batch_idx:global_batch_idx+local_batch_idx] = local_batch
                self.global_attention_mask[global_batch_idx:global_batch_idx+local_batch_idx] = local_attention_mask

                yield (self.encode_global_batch(self.batch_size), file_pointers)

                file_pointers = []
                global_batch_idx = 0

        if global_batch_idx > 0:
            logger.debug(f'Global batch is not full, yielding, Local batch index {local_batch_idx} and global batch index {global_batch_idx}')
            yield (self.encode_global_batch(global_batch_idx), file_pointers)

if __name__ == '__main__':
    from time import time
    from pathlib import Path
    from argparse import ArgumentParser
    from .configs import VoiceEncoderConfig

    parser = ArgumentParser(description='Encode audio files.')
    parser.add_argument('--audio_file', type=str, required=True, help='Input filename for audio files.')
    args = parser.parse_args()

    audio_file_paths = [args.audio_file]
    audio_files = []
    save_path = './tmp/tokens_0.pt'
    device = 'cuda:0'

    for p in audio_file_paths:
        a = read_audio(Path(p).expanduser(), VoiceEncoderConfig.model_sample_rate)
        audio_files.append((
            a,
            AudioConfig(file_name=p, length_seconds=a.shape[-1], length_samples=a.shape[-1] * 24_000, length_tokens=50)
        ))

    voice_encoder = VoiceEncoder(
        bandwidth=VoiceEncoderConfig.bandwidth,
        single_segment_duration=VoiceEncoderConfig.single_segment_duration,
        batch_size=VoiceEncoderConfig.batch_size,
        overlap=VoiceEncoderConfig.overlap,
        device=device
    )

    start_time = time()
    encoded_audio = voice_encoder(read_q=audio_files)

    result = []
    for idx, batch in enumerate(encoded_audio):
        print(idx, batch)
        result.append(batch)
        np.save(save_path, batch[0].detach().cpu().numpy())

    print(f'Encodec encoding took {time() - start_time:.2f}s')

    audio_files = []
    processor = Wav2Vec2FeatureExtractor.from_pretrained(HubertEncoderConfig.model_id)

    for p in audio_file_paths:
        a = read_audio(Path(p).expanduser(), 16_000)
        a = preprocess_audio(a, 16_000, processor)
        audio_files.append((
            a,
            AudioConfig(file_name=p, length_seconds=a.shape[-1], length_samples=a.shape[-1] * 16_000)
        ))

    hubert_encoder = HubertEncoder(
        config=HubertEncoderConfig(),
        device=device
    )

    start_time = time()
    encoded_audio = hubert_encoder(audio_files)

    for idx, batch in enumerate(encoded_audio):
        print(idx, batch, batch[0].shape)

    print(f'Hubert encoding took {time() - start_time:.2f}s')
