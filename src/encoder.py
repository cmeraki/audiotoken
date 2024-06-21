import pdb
import sys
import torch
import torchaudio
import tiktoken
import joblib
from loguru import logger
from typing import List
from queue import Queue
from encodec import EncodecModel
from huggingface_hub import hf_hub_download
from transformers import HubertModel, Wav2Vec2FeatureExtractor

from .utils import process_audio
from .configs import HubertEncoderConfig, AudioConfig

logger.add(sys.stdout, format="[{time: YYYY-MM-DD HH:mm:ss} {level}] {message}", level="ERROR")

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
    >>> audio_files = Queue()
    >>> ... # Add audio files to the queue
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

    def __call__(self, read_q: Queue[torch.Tensor]):
        """
        Implements forward pass of the Encodec model

        The input x is a list of audio files and the output is a list of list of tensors
        representing the encoded audio files
        """
        global_batch_idx = 0
        file_pointers = []

        # Have a global batch size
        while not read_q.empty():
            local_sample, local_config = read_q.get()
            local_batch, local_batch_idx = self.prepare_batch(local_sample)
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
        config: HubertEncoderConfig,
        device: str = 'cpu'
    ):

        model_id = config.model_id
        self.batch_size = config.batch_size
        self.audio_sample_rate = config.audio_sample_rate
        self.segment_length = config.audio_sample_rate * config.single_segment_duration
        self.overlap = config.overlap
        self.stride = int(self.segment_length - self.overlap * self.audio_sample_rate)
        self.device = torch.device(device)

        self.pad_token = 0
        self.output_layer = 11

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        self.model = HubertModel.from_pretrained(model_id)
        self.global_batch = torch.zeros(self.batch_size, self.segment_length, device=self.device)

        if device != 'cpu':
            self.model = self.model.to(self.device)

            torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
            torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
            torch.set_float32_matmul_precision("medium") # set matmul precision to use either bfloat16 or tf32
            # torch.backends.cudnn.benchmark = True  # Selects the best conv algo

            self.model = torch.compile(self.model, mode="reduce-overhead")

            # warmup the model
            input = torch.randn((1, 16000), device=self.device)
            for _ in range(5):
                i = self.processor(input, sampling_rate=self.audio_sample_rate, return_tensors='pt').input_values[0]
                i = i.to(self.device)
                _ = self.model(i, output_hidden_states=True).hidden_states

        kmeans_path = hf_hub_download(repo_id=model_id, filename='mhubert_base_vp_en_es_fr_it3_L11_km1000.bin')
        self.km = joblib.load(kmeans_path)
        self.C_np = self.km.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0)

        self.C = torch.from_numpy(self.C_np).t().to(self.device)
        self.Cnorm = torch.from_numpy(self.Cnorm_np).to(self.device)

        # pdb.set_trace()

        del(self.C_np)
        del(self.Cnorm_np)


    def prepare_batch(self, audio: torch.Tensor):
        _, length = audio.shape
        segments = []

        for i in range(0, length, self.stride):
            segment = audio[0, i:i+self.segment_length]
            if segment.shape[0] < self.segment_length:
                segment = torch.nn.functional.pad(segment, (0, self.segment_length - segment.shape[0]), value=0)

            segments.append(segment)

        return torch.vstack(segments), len(segments), # (B, T)

    def encode_global_batch(self, global_batch_idx: int):
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                waveforms = self.global_batch[:global_batch_idx]
                waveforms = self.processor(
                    waveforms,
                    sampling_rate=self.audio_sample_rate,
                    return_tensors='pt'
                ).input_values[0]
                waveforms = waveforms.to(self.device)

                embeddings = self.model(waveforms, output_hidden_states=True).hidden_states
                embeddings = embeddings[self.output_layer] # (B, T, D)

                logger.info(f'Embeddings size: {embeddings.shape}, C size: {self.C.shape}')
                distances = torch.sum(
                    embeddings.unsqueeze(2) - self.C.unsqueeze(0).unsqueeze(0), dim=-1
                )
                distances = torch.sqrt(distances) # B, T, K

                min_dist = torch.topk(distances, 1, dim=-1, largest=False)
                greedy_output = min_dist.indices.T

                return greedy_output

    def __call__(self, read_q):
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

                try:
                    # Flush the reamining local batch to the global batch
                    self.global_batch[:local_batch_idx - (self.batch_size-global_batch_idx)] = local_batch[self.batch_size-global_batch_idx:]
                    global_batch_idx = local_batch_idx - (self.batch_size-global_batch_idx)

                except:
                    import pdb
                    pdb.set_trace()

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

if __name__ == '__main__':
    import os
    from time import time
    from pathlib import Path
    from .configs import VoiceEncoderConfig

    audio_file_paths = ['~/Desktop/meraki/encodec/test_24k.wav']# * 10
    # audio_files: Queue[torch.Tensor] = Queue()
    save_path = './data/tokens_0.pt'
    device = 'cuda:0'

    # for p in audio_file_paths:
    #     audio_files.put(process_audio(Path(p).expanduser(), VoiceEncoderConfig.model_sample_rate))

    # voice_encoder = VoiceEncoder(
    #     bandwidth=VoiceEncoderConfig.bandwidth,
    #     single_segment_duration=VoiceEncoderConfig.single_segment_duration,
    #     batch_size=VoiceEncoderConfig.batch_size,
    #     overlap=VoiceEncoderConfig.overlap,
    #     device=device
    # )

    # start_time = time()
    # encoded_audio = voice_encoder(read_q=audio_files)

    # result = []
    # for idx, batch in enumerate(encoded_audio):
    #     print(idx, batch)
    #     result.append(batch)

    # print(f'Encoding took {time() - start_time:.2f}s')

    # torch.save(
    #     result,
    #     os.path.abspath(save_path)
    # )

    # audio_files_n = []
    # for p in audio_file_paths:
    #     audio_files_n.append(process_audio(Path(p).expanduser(), VoiceEncoderConfig.model_sample_rate).unsqueeze(0))

    # start_time = time()

    # batches = []
    # for idx, batch in enumerate(audio_files_n):
    #     batches.append(batch)

    # tensor_batches = torch.vstack(batches).to(device)
    # op = voice_encoder.model.encode(tensor_batches)

    # print(f'Encoding took {time() - start_time:.2f}s')

    audio_files = []
    for p in audio_file_paths:
        a, sr = torchaudio.load(Path(p).expanduser())
        audio_files.append((
            a,
            AudioConfig(file_name=p, length_seconds=a.shape[-1], length_samples=a.shape[-1] * 16_000, length_tokens=50)
        ))

    hubert_encoder = HubertEncoder(
        config=HubertEncoderConfig(),
        device=device
    )

    start_time = time()
    encoded_audio = hubert_encoder(audio_files)

    for idx, batch in enumerate(encoded_audio):
        print(idx, batch, batch[0].shape)

    print(f'Encoding took {time() - start_time:.2f}s')
