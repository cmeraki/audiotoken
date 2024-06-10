import torch
import tiktoken
import torchaudio
import concurrent.futures
import multiprocessing as mp
from typing import List
from functools import partial
from encodec import EncodecModel
from encodec.utils import convert_audio
from concurrent.futures import ThreadPoolExecutor

class TextEncoder():
    """
    Simple wrapper around the TikToken encoder to encode a list of strings
    """

    def __init__(self, tokenizer_name: str = "cl100k_base"):
        self.encoder = tiktoken.get_encoding(tokenizer_name)

    def __call__(self, x: List[str]) -> List[List[int]]:
        return self.encoder.encode_batch(x)

class VoiceEncoder():
    """
    Wrapper over encodec model to encode a list of audio files.
    Much faster than the original implementation as it uses multi-threading
    and batch processing
    """
    def __init__(self, bandwidth, batch_size: int):
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(bandwidth)
        self.device = 'cpu'
        self.pad_token = 0

        # Params for batch processing
        self.segment_length = self.model.sample_rate
        self.stride = self.segment_length - 0.1*self.model.sample_rate # 10 ms overlap
        self.global_batch_size = batch_size

        # Multiprocessing Queues
        self.audio_q: mp.Queue = mp.Queue()
        self.processed_q: mp.Queue = mp.Queue()
        self.read_complete = mp.Value('b', False)

    def preprocessing(self, x: str, model_sample_rate: int):
        """
        Given a audio file, this function reads the audio file and returns the audio tensor
        suitable for processing by the model
        """
        audio, sr = torchaudio.load(x)
        audio = convert_audio(audio, sr, model_sample_rate, 1)
        assert audio.channels == 1, f"Audio needs to be mono, provided {audio.channels} for {x}"
        assert audio.sample_rate == model_sample_rate, f"Audio needs to be {model_sample_rate}Hz, provided {audio.sample_rate}Hz for {x}"
        assert audio.dim() == 2, f"Audio needs to be 2D tensor, provided {audio.dim()}D for {x}"

        self.audio_q.put(audio)

    def read_audio_files(self, x: List[str], num_threads: int):

        with ThreadPoolExecutor(num_threads) as executor:
            futures = []
            for audio_file in x:
                futures.append(
                    executor.submit(partial(self.preprocessing, audio_file, self.model.sample_rate))
                )
            concurrent.futures.wait(futures)

        with self.read_complete.get_lock():
            self.read_complete.value = True

    def prepare_batch(self, batch_size: int):
        local_batch = torch.zeros(batch_size, 1, self.segment_length, device=self.device)
        local_batch_idx = 0

        while self.audio_q.qsize() > 0 or not self.read_complete.value:
            audio = self.audio_q.get()

            if audio is None:
                continue
            _, length = audio.shape

            for i in range(0, length, self.stride):
                # We have reached the end of the audio and we need to pad this segment
                if i + self.segment_length > length:
                    local_batch[local_batch_idx, 0, :length-i] = audio[:, i:]
                    local_batch[local_batch_idx, 0, length-i:] = self.pad_token
                    self.processed_q.put(local_batch)
                    break

                local_batch[local_batch_idx, 0, :] = audio[:, i:i+self.segment_length]
                local_batch_idx += 1

                if local_batch_idx == batch_size:
                    self.processed_q.put(local_batch)
                    local_batch_idx = 0

        self.processed_q.put(None)

    def encode(self) -> torch.Tensor:

        while self.processed_q.qsize() > 0:
            local_batch = self.processed_q.get()
            if local_batch is None:
                break

            with torch.no_grad():
                emb = self.model.encoder(local_batch)
                codes = self.model.quantizer.encode(emb, self.model.frame_rate, self.model.get_bandwidth())
                # codes is [B, K, T], with T frames, K nb of codebooks.
                codes = codes.transpose(0, 1)
                yield codes

    def __call__(self, x: List[str], batch_size: int, num_threads: int = 1) -> torch.Tensor:
        """
        Implements forward pass of the encodec model

        The input x is a list of audio files and the output is a list of list of tensors
        representing the encoded audio files
        """
        read_process = mp.Process(target=partial(self.read_audio_files, x, num_threads))
        chunk_process = mp.Process(target=partial(self.prepare_batch, batch_size))

        read_process.start()
        chunk_process.start()

        read_process.join()
        chunk_process.join()

        return self.encode()
