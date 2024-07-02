import torch
import tiktoken
import joblib
import numpy as np

from typing import List, Tuple
from encodec import EncodecModel
from huggingface_hub import hf_hub_download
from transformers import HubertModel, Wav2Vec2BertModel, AutoFeatureExtractor

from .utils import read_audio, preprocess_audio
from .configs import HubertEncoderConfig, AudioConfig, VoiceEncoderConfig, Wav2VecBertConfig
from .logger import logger

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
    def __init__(
            self,
            bandwidth: float,
            single_segment_duration: int,
            batch_size: int = 100,
            device: str = 'cpu'
        ):

        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(bandwidth)

        self.device = torch.device(device)
        self.segment_length = self.model.sample_rate * single_segment_duration

        self.model.eval()

        if device != 'cpu':
            self.model = self.model.to(device)

            torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
            torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
            torch.set_float32_matmul_precision("high") # set matmul precision to use either bfloat16 or tf32
            # torch.backends.cudnn.benchmark = True  # Selects the best conv algo

            self.model = torch.compile(self.model, mode="reduce-overhead")

            # Warmup the model
            input = torch.randn(1, 1, self.segment_length, device=device)
            for _ in range(5):
                self.model(input)

    def __call__(self, input_batch: torch.Tensor, attention_mask: torch.Tensor):
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.no_grad():

                emb = self.model.encoder(input_batch.unsqueeze(1))

                codes = self.model.quantizer.encode(
                    emb, self.model.frame_rate, self.model.bandwidth
                )

                # TODO: Add cutoff support
                codes = codes.transpose(0, 1).to(dtype=torch.int16)  # [B, K, T]
                logger.info(f'Embedding shape: {emb.shape}, Codes shape: {codes.shape}')

                return codes.detach()


class HubertEncoder:
    def __init__(
        self,
        config: HubertEncoderConfig=HubertEncoderConfig(),
        device: str = 'cpu'
    ):

        model_id = config.model_id
        self.batch_size = config.batch_size
        self.segment_length = config.model_sample_rate * config.single_segment_duration
        self.device = torch.device(device)
        self.config = config

        self.pad_token = 0
        self.output_layer = 11

        self.model = HubertModel.from_pretrained(model_id)#,attn_implementation="flash_attention_2", torch_dtype=torch.float16)

        self.model.eval()

        kmeans_path = hf_hub_download(repo_id=model_id, filename='mhubert_base_vp_en_es_fr_it3_L11_km1000.bin')
        self.km = joblib.load(kmeans_path)
        self.C_np = self.km.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0)

        self.C = torch.from_numpy(self.C_np).t().to(self.device)
        self.Cnorm = torch.from_numpy(self.Cnorm_np).to(self.device)

        del(self.C_np)
        del(self.Cnorm_np)

        if device != 'cpu':
            self.model.to(self.device)

            torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
            torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
            torch.set_float32_matmul_precision("high") # set matmul precision to use either bfloat16 or tf32
            # torch.backends.cudnn.benchmark = True  # Selects the best conv algo

            self.model = torch.compile(self.model, mode="reduce-overhead")

            # warmup the model
            input = torch.randn((self.batch_size, 16000), device=self.device)#, dtype=torch.float16)
            am = torch.ones((self.batch_size, 16000), device=self.device)#, dtype=torch.float16
            for _ in range(5):
                _ = self.model(input, attention_mask=am, output_hidden_states=True).hidden_states

    def __call__(self, input_batch: torch.Tensor, attention_mask: torch.Tensor):
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                logger.info(f"Waveform size: {input_batch.shape}, Attention mask size: {attention_mask.shape}")

                embeddings = self.model.forward(input_batch, attention_mask=attention_mask, output_hidden_states=True).hidden_states[self.output_layer] # B, T, D

                logger.info(f'Embeddings size: {embeddings.shape}, C size: {self.C.shape}, dtype: {embeddings.dtype}')

                # Compute L2 norm
                distances = torch.cdist(embeddings, self.C)  # (B, T, K)
                min_dist = torch.argmin(distances, dim=-1, keepdim=True)  # (B, T, 1)

                min_dist = min_dist.transpose(1, 2).to(dtype=torch.int16).detach()  # B, 1, T
                logger.info(f'Min dist size: {min_dist.shape}')

                return min_dist


class Wav2VecBertEncoder:
    def __init__(
        self,
        config: Wav2VecBertConfig,
        device: str = 'cpu'
    ):

        model_id = config.model_id
        self.segment_length = config.model_sample_rate * config.single_segment_duration
        self.device = torch.device(device)

        self.output_layer = config.output_layer

        self.processor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = Wav2Vec2BertModel.from_pretrained(model_id).to(self.device)

        self.model.eval()

        logger.info(f'Ouput layer: {self.output_layer}')

        # K Means model loading
        """
        kmeans_path = hf_hub_download(repo_id=model_id, filename='mhubert_base_vp_en_es_fr_it3_L11_km1000.bin')
        self.km = joblib.load(kmeans_path)
        self.C_np = self.km.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0)

        self.C = torch.from_numpy(self.C_np).t().to(self.device)
        self.Cnorm = torch.from_numpy(self.Cnorm_np).to(self.device)

        del(self.C_np)
        del(self.Cnorm_np)
        """

        if device != 'cpu':

            torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
            torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
            torch.set_float32_matmul_precision("high") # set matmul precision to use either bfloat16 or tf32
            # torch.backends.cudnn.benchmark = True  # Selects the best conv algo

            self.model = torch.compile(self.model, mode="reduce-overhead")

            # Warmup the model, model expects dimension length to be 160
            input = torch.randn((1, 64, 160), device=self.device)
            am = torch.ones((1, 64), device=self.device)

            for _ in range(5):
                _ = self.model(input, attention_mask=am, output_hidden_states=True)

    def __call__(self, input_batch: List[torch.Tensor], attention_mask: List[torch.Tensor]):
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                logger.info(f"Batch size: {len(input_batch)}, Attention mask size: {len(attention_mask)}")

                processed = self.processor(
                    input_batch,
                    sampling_rate=16_000,
                    return_attention_masks=True,
                    return_tensors='pt'
                )

                logger.info(f'Processed size: {processed.input_features.shape}, {processed.attention_mask.shape}')

                input_features = processed.input_features.to(self.device)
                attention_mask = processed.attention_mask.to(self.device)

                embeddings = self.model.forward(input_features, attention_mask=attention_mask, output_hidden_states=True).hidden_states[self.output_layer] # B, T, D

                logger.info(f'Embeddings size: {embeddings.shape}, dtype: {embeddings.dtype}')

                return embeddings

if __name__ == '__main__':
    pass

    """
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
    """
