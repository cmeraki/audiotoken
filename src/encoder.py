import torch
import tiktoken
import joblib

from pathlib import Path
from typing import List, Optional
from encodec import EncodecModel
from transformers import HubertModel, Wav2Vec2BertModel, AutoFeatureExtractor, WhisperForAudioClassification, WhisperFeatureExtractor

from .utils import read_audio
from .configs import HubertEncoderConfig, AudioConfig, VoiceEncoderConfig, Wav2VecBertConfig, WhisperEncoderConfig
from .logger import logger


torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
torch.set_float32_matmul_precision("high") # set matmul precision to use either bfloat16 or tf32
# torch.backends.cudnn.benchmark = True  # Selects the best conv algo

def w2vbert2_processor(audio, processor):

    proc = processor(
        audio,
        sampling_rate=Wav2VecBertConfig.model_sample_rate,
        return_attention_masks=True,
        return_tensors='pt'
    )

    return proc.input_features, proc.attention_mask


def hubert_processor(audio, processor):

    return processor(
        audio,
        sampling_rate=HubertEncoderConfig.model_sample_rate,
        return_tensors='pt'
    ).input_values[0]


def whisper_processor(audio, processor):
    proc = processor(
        audio.numpy(),
        sampling_rate=WhisperEncoderConfig.model_sample_rate,
        return_attention_mask=True,
        return_tensors='pt',
        truncation=False,
        padding="longest",
        pad_to_multiple_of=WhisperEncoderConfig.model_sample_rate * WhisperEncoderConfig.single_segment_duration,
    )

    return proc.input_features.transpose(1, 2), proc.attention_mask


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
            device: str = 'cpu',
            compile: bool = True
        ):

        self.model = EncodecModel.encodec_model_24khz()
        self.model.to(device)
        self.model.set_target_bandwidth(bandwidth)

        self.device = torch.device(device)
        self.segment_length = self.model.sample_rate * single_segment_duration

        self.model.eval()

        if compile:
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
        quantize: bool = True,
        device: str = 'cpu',
        compile: bool = True
    ):

        model_id = config.model_id
        self.batch_size = config.batch_size
        self.segment_length = config.model_sample_rate * config.single_segment_duration
        self.device = torch.device(device)
        self.config = config
        self.quantize = quantize

        self.pad_token = 0
        self.output_layer = 11

        self.model = HubertModel.from_pretrained(model_id).to(self.device)

        self.model.eval()

        if self.quantize:
            self.km = joblib.load(config.quantizer_path)
            self.C_np = self.km.cluster_centers_.transpose()
            self.Cnorm_np = (self.C_np ** 2).sum(0)

            self.C = torch.from_numpy(self.C_np).t().to(self.device)
            self.Cnorm = torch.from_numpy(self.Cnorm_np).to(self.device)

            del(self.C_np)
            del(self.Cnorm_np)

        if compile:
            self.model = torch.compile(self.model)

            # warmup the model
            input = torch.randn((self.batch_size, 16000), device=self.device)#, dtype=torch.float16)
            am = torch.ones((self.batch_size, 16000), device=self.device)#, dtype=torch.float16
            for _ in range(5):
                _ = self.model(input, attention_mask=am, output_hidden_states=True).hidden_states

    def __call__(self, input_batch: torch.Tensor, attention_mask: torch.Tensor):
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                logger.info(f"Waveform size: {input_batch.shape}, Attention mask size: {attention_mask.shape}")

                embeddings = self.model.forward(input_batch, attention_mask=attention_mask, output_hidden_states=True).hidden_states # N, B, T, D

                if self.quantize:
                    embeddings = embeddings[self.output_layer]  # B, T, D
                    logger.info(f'Embeddings size: {embeddings.shape}, dtype: {embeddings.dtype}')
                    # Compute L2 norm
                    distances = torch.cdist(embeddings, self.C)  # (B, T, K)
                    min_dist = torch.argmin(distances, dim=-1, keepdim=True)  # (B, T, 1)

                    min_dist = min_dist.transpose(1, 2).to(dtype=torch.int16).detach()  # B, 1, T
                    logger.info(f'Min dist size: {min_dist.shape}')

                    return min_dist

                return embeddings


class Wav2VecBertEncoder:
    def __init__(
        self,
        config: Wav2VecBertConfig,
        quantize: bool = False,
        device: str = 'cpu',
        compile: bool = True
    ):

        model_id = config.model_id
        self.segment_length = config.model_sample_rate * config.single_segment_duration
        self.device = torch.device(device)

        self.output_layer = config.output_layer

        self.model = Wav2Vec2BertModel.from_pretrained(model_id).to(self.device)
        self.quantize = quantize

        self.model.eval()

        self.layer_norm = torch.nn.LayerNorm(
            normalized_shape=1024,
            elementwise_affine=False,
            bias=False,
            device=self.device,
        )

        logger.info(f'Ouput layer: {self.output_layer}')

        if self.quantize:
            kmeans_path = Path(config.quantizer_path) # type: ignore
            km = joblib.load(kmeans_path)

            self.C = torch.from_numpy(km.cluster_centers_).to(self.device)

            del(km)

        if compile:
            self.model = torch.compile(self.model)

            # Warmup the model, model expects dimension length to be 160
            input = torch.randn((1, 64, 160), device=self.device)
            am = torch.ones((1, 64), device=self.device)

            for _ in range(5):
                _ = self.model(input, attention_mask=am, output_hidden_states=True)

            del(input)
            del(am)

    def __call__(self, input_batch: torch.Tensor, attention_mask: torch.Tensor):
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                logger.info(f"Batch size: {input_batch.shape}, Attention mask size: {attention_mask.shape}")

                embeddings = self.model.forward(input_batch, attention_mask=attention_mask, output_hidden_states=True).hidden_states # (N, B, T, D)

                if self.quantize:
                    embeddings = embeddings[self.output_layer]  # B, T, D
                    embeddings = self.layer_norm(embeddings)

                    logger.info(f'Embeddings size: {embeddings.shape}, dtype: {embeddings.dtype}')
                    # Compute L2 norm
                    distances = torch.cdist(embeddings, self.C)  # (B, T, K)
                    min_dist = torch.argmin(distances, dim=-1, keepdim=True)  # (B, T, 1)

                    min_dist = min_dist.transpose(1, 2).to(dtype=torch.int16).detach()  # B, 1, T
                    logger.info(f'Min dist size: {min_dist.shape}')

                    return min_dist

                return embeddings


class WhisperEncoder:
    def __init__(
        self,
        config: WhisperEncoderConfig,
        quantize: bool = False,
        device: str = 'cpu',
        compile: bool = True
    ):

        model_id = config.model_id
        self.segment_length = config.model_sample_rate * config.single_segment_duration
        self.device = torch.device(device)

        self.output_layer = config.output_layer

        self.model = WhisperForAudioClassification.from_pretrained(model_id, attn_implementation="sdpa").to(self.device)
        self.quantize = quantize

        self.model.eval()

        logger.info(f'Ouput layer: {self.output_layer}')

        if self.quantize:
            kmeans_path = Path(config.quantizer_path) # type: ignore
            km = joblib.load(kmeans_path)

            self.C = torch.from_numpy(km.cluster_centers_).to(self.device)

            del(km)

        if compile:
            self.model = torch.compile(self.model)

            # Warmup the model, model expects dimension length to be 160
            input = torch.randn((1, 80, 3000), device=self.device)

            for _ in range(5):
                _ = self.model(input, output_hidden_states=True)

            del(input)

    def __call__(self, input_batch: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                input_batch = input_batch.transpose(1, 2)
                logger.info(f"Batch size: {input_batch.shape}")

                embeddings = self.model.forward(input_batch, output_hidden_states=True).hidden_states # (N, B, T, D)
                logger.info(f'Embeddings size: {len(embeddings)}, dtype: {embeddings[0].dtype}')

                if self.quantize:
                    embeddings = embeddings[self.output_layer]  # B, T, D

                    # Compute L2 norm
                    distances = torch.cdist(embeddings, self.C)  # (B, T, K)
                    min_dist = torch.argmin(distances, dim=-1, keepdim=True)  # (B, T, 1)

                    min_dist = min_dist.transpose(1, 2).to(dtype=torch.int16).detach()  # B, 1, T
                    logger.info(f'Min dist size: {min_dist.shape}')

                    return min_dist

                return embeddings



if __name__ == '__main__':
    """
    python -m src.encoder --tokenizer encodec --indir /path/to/audio/files --outdir /path/to/output/directory
    """
    import os
    from time import time
    from tqdm import tqdm
    from pathlib import Path
    from argparse import ArgumentParser

    from .configs import VoiceEncoderConfig
    from .utils import find_audio_files, save_audio_tokens

    parser = ArgumentParser(description='Encode audio files in indir one by one using the tokenizer specified. Writes tokens to outdir')

    parser.add_argument('--tokenizer', choices=['encodec', 'hubert', 'w2vbert2', 'whisper'], type=str, required=True, help='Encoder to run.')
    parser.add_argument('--indir', type=str, required=True, help='Input filename for audio files.')
    parser.add_argument('--outdir', type=str, required=False, help='Output directory for encoded files.')

    args = parser.parse_args()

    audio_file_paths = find_audio_files(args.indir)

    os.makedirs(args.outdir, exist_ok=True)

    device = 'cuda:0'

    print(f'Found {len(audio_file_paths)} audio files.')

    if args.tokenizer == 'encodec':
        print(f'Encoding using encodec')
        voice_encoder = VoiceEncoder(
            bandwidth=VoiceEncoderConfig.bandwidth,
            single_segment_duration=VoiceEncoderConfig.single_segment_duration,
            device=device,
            compile=False
        )

        start_time = time()

        for p in tqdm(audio_file_paths):
            a = read_audio(Path(p).expanduser(), VoiceEncoderConfig.model_sample_rate)
            audio_config = AudioConfig(file_name=p, length_seconds=a.shape[-1], length_samples=a.shape[-1] * 24_000, length_tokens=50)

            encoded = voice_encoder(a.to(device), torch.ones_like(a, device=device))
            print(encoded.shape)
            save_audio_tokens(encoded, audio_config, args.outdir)

        print(f'Encodec encoding took {time() - start_time:.2f}s')

    elif args.tokenizer == 'hubert':
        print(f'Encoding using hubert')

        from transformers import Wav2Vec2FeatureExtractor
        processor = Wav2Vec2FeatureExtractor.from_pretrained(HubertEncoderConfig.model_id)
        hubert_encoder = HubertEncoder(
            config=HubertEncoderConfig(),
            device=device,
            compile=False
        )

        start_time = time()

        for p in tqdm(audio_file_paths):
            a = read_audio(Path(p).expanduser(), 16_000)
            a = hubert_processor(a, processor)
            am = torch.ones_like(a, device=device)

            audio_config = AudioConfig(file_name=p, length_seconds=a.shape[-1], length_samples=a.shape[-1] * 16_000, length_tokens=50)

            encoded = hubert_encoder(a.to(device), am)
            save_audio_tokens(encoded.squeeze(0), audio_config, args.outdir)

        print(f'Hubert encoding took {time() - start_time:.2f}s')

    elif args.tokenizer == 'w2vbert2':
        print(f'Encoding using wav2vec')

        processor = AutoFeatureExtractor.from_pretrained(Wav2VecBertConfig.model_id)
        wav2vec_encoder = Wav2VecBertEncoder(
            config=Wav2VecBertConfig(),
            quantize=True,
            device=device,
            compile=False
        )

        start_time = time()

        for p in tqdm(audio_file_paths):
            a = read_audio(Path(p).expanduser(), 16_000)
            i, am = w2vbert2_processor(a, processor)

            audio_config = AudioConfig(
                file_name=p,
                length_seconds=a.shape[-1]/16_000,
                length_samples=a.shape[-1],
                length_tokens=50
            )

            encoded = wav2vec_encoder(i.to(device), am.to(device))
            save_audio_tokens(encoded, audio_config, args.outdir)

        print(f'Wav2Vec encoding took {time() - start_time:.2f}s')

    elif args.tokenizer == 'whisper':
        print(f'Encoding using Whisper')

        processor = WhisperFeatureExtractor.from_pretrained(WhisperEncoderConfig.model_id)
        whisper_encoder = WhisperEncoder(
            config=WhisperEncoderConfig(),
            quantize=True,
            device=device,
            compile=False
        )

        start_time = time()

        for p in tqdm(audio_file_paths):
            a = read_audio(Path(p).expanduser(), 16_000)
            i, am = whisper_processor(a, processor)

            audio_config = AudioConfig(
                file_name=p,
                length_seconds=a.shape[-1]/16_000,
                length_samples=a.shape[-1],
                length_tokens=100
            )

            encoded = whisper_encoder(i.to(device), am.to(device))
            save_audio_tokens(encoded.squeeze(0), audio_config, args.outdir)

        print(f'Whisper encoding took {time() - start_time:.2f}s')
