import torch
import joblib
from pathlib import Path
from encodec import EncodecModel
from transformers import HubertModel, Wav2Vec2BertModel
from transformers.models.wav2vec2_bert.modeling_wav2vec2_bert import Wav2Vec2BertSelfAttention
from vector_quantize_pytorch import VectorQuantize

from .utils import read_audio, load_vq_weights
from .configs import HubertEncoderConfig, AudioConfig, AcousticEncoderConfig, Wav2VecBertConfig
from .logger import get_logger
from .processors import Wav2VecBertProcessor

from .modeling_wav2vec2_bert import forward as sdpa_forward
Wav2Vec2BertSelfAttention.forward = sdpa_forward

logger = get_logger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
torch.set_float32_matmul_precision("high") # set matmul precision to use either bfloat16 or tf32
# torch.backends.cudnn.benchmark = True  # Selects the best conv algo


def hubert_processor(audio, processor):

    return processor(
        audio,
        sampling_rate=HubertEncoderConfig.model_sample_rate,
        return_tensors='pt'
    ).input_values[0]


class AcousticEncoder(torch.nn.Module):
    def __init__(
            self,
            config: 'AcousticEncoderConfig' = AcousticEncoderConfig(),
            device: str = 'cpu',
        ):

        super().__init__()

        self.model = EncodecModel.encodec_model_24khz()
        self.model.to(device)
        self.model.set_target_bandwidth(config.bandwidth)

        self.model.eval()

    def forward(self, input_batch: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.no_grad():

                emb = self.model.encoder(input_batch.unsqueeze(1))

                codes = self.model.quantizer.encode(
                    emb, self.model.frame_rate, self.model.bandwidth
                )

                codes = codes.transpose(0, 1).to(dtype=torch.int16)  # [B, K, T]
                logger.info(f'Embedding shape: {emb.shape}, Codes shape: {codes.shape}')

                return codes.detach()


class HubertEncoder(torch.nn.Module):
    def __init__(
        self,
        config: HubertEncoderConfig=HubertEncoderConfig(),
        device: str = 'cpu',
        quantize: bool = True,
    ):
        super().__init__()

        self.quantize = quantize
        self.output_layer = config.output_layer

        self.model = HubertModel.from_pretrained(config.model_id).to(device)
        self.model.eval()

        self.layer_norm = torch.nn.LayerNorm(
            normalized_shape=768,
            elementwise_affine=False,
            bias=False,
            device=device,
        )
        self.layer_norm.eval()

        if self.quantize:
            km = joblib.load(config.quantizer_path)
            self.C = torch.from_numpy(km.cluster_centers_).to(device)

    def __call__(self, input_batch: torch.Tensor, attention_mask: torch.Tensor):
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                logger.info(f"Waveform size: {input_batch.shape}, Attention mask size: {attention_mask.shape}")

                embeddings = self.model.forward(input_batch, attention_mask=attention_mask, output_hidden_states=True).hidden_states # N, B, T, D

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


class Wav2VecBertEncoder(torch.nn.Module):
    def __init__(
        self,
        config: 'Wav2VecBertConfig' = Wav2VecBertConfig(),
        device: str = 'cpu',
        quantize: bool = False
    ):

        super().__init__()

        # Defaults from huggingface
        self.processor = Wav2VecBertProcessor(
            feature_size=80,
            num_mel_bins=80,
            sampling_rate=16000,
            stride=2,
            padding_value=1
        )

        self.output_layer = config.output_layer

        self.model = Wav2Vec2BertModel.from_pretrained(config.model_id)
        self.quantize = quantize

        self.model.to(device)
        self.model.eval()

        self.layer_norm = torch.nn.LayerNorm(
            normalized_shape=1024,
            elementwise_affine=False,
            bias=False,
            device=device,
        )
        self.layer_norm.eval()

        if self.quantize:
            self.vq = VectorQuantize(
                dim=1024,
                codebook_size=2048,
                decay=0.8,
                commitment_weight=1
            )
            self.vq.to(device)  # type: ignore
            self.vq.eval()

            model_weights = torch.load(config.quantizer_path, map_location=device) # type: ignore

            self.vq = load_vq_weights(
                model_weights=model_weights,
                model=self.vq,
            )

    def forward(self, input_batch: torch.Tensor, mask: torch.Tensor, pad_to_multiple_of: int = 2):
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                proc_out = self.processor(input_batch, mask, pad_to_multiple_of)
                input_features, attention_mask = proc_out['input_features'], proc_out['attention_mask']

                logger.info(f"Batch size: {input_batch.shape}, Mask size: {mask.shape}")
                logger.info(f"Processed size: {input_features.shape}, Mask size: {attention_mask.shape}")

                embeddings = self.model(input_features, attention_mask=attention_mask, output_hidden_states=True).hidden_states # (N, B, T, D)

                if self.quantize:
                    embeddings = embeddings[self.output_layer]  # B, T, D
                    embeddings = self.layer_norm(embeddings)

                    logger.info(f'Embeddings size: {embeddings.shape}, dtype: {embeddings.dtype}')

                    _, clusters, _ = self.vq(embeddings) # (B, T, 1)
                    clusters = clusters.unsqueeze(-1).transpose(1, 2).to(dtype=torch.int16).detach()  # B, 1, T
                    logger.info(f'Clusters size: {clusters.shape}')

                    return clusters

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

    from .configs import AcousticEncoderConfig
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
        voice_encoder = AcousticEncoder(device=device)

        start_time = time()

        for p in tqdm(audio_file_paths):
            a = read_audio(Path(p).expanduser(), AcousticEncoderConfig.model_sample_rate)
            audio_config = AudioConfig(file_name=p, length_seconds=a.shape[-1], length_samples=a.shape[-1] * 24_000, model_token_rate=50)

            encoded = voice_encoder(a.to(device), torch.ones_like(a, device=device))
            save_audio_tokens(encoded, audio_config, args.outdir)

        print(f'Encodec encoding took {time() - start_time:.2f}s')

    elif args.tokenizer == 'hubert':
        print(f'Encoding using hubert')

        from transformers import Wav2Vec2FeatureExtractor
        processor = Wav2Vec2FeatureExtractor.from_pretrained(HubertEncoderConfig.model_id)
        hubert_encoder = HubertEncoder(
            quantize=True,
            device=device
        )

        start_time = time()

        for p in tqdm(audio_file_paths):
            a = read_audio(Path(p).expanduser(), 16_000)
            a = hubert_processor(a, processor)
            am = torch.ones_like(a, device=device)

            audio_config = AudioConfig(file_name=p, length_seconds=a.shape[-1], length_samples=a.shape[-1] * 16_000, model_token_rate=50)

            encoded = hubert_encoder(a.to(device), am)
            save_audio_tokens(encoded.squeeze(0), audio_config, args.outdir)

        print(f'Hubert encoding took {time() - start_time:.2f}s')

    elif args.tokenizer == 'w2vbert2':
        print(f'Encoding using wav2vec')

        wav2vec_encoder = Wav2VecBertEncoder(
            quantize=True,
            device=device
        )

        start_time = time()

        for p in tqdm(audio_file_paths):
            a = read_audio(Path(p).expanduser(), 16_000)
            am = torch.ones_like(a, device=device)

            audio_config = AudioConfig(
                file_name=p,
                length_seconds=a.shape[-1]/16_000,
                length_samples=a.shape[-1],
                model_token_rate=50
            )

            encoded = wav2vec_encoder(a.to(device), am.to(device), pad_to_multiple_of=10*50)
            save_audio_tokens(encoded.squeeze(0), audio_config, args.outdir)

        print(f'Wav2Vec encoding took {time() - start_time:.2f}s')
