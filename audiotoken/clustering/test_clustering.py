import pdb
import torch
import joblib
import numpy as np
from tqdm import tqdm
from typing import Tuple
from pathlib import Path
import torch.nn.functional as F
from time import time

import matplotlib.pyplot as plt
from transformers import Wav2Vec2FeatureExtractor, AutoFeatureExtractor, WhisperFeatureExtractor

from ..utils import read_audio, find_files
from ..encoder import Wav2VecBertEncoder, HubertEncoder, WhisperEncoder, w2vbert2_processor, hubert_processor, whisper_processor
from ..configs import Wav2VecBertConfig, HubertEncoderConfig, WhisperEncoderConfig

EMBEDDING_DIM = 1024

def get_parser():
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Test the k-means cluster and compare the cluster distance of audio tokens vs random tokens."
    )

    # Features arguments
    parser.add_argument(
        '--tokenizer',
        choices=['hubert', 'w2vbert2', 'whisper'],
        type=str,
        required=True,
        help='Encoder to run.'
    )
    parser.add_argument(
        '--indir',
        type=str,
        required=False,
        help='Input directory or filename for audio files.'
    )
    parser.add_argument(
        "--kmeans",
        type=str,
        required=False,
        help="Path to the kmeans model",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Path to write the images",
    )
    parser.add_argument(
        "--layer",
        type=int,
        help="The layer of the pretrained model to extract features from",
        default=-1,
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="Number of samples for testing",
        default=1000,
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to use for encoding.'
    )

    return parser


def get_dist(embeddings: torch.Tensor, centroids: torch.Tensor) -> Tuple:
    """
    Compute the distance between embeddings and centroids

    Args:
        embeddings (torch.Tensor): B, T, D
        centroids (torch.Tensor): K, D

    Returns:
        Tuple: (Value, Indices): B, T
    """

    distances = torch.cdist(embeddings, centroids)
    return torch.min(distances, dim=-1)


def main(args):
    audio_files = find_files(args.indir, ('.flac', '.wav'))
    print(f'Found {len(audio_files)} audio files')

    layer = args.layer
    samples = args.samples
    kmeans_path = Path(args.kmeans).expanduser() if args.kmeans else None

    assert len(audio_files) > samples, f'Number of samples {samples} needs to be less than number of audio files {len(audio_files)}'

    if args.tokenizer == 'hubert':
        # kmeans_path = HubertEncoderConfig.quantizer_path
        print(f'Loading Hubert model and kmeans model from {kmeans_path}')

        kmeans = joblib.load(kmeans_path)
        processor = Wav2Vec2FeatureExtractor.from_pretrained(HubertEncoderConfig.model_id)
        encoder = HubertEncoder(quantize=False, compile=False, device=args.device)

    elif args.tokenizer == 'w2vbert2':
        print(f'Loading Wav2VecBert2 and kmeans model from {kmeans_path}')

        kmeans = joblib.load(kmeans_path)
        processor = AutoFeatureExtractor.from_pretrained(Wav2VecBertConfig.model_id)
        encoder = Wav2VecBertEncoder(
            config=Wav2VecBertConfig(),
            quantize=False,
            compile=False,
            device=args.device
        )

    elif args.tokenizer == 'whisper':
        print(f'Loading Whisper and kmeans model from {kmeans_path}')

        kmeans = joblib.load(kmeans_path)
        processor = WhisperFeatureExtractor.from_pretrained(WhisperEncoderConfig.model_id)
        encoder = WhisperEncoder(
            config=WhisperEncoderConfig(),
            quantize=False,
            device=args.device,
            compile=False
        )

    try:
        centroids = torch.from_numpy(kmeans.cluster_centers_)
    except AttributeError:
        centroids = torch.from_numpy(kmeans)

    audio_files = np.random.choice(audio_files, samples, replace=False)

    audio_distances = []
    audio_tokens = []
    embeddings = []

    layer_norm = torch.nn.LayerNorm(
        normalized_shape=EMBEDDING_DIM,
        elementwise_affine=False,
        bias=False,
        device=args.device,
    )

    print(f'Computing embeddings')

    for a in tqdm(audio_files, total=samples):
        audio = read_audio(a, 16_000)
        audio = audio[0, :160_000]

        if args.tokenizer == 'hubert':
            ii = hubert_processor(audio, processor).unsqueeze(0)
            am = torch.ones_like(ii)
            ii = F.pad(ii, (0, 160_000-ii.shape[1]), value=0)
            am = F.pad(am, (0, 160_000-am.shape[1]), value=0)

            out = encoder(ii.to(args.device), am.to(args.device))

        elif args.tokenizer == 'w2vbert2':
            ii, am = w2vbert2_processor(audio, processor)
            ii = F.pad(ii, (0, 0, 500-ii.shape[1], 0, 0, 0), value=0)
            am = F.pad(am, (500-am.shape[1], 0), value=0)
            out = encoder(ii.to(args.device), am.to(args.device))

        elif args.tokenizer == 'whisper':
            ii, am = whisper_processor(audio, processor)
            out = encoder(ii.to(args.device), am.to(args.device))

        out = layer_norm(out[layer]).cpu()
        d = get_dist(out, centroids)

        embeddings.append(out.numpy())
        audio_distances.extend(d.values)
        audio_tokens.extend(d.indices)

    seq_len, dim = embeddings[0].shape[1:]
    embeddings = torch.from_numpy(np.array(embeddings)).reshape(samples*seq_len, dim)
    audio_distances = np.array(audio_distances).reshape(-1, 1)
    audio_tokens = np.array(audio_tokens).reshape(-1, 1)

    print(f'Shape of embeddings: {embeddings.shape} and audio_distances: {audio_distances.shape} and audio_tokens: {audio_tokens.shape}')

    norms = torch.linalg.vector_norm(embeddings, dim=-1)

    random_embeddings = []
    random_distances = []
    random_tokens = []

    print(f'Generating random embeddings')

    for norm in tqdm(norms):
        random_vec = torch.randn(1, dim)
        random_vec = random_vec / torch.norm(random_vec)
        random_vec = random_vec * norm
        random_embeddings.append(random_vec)

        d = get_dist(random_vec, centroids)

        random_distances.append(d.values)
        random_tokens.append(d.indices)

    random_distances = np.array(random_distances)
    random_tokens = np.array(random_tokens)

    print(f'Shape of random_embeddings: {len(random_embeddings)} and random_distances: {random_distances.shape} and random_tokens: {random_tokens.shape}')

    # Plot the distances of audio tokens from their centroids vs random tokens
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].hist(audio_distances, bins=100, alpha=0.75, label='Audio Tokens')
    ax[0].hist(random_distances, bins=100, alpha=0.5, label='Random Tokens')
    ax[0].set_title('Histogram of Distances')
    ax[0].set_xlabel('Distance')
    ax[0].set_ylabel('Frequency')
    ax[0].legend()

    # Plot the distribution of tokens across the centroids
    ax[1].hist(audio_tokens, bins=100, alpha=0.75, label='Audio Tokens')
    ax[1].hist(random_tokens, bins=100, alpha=0.5, label='Random Tokens')
    ax[1].set_title('Histogram of Tokens')
    ax[1].set_xlabel('Token')
    ax[1].set_ylabel('Frequency')
    ax[1].legend()

    # Save the plots
    if args.outdir:
        outdir = Path(args.outdir).expanduser()
        outdir.mkdir(parents=True, exist_ok=True)

        plt.savefig(outdir / f'kmeans_{args.tokenizer}_{args.layer}_{int(time())}.png')
    else:
        plt.show()


if __name__ == '__main__':
    """
    python -m src.clustering.test_clustering --tokenizer hubert --indir ./data/test-clean/LibriSpeech/test-clean/121/121726 --layer -1 --samples 100
    """
    parser = get_parser()
    args = parser.parse_args()
    main(args)
