# From: https://github.com/facebookresearch/fairseq/blob/main/examples/textless_nlp/gslm/speech2unit/clustering/cluster_kmeans.py

import os
import time
import joblib
import random
import argparse
import numpy as np
from tqdm import tqdm
import psutil
import torch
from functools import partial

from torch.utils.data import DataLoader
# from sklearn.cluster import MiniBatchKMeans
from vector_quantize_pytorch import VectorQuantize

from ..logger import get_logger
from ..configs import KMeansClusterConfig, TAR_EXTS, AUDIO_EXTS, ZIP_EXTS
from ..datasets import AudioBatchDataset, collate_fn
from ..utils import set_process_affinity, find_files

logger = get_logger('clustering.log')

def get_parser():
    parser = argparse.ArgumentParser(
        description="Learn K-means clustering over acoustic features."
    )

    # Features arguments
    parser.add_argument(
        '--embedder',
        choices=['hubert', 'w2vbert2', 'whisper'],
        type=str,
        required=True,
        help='Embedder to run.'
    )
    parser.add_argument(
        '--indir',
        type=str,
        required=False,
        help='Input directory or filename for audio files.'
    )
    parser.add_argument(
        '--hf_dataset',
        type=str,
        required=False,
        help='Name of the huggingface dataset.'
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Features file path to write to",
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=1024,
        help="Number of clusters to learn",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for encoder model. This is different from the batch size for k-means training",
        default=16,
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        help="Number of steps after which the embeddings are saved",
        default=100,
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to use for encoding.'
    )

    return parser


def get_features_batch(dataset, encoder, max_size=1000):
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=12,
        prefetch_factor=4,
        pin_memory=True
    )

    layer_norm = torch.nn.LayerNorm(
        normalized_shape=EMBEDDING_DIM,
        elementwise_affine=False,
        bias=False,
        device=DEVICE,
    )

    accumulated_batch_size = 0
    features_batch = {}
    for l in LAYERS:
        features_batch[l] = []

    start_time = time.time()

    for idx, (input_ids, attention_masks, _) in enumerate(dataloader):
        logger.info(f'Processing batch: {idx}, with size: {input_ids.shape}, {attention_masks.shape}')

        input_ids = input_ids.to(DEVICE)
        attention_masks = attention_masks.to(DEVICE)

        encoded_audio = encoder(input_ids, attention_masks) # N, B, T, D

        for l in LAYERS:
            single_layer = encoded_audio[l]
            single_layer = layer_norm(single_layer)
            logger.info(f"Layer {l}: {single_layer.shape}")
            B, T, D = single_layer.shape
            single_layer = single_layer.reshape(B*T, D)  # B*T, D
            features_batch[l].append(single_layer)

        accumulated_batch_size += B*T

        if accumulated_batch_size >= max_size:
            logger.info(f"Yielding batch of size {accumulated_batch_size}")
            logger.info(f"Batch processing took {time.time() - start_time:.2f}s")
            yield features_batch, accumulated_batch_size

            start_time = time.time()
            accumulated_batch_size = 0

            for l in LAYERS:
                features_batch[l] = []

    if accumulated_batch_size > 0:
        logger.info(f"Yielding batch of size {accumulated_batch_size}")
        logger.info(f"Batch processing took {time.time() - start_time:.2f}s")
        yield features_batch, accumulated_batch_size


def get_vq_model(n_clusters: int, batch_size: int = 16):
    global EMBEDDING_DIM
    global DEVICE

    vq = VectorQuantize(
        dim=EMBEDDING_DIM,
        codebook_size=n_clusters,
        decay=0.8,
        commitment_weight=1
    )
    vq.to(DEVICE) # type:ignore

    new_state_dict = {}
    old_vq = torch.load('data/vq_w2vbert_mix_2/quantizer__L19_C2048_ckpt9000.pkl', map_location=DEVICE)

    for k, v in old_vq.items():
        new_state_dict[k] = v

    vq.load_state_dict(new_state_dict) # type:ignore

    print('Loaded the checkpoint')

    # vq = torch.compile(vq)
    # vq(torch.randn((batch_size, EMBEDDING_DIM), device=DEVICE))

    return vq


# def get_kmeans_model(n_clusters: int, config: KMeansClusterConfig):
#     return MiniBatchKMeans(
#         n_clusters=n_clusters,
#         max_iter=config.max_iter,
#         batch_size=config.batch_size,
#         max_no_improvement=config.max_no_improvement,
#         n_init=config.n_init,
#         reassignment_ratio=config.reassignment_ratio,
#         verbose=0,
#         compute_labels=True,
#         init_size=None,
#     )


# def train_kmeans(kmodel: MiniBatchKMeans, features_batch: np.ndarray) -> MiniBatchKMeans:
#     logger.info(f'Fitting k-means model with {features_batch.shape[0]} samples')

#     start_time = time.time()
#     kmodel.partial_fit(features_batch)

#     logger.info(f"K-means partial training took {time.time() - start_time:.2f}s")

#     inertia = -kmodel.score(features_batch) / len(features_batch)
#     logger.info(f"Total inertia: {round(inertia, 2)}\n")

#     return kmodel


def main(args):

    global DEVICE
    global LAYERS
    global EMBEDDING_DIM

    DEVICE = args.device

    logger.info(f"Process running on core: {psutil.Process().cpu_affinity()}")

    # Get list of files based on either local directory or HF dataset
    files = find_files(args.indir, AUDIO_EXTS + TAR_EXTS + ZIP_EXTS)
    # files = sorted(files)
    random.shuffle(files)
    files.remove('/home/meraki/projects/indri/data/audio/gigaspeech/podcast_tar/P0091_part1.tar')
    print(f'Found {len(files)} files')

    with open('./logs/processed_mix.txt', 'r') as fp:
         d = fp.read()
         processed_files = []
         for ln in d.split('\n'):
             processed_files.append(ln)

    files = [f for f in files if f not in processed_files]
    print(f'Found: {len(files)} files after excluding {len(processed_files)} files')

    out_kmeans_model_path = args.outdir
    os.makedirs(out_kmeans_model_path, exist_ok=True)

    if args.embedder == 'w2vbert2':
        # Create the dataset and the encoder
        from transformers import AutoFeatureExtractor
        from ..configs import Wav2VecBertConfig
        from ..encoder import Wav2VecBertEncoder

        LAYERS = [19]
        EMBEDDING_DIM = 1024

        processor = AutoFeatureExtractor.from_pretrained(Wav2VecBertConfig.model_id)

        dataset = AudioBatchDataset(
            audio_files=files,
            sample_rate=Wav2VecBertConfig.model_sample_rate,
            single_segment_duration=Wav2VecBertConfig.single_segment_duration,
            model_token_rate=Wav2VecBertConfig.model_token_rate,
            pad_token=Wav2VecBertConfig.pad_token
        )

        # Create the encoder model
        encoder = Wav2VecBertEncoder(
            config=Wav2VecBertConfig(),
            device=args.device,
            batch_size=args.batch_size,
            compile=False
        )

    elif args.embedder == 'hubert':
        from transformers import Wav2Vec2FeatureExtractor
        from ..encoder import HubertEncoder, hubert_processor
        from ..configs import HubertEncoderConfig

        LAYERS = [11]
        EMBEDDING_DIM = 768

        processor = Wav2Vec2FeatureExtractor.from_pretrained(HubertEncoderConfig.model_id)
        tranform_func = partial(hubert_processor, processor=processor)

        dataset = AudioBatchDataset(
            audio_files=files,
            sample_rate=HubertEncoderConfig.model_sample_rate,
            single_segment_duration=HubertEncoderConfig.single_segment_duration,
            transform=tranform_func,
            model_token_rate=HubertEncoderConfig.model_token_rate,
            pad_token=HubertEncoderConfig.pad_token
        )

        encoder = HubertEncoder(
            device=args.device,
            quantize=False,
            batch_size=args.batch_size,
        )

    # Create the quantizer model
    total_batches = 0
    quantizers = {}

    for l in LAYERS:
        logger.info(f'Creating quantizer model for layer: {l}')
        quantizers[l] = get_vq_model(n_clusters=args.num_clusters)

    pbar = tqdm(position=0, leave=True)

    # Iterate and train the k-means model batch by batch
    for idx, (kmeans_batch, batch_size) in enumerate(get_features_batch(
            dataset=dataset,
            encoder=encoder,
            max_size=KMeansClusterConfig.batch_size
        )
    ):
        for layer_num, features in kmeans_batch.items():
            start_time = time.time()
            # kmeans_batch[k] = np.concatenate(v, axis=0)
            # quantizers[k] = train_kmeans(quantizers[k], kmeans_batch[k])

            _, indices, commit_loss = quantizers[layer_num](torch.stack(features))

            logger.info(f"VQ took {time.time() - start_time:.2f}s")

            pbar.set_description(
                f"Commitment loss: {commit_loss.item():.3f} | "
                + f"active %: {indices.unique().numel() / args.num_clusters * 100:.3f}"
            )
            pbar.n += 1
            pbar.refresh()

            if idx % args.save_freq == 0:
                ckpt_name = f"quantizer__L{layer_num}_C{args.num_clusters}_ckpt{idx}.pkl"
                save_path = os.path.join(args.outdir, ckpt_name)
                logger.info(f'Saving quanitzer to {save_path}')

                torch.save(quantizers[layer_num].state_dict(), save_path)

        total_batches += batch_size

if __name__ == "__main__":
    """
    python -m src.cluster_tokens --indir data/test-clean/ --outdir data/kmeans --num_cluster 1024 --device cuda
    """

    set_process_affinity(os.getpid(), [p for p in range(22)])

    parser = get_parser()
    args = parser.parse_args()

    logger.info(args)

    main(args)
