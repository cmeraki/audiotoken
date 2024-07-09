# From: https://github.com/facebookresearch/fairseq/blob/main/examples/textless_nlp/gslm/speech2unit/clustering/cluster_kmeans.py

import os
import time
import joblib
import argparse
import numpy as np
from tqdm import tqdm
import psutil
from functools import partial
from transformers import AutoFeatureExtractor

from torch.utils.data import DataLoader
from sklearn.cluster import MiniBatchKMeans

from ..logger import logger
from ..configs import KMeansClusterConfig, Wav2VecBertConfig
from ..datasets import AudioBatchDataset, collate_fn
from ..utils import get_dataset_files, set_process_affinity
from ..encoder import Wav2VecBertEncoder, wav2vec2_processor


def get_parser():
    parser = argparse.ArgumentParser(
        description="Learn K-means clustering over acoustic features."
    )

    # Features arguments
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
        "--layer",
        type=int,
        help="The layer of the pretrained model to extract features from",
        default=-1,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for encoder model. This is different from the batch size for k-means training",
        default=16,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs for k-means training",
        default=10,
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to use for encoding.'
    )

    return parser


def get_kmeans_batch(dataset, encoder, epochs, max_size=1000):
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True
    )

    layers = [15, 18, 21]
    accumulated_batch_size = 0
    features_batch = {
        15: [],
        18: [],
        21: []
    }
    start_time = time.time()

    for e in tqdm(range(epochs)):

        logger.info(f"Starting epoch: {e}")

        for idx, (input_ids, attention_masks, _) in tqdm(enumerate(dataloader)):
            logger.info(f'Processing batch: {idx}')

            input_ids = input_ids.to(DEVICE)
            attention_masks = attention_masks.to(DEVICE)

            encoded_audio = encoder(input_ids, attention_masks) # N, B, T, D

            for l in layers:
                single_layer = encoded_audio[l]
                logger.info(f"Layer: {single_layer.shape}")
                B, T, D = single_layer.shape
                single_layer = single_layer.cpu().numpy().reshape(B*T, D)  # B*T, D
                features_batch[l].append(single_layer)

            accumulated_batch_size += B*T

            if accumulated_batch_size >= max_size:
                logger.info(f"Yielding batch of size {accumulated_batch_size}")
                logger.info(f"Batch processing took {time.time() - start_time:.2f}s")
                yield features_batch, accumulated_batch_size
                start_time = time.time()
                accumulated_batch_size = 0
                features_batch = {
                    15: [],
                    18: [],
                    21: []
                }

    if accumulated_batch_size > 0:
        logger.info(f"Yielding batch of size {accumulated_batch_size}")
        logger.info(f"Batch processing took {time.time() - start_time:.2f}s")
        yield features_batch, accumulated_batch_size


def get_kmeans_model(n_clusters: int, config: KMeansClusterConfig):
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        max_iter=config.max_iter,
        batch_size=config.batch_size,
        max_no_improvement=config.max_no_improvement,
        n_init=config.n_init,
        reassignment_ratio=config.reassignment_ratio,
        verbose=1,
        compute_labels=True,
        init_size=None,
    )


def train_kmeans(kmodel: MiniBatchKMeans, features_batch: np.ndarray) -> MiniBatchKMeans:
    logger.info(f'Fitting k-means model with {features_batch.shape[0]} samples')

    start_time = time.time()
    kmodel.partial_fit(features_batch)

    logger.info(f"K-means partial training took {time.time() - start_time:.2f}s")

    inertia = -kmodel.score(features_batch) / len(features_batch)
    print(f"Total inertia: {round(inertia, 2)}\n")

    return kmodel


def main(args):

    global DEVICE
    DEVICE = args.device

    logger.info(f"Process running on core: {psutil.Process().cpu_affinity()}")

    # Get list of files based on either local directory or HF dataset
    files = get_dataset_files(args.indir, args.hf_dataset)

    out_kmeans_model_path = args.outdir
    os.makedirs(out_kmeans_model_path, exist_ok=True)

    # Create the dataset and the dataloader
    wav2vecbert_config:Wav2VecBertConfig = Wav2VecBertConfig(output_layer=args.layer) # type:ignore

    processor = AutoFeatureExtractor.from_pretrained(Wav2VecBertConfig.model_id)
    post_transform_func = partial(wav2vec2_processor, processor=processor)

    dataset = AudioBatchDataset(
        files,
        sample_rate=Wav2VecBertConfig.model_sample_rate,
        post_transform=post_transform_func,
        single_segment_duration=Wav2VecBertConfig.single_segment_duration,
        model_token_rate=Wav2VecBertConfig.model_token_rate,
    )

    # Create the encoder model
    encoder = Wav2VecBertEncoder(
        config=wav2vecbert_config,
        device=args.device
    )

    # Create the k-means model
    kmeans_model = get_kmeans_model(
        n_clusters=args.num_clusters,
        config=KMeansClusterConfig,
    )
    total_batches = 0
    quantizers = {
        15: get_kmeans_model(
            n_clusters=args.num_clusters,
            config=KMeansClusterConfig,
        ),
        18: get_kmeans_model(
            n_clusters=args.num_clusters,
            config=KMeansClusterConfig,
        ),
        21: get_kmeans_model(
            n_clusters=args.num_clusters,
            config=KMeansClusterConfig,
        )
    }

    # Iterate and train the k-means model batch by batch
    for idx, (kmeans_batch, batch_size) in enumerate(
        get_kmeans_batch(
            dataset=dataset,
            encoder=encoder,
            epochs=args.epochs,
            max_size=KMeansClusterConfig.batch_size
        )
    ):
        for k, v in kmeans_batch.items():
            kmeans_batch[k] = np.concatenate(v, axis=0)
            quantizers[k] = train_kmeans(quantizers[k], kmeans_batch[k])

            if idx % 50 == 0:
                ckpt_name = f"kmeans__L{k}_C{args.num_clusters}_ckpt{idx}.pkl"
                save_path = os.path.join(args.outdir, ckpt_name)
                logger.info(f'Saving k-means model to {save_path}')

                with open(save_path, "wb+") as f:
                    joblib.dump(quantizers[k], f)

        total_batches += batch_size

if __name__ == "__main__":
    """
    python -m src.cluster_tokens --indir data/test-clean/ --outdir data/kmeans --num_cluster 1024 --layer -1 --device cuda
    """

    set_process_affinity(os.getpid(), [p for p in range(16)])

    parser = get_parser()
    args = parser.parse_args()

    logger.info(args)

    main(args)
