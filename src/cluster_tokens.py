# From: https://github.com/facebookresearch/fairseq/blob/main/examples/textless_nlp/gslm/speech2unit/clustering/cluster_kmeans.py

import os
import time
import joblib
import argparse
import numpy as np
from tqdm import tqdm
import psutil

from torch.utils.data import DataLoader
from sklearn.cluster import MiniBatchKMeans

from .logger import logger
from .configs import KMeansClusterConfig, Wav2VecBertConfig
from .datasets import AudioBatchDataset, batch_generator
from .utils import get_dataset_files, set_process_affinity
from .encoder import Wav2VecBertEncoder

def collate_fn(batch):
    segments, attention_masks, file_names = zip(*batch)
    return [s for s in segments], [a for a in attention_masks], file_names


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
        '--device',
        type=str,
        default='cpu',
        help='Device to use for encoding.'
    )

    return parser


def get_kmeans_batch(dataset, encoder, device, max_size=1000):
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True
    )
    dataloader = batch_generator(dataloader)

    features_batch = []
    batch_size = 0

    start_time = time.time()

    for idx, (input_ids, attention_masks, _) in enumerate(dataloader):
        logger.info(f'Processing batch: {idx}')

        input_ids = input_ids
        attention_masks = attention_masks

        encoded_audio = encoder(input_ids, attention_masks) # B, T, D
        B, T, D = encoded_audio.shape
        encoded_audio = encoded_audio.cpu().numpy().reshape(B*T, D)  # B*T, D
        features_batch.append(encoded_audio)
        batch_size += B*T

        if batch_size >= max_size:
            logger.info(f"Yielding batch of size {batch_size}")
            logger.info(f"Batch processing took {time.time() - start_time:.2f}s")
            yield np.concatenate(features_batch, axis=0), batch_size
            start_time = time.time()
            batch_size = 0

    if batch_size > 0:
        logger.info(f"Yielding batch of size {batch_size}")
        logger.info(f"Batch processing took {time.time() - start_time:.2f}s")
        yield np.concatenate(features_batch, axis=0), batch_size


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

    logger.info(f"Process running on core: {psutil.Process().cpu_affinity()}")

    # Get list of files based on either local directory or HF dataset
    files = get_dataset_files(args.indir, args.hf_dataset)

    out_kmeans_model_path = args.outdir
    os.makedirs(out_kmeans_model_path, exist_ok=True)

    # Create the dataset and the dataloader
    wav2vecbert_config:Wav2VecBertConfig = Wav2VecBertConfig(output_layer=args.layer)

    dataset = AudioBatchDataset(
        files,
        sample_rate=Wav2VecBertConfig.model_sample_rate,
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

    # Iterate and train the k-means model batch by batch
    for idx, (kmeans_batch, batch_size) in tqdm(
        enumerate(get_kmeans_batch(
            dataset=dataset,
            encoder=encoder,
            device=args.device,
            max_size=KMeansClusterConfig.batch_size # This data will be stored in the memory
        ))
    ):
        quantizer = train_kmeans(kmeans_model, kmeans_batch)
        total_batches += batch_size

        if idx % 10 == 0:
            ckpt_name = f"kmeans__L{args.layer}_C{args.num_clusters}_ckpt{idx}.pkl"
            save_path = os.path.join(args.outdir, ckpt_name)
            logger.info(f'Saving k-means model to {save_path}')

            with open(save_path, "wb+") as f:
                joblib.dump(quantizer, f)

if __name__ == "__main__":
    """
    python -m src.cluster_tokens --indir data/test-clean/ --outdir data/kmeans --num_cluster 1024 --layer -1 --device cuda
    """

    set_process_affinity(os.getpid(), [p for p in range(12)])

    parser = get_parser()
    args = parser.parse_args()

    logger.info(args)

    main(args)
