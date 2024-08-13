import os
import time
import torch
from tqdm import tqdm
from functools import partial
from pathlib import Path
from torch.utils.data import DataLoader

from .utils import save_audio_tokens, find_files, set_process_affinity
from .datasets import AudioBatchDataset, collate_fn
from .logger import get_logger
from .configs import AUDIO_EXTS, TAR_EXTS, ZIP_EXTS

logger = get_logger(__name__)

@torch.inference_mode()
def encode(voice_encoder, dataset, batch_size, outdir):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=12,
        prefetch_factor=4,
        pin_memory=True
    )

    start_time = time.time()

    for idx, (input_ids, attention_masks, file_pointers) in tqdm(enumerate(dataloader)):
        logger.info(f'Processing batch: {idx}')

        input_ids = input_ids.to(DEVICE)
        attention_masks = attention_masks.to(DEVICE)
        encoded_audio = voice_encoder(input_ids, attention_masks)

        logger.info(f"Processed batch: {idx}")

        for jdx, (tokens_batch, file_pointer) in enumerate(zip(encoded_audio, file_pointers)):
            logger.debug(f"Submitted saving for iteration {jdx}, batch: {idx}")
            save_audio_tokens(tokens_batch, file_pointer, outdir)


    print(f"Fixed batching encoding time: {time.time() - start_time:.2f}s")


if __name__ == '__main__':
    """
    python -m src.tokenize_audio \
        --tokenizer encodec \
        --indir /path/to/audio \
        --outdir /path/to/output \
        --batch_size 16 \
        --device cuda:0

    python -m src.tokenize_audio \
        --tokenizer hubert \
        --indir /path/to/audio \
        --outdir /path/to/output \
        --batch_size 16 \
        --device cuda:0
    """
    import argparse
    import random

    set_process_affinity(os.getpid(), list(range(0, 20)))

    parser = argparse.ArgumentParser(description='Encode audio files.')
    parser.add_argument('--tokenizer', choices=['encodec', 'hubert', 'w2vbert2', 'whisper'], type=str, required=True, help='Encoder to run.')
    parser.add_argument('--indir', type=str, required=False, help='Input directory or filename for audio files.')
    parser.add_argument('--hf_dataset', type=str, required=False, help='Name of the huggingface dataset.')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory for encoded audio.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for encoding.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for encoding.')

    global DEVICE

    args = parser.parse_args()
    DEVICE = args.device

    ALL_EXTS = AUDIO_EXTS + TAR_EXTS + ZIP_EXTS
    files = find_files(args.indir, ALL_EXTS)
    random.shuffle(files)

    logger.info(f'Found {len(files)} audio files in the dataset.')
    single_segment_duration = 10

    if args.tokenizer == 'encodec':
        from .encoder import AcousticEncoder
        from .configs import AcousticEncoderConfig

        encoder = AcousticEncoder(device=DEVICE)

        dataset = AudioBatchDataset(
            audio_files=files,
            sample_rate=AcousticEncoderConfig.model_sample_rate,
            chunk_size=single_segment_duration,
            model_token_rate=AcousticEncoderConfig.model_token_rate,
            pad_token=AcousticEncoderConfig.pad_token
        )

    elif args.tokenizer == 'hubert':
        from transformers import Wav2Vec2FeatureExtractor
        from .encoder import HubertEncoder, hubert_processor
        from .configs import HubertEncoderConfig

        encoder = HubertEncoder(device=DEVICE) # type: ignore

        processor = Wav2Vec2FeatureExtractor.from_pretrained(HubertEncoderConfig.model_id)

        tranform_func = partial(hubert_processor, processor=processor)

        dataset = AudioBatchDataset(
            audio_files=files,
            sample_rate=HubertEncoderConfig.model_sample_rate,
            chunk_size=single_segment_duration,
            transform=tranform_func,
            model_token_rate=HubertEncoderConfig.model_token_rate,
            pad_token=HubertEncoderConfig.pad_token
        )

    elif args.tokenizer == 'w2vbert2':
        from .encoder import Wav2VecBertEncoder
        from .configs import Wav2VecBertConfig

        encoder = Wav2VecBertEncoder( # type: ignore
            quantize=True,
            device=DEVICE
        )

        dataset = AudioBatchDataset(
            audio_files=files,
            sample_rate=Wav2VecBertConfig.model_sample_rate,
            chunk_size=single_segment_duration,
            model_token_rate=Wav2VecBertConfig.model_token_rate,
            pad_token=Wav2VecBertConfig.pad_token
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # TODO: Add compile support
    print(f'Model is not compiled')

    encode(
        voice_encoder=encoder,
        dataset=dataset,
        batch_size=args.batch_size,
        outdir=outdir
    )
