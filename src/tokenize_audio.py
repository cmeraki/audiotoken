import os
import time
import torch
from tqdm import tqdm
from functools import partial
from pathlib import Path
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from transformers import Wav2Vec2FeatureExtractor

from .encoder import VoiceEncoder, HubertEncoder
from .configs import VoiceEncoderConfig, HubertEncoderConfig
from .utils import find_audio_files, save_audio_tokens, preprocess_audio
from .datasets import AudioBatchDataset, GigaSpeechDataset
from .logger import logger


def collate_fn(batch):
        segments, attention_masks, file_names = zip(*batch)
        return torch.stack(segments), torch.stack(attention_masks), file_names

def batch_generator(dataloader):
    for batch in dataloader:
        yield batch

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
    dataloader = batch_generator(dataloader)

    start_time = time.time()

    for idx, (input_ids, attention_masks, file_pointers) in tqdm(enumerate(dataloader)):
        logger.info(f'Processing batch: {idx}')

        input_ids = input_ids.to(DEVICE)
        attention_masks = attention_masks.to(DEVICE)
        encoded_audio = voice_encoder(input_ids, attention_masks)

        logger.info(f"Processed batch: {idx}")

        for jdx, (tokens_batch, file_pointer) in enumerate(zip(encoded_audio, file_pointers)):
            logger.info(f"Submitted saving for iteration {jdx}, batch: {idx}")
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

    parser = argparse.ArgumentParser(description='Encode audio files.')
    parser.add_argument('--tokenizer', choices=['encodec', 'hubert'], type=str, required=True, help='Encoder to run.')
    parser.add_argument('--indir', type=str, required=True, help='Input directory or filename for audio files.')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory for encoded audio.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for encoding.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for encoding.')

    global DEVICE

    args = parser.parse_args()
    DEVICE = args.device

    if os.path.isdir(args.indir):
        files = find_audio_files(args.indir)

    else:
        files = [args.indir]

    if args.tokenizer == 'encodec':
        encoder = VoiceEncoder(
            bandwidth=VoiceEncoderConfig.bandwidth,
            single_segment_duration=VoiceEncoderConfig.single_segment_duration,
            device=DEVICE,
        )

        dataset = AudioBatchDataset(
            files,
            sample_rate=VoiceEncoderConfig.model_sample_rate,
            single_segment_duration=VoiceEncoderConfig.single_segment_duration,
            model_token_rate=VoiceEncoderConfig.model_token_rate
        )

        # dataset = GigaSpeechDataset(  # type: ignore
        #     sample_rate=VoiceEncoderConfig.model_sample_rate,
        #     size="xs",
        #     split="train",
        # )

    elif args.tokenizer == 'hubert':
        encoder = HubertEncoder(device=DEVICE) # type: ignore

        processor = Wav2Vec2FeatureExtractor.from_pretrained(HubertEncoderConfig.model_id)

        tranform_func = partial(
            preprocess_audio,
            sample_rate=HubertEncoderConfig.model_sample_rate,
            processor=processor
        )

        dataset = AudioBatchDataset(
            files,
            sample_rate=HubertEncoderConfig.model_sample_rate,
            single_segment_duration=HubertEncoderConfig.single_segment_duration,
            transform=tranform_func,
            model_token_rate=HubertEncoderConfig.model_token_rate
        )

        # dataset = GigaSpeechDataset( # type: ignore
        #     sample_rate=HubertEncoderConfig.audio_sample_rate,
        #     size="xs",
        #     split="train",
        #     transform=tranform_func
        # )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    encode(
        voice_encoder=encoder,
        dataset=dataset,
        batch_size=args.batch_size,
        outdir=outdir
    )
