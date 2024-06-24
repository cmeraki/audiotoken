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
from .datasets import AudioDataset, GigaSpeechDataset

logger.remove()

def collate_fn_batched(batch):
    return [(waveform, audio_config) for waveform, audio_config in batch]

@torch.inference_mode()
def encode(voice_encoder, dataset, batch_size, outdir):
    dataloader_batched = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_batched,
        num_workers=20,
        prefetch_factor=12,
        pin_memory=True
    )

    start_time = time.time()

    with ThreadPoolExecutor() as executor:
        for batch in tqdm(dataloader_batched, total=len(dataloader_batched)):
            encoded_audio = voice_encoder(batch)
            for tokens_batch, file_pointers in encoded_audio:
                for fp in file_pointers:
                     executor.submit(save_audio_tokens, tokens_batch, fp, outdir)

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
            batch_size=VoiceEncoderConfig.batch_size,
            overlap=VoiceEncoderConfig.overlap,
            device=DEVICE,
        )

        # dataset = AudioDataset(
        #     files,
        #     sample_rate=VoiceEncoderConfig.model_sample_rate,
        #     channels=1,
        # )

        dataset = GigaSpeechDataset(  # type: ignore
            sample_rate=VoiceEncoderConfig.model_sample_rate,
            size="xs",
            split="train",
        )

    elif args.tokenizer == 'hubert':
        encoder = HubertEncoder(device=DEVICE) # type: ignore

        processor = Wav2Vec2FeatureExtractor.from_pretrained(HubertEncoderConfig.model_id)

        tranform_func = partial(
            preprocess_audio, sample_rate=HubertEncoderConfig.audio_sample_rate, processor=processor
        )

        # dataset = AudioDataset(
        #     files,
        #     sample_rate=HubertEncoderConfig.audio_sample_rate,
        #     channels=1,
        #     transform=tranform_func
        # )

        dataset = GigaSpeechDataset( # type: ignore
            sample_rate=HubertEncoderConfig.audio_sample_rate,
            size="xs",
            split="train",
            transform=tranform_func
        )


    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    encode(
        voice_encoder=encoder,
        dataset=dataset,
        batch_size=args.batch_size,
        outdir=outdir
    )
