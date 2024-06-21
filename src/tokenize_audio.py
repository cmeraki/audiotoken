import os
import time
import torch
from tqdm import tqdm
from queue import Queue
from pathlib import Path
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor
from loguru import logger

from .encoder import VoiceEncoder, HubertEncoder
from .configs import VoiceEncoderConfig, HubertEncoderConfig
from .utils import find_audio_files, save_audio_tokens
from .datasets import AudioDataset, GigaSpeechDataset

logger.remove()

def collate_fn_batched(batch):
    return batch

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
            audio_q = []
            for waveform, audio_config in batch:
                audio_q.append((waveform, audio_config))

            encoded_audio = voice_encoder(audio_q)
            for tokens_batch, file_pointers in encoded_audio:
                for fp in file_pointers:
                        executor.submit(save_audio_tokens, tokens_batch, fp, outdir)

    print(f"Fixed batching encoding time: {time.time() - start_time:.2f}s")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Encode audio files.')
    parser.add_argument('--indir', type=str, required=True, help='Input directory or filename for audio files.')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory for encoded audio.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for encoding.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for encoding.')

    global DEVICE

    args = parser.parse_args()
    DEVICE = args.device

    # voice_encoder = VoiceEncoder(
    #     bandwidth=VoiceEncoderConfig.bandwidth,
    #     single_segment_duration=VoiceEncoderConfig.single_segment_duration,
    #     batch_size=VoiceEncoderConfig.batch_size,
    #     overlap=VoiceEncoderConfig.overlap,
    #     device=DEVICE,
    # )

    voice_encoder = HubertEncoder(
         config=HubertEncoderConfig(),
         device=DEVICE
    )

    if os.path.isdir(args.indir):
        files = find_audio_files(args.indir)

    else:
        files = [args.indir]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = AudioDataset(
        files,
        sample_rate=16000,
        channels=1
    )

    # dataset = GigaSpeechDataset(
    #     sample_rate=VoiceEncoderConfig.model_sample_rate,
    #     size="m",
    #     split="train"
    # )

    encode(
        voice_encoder=voice_encoder,
        dataset=dataset,
        batch_size=args.batch_size,
        outdir=outdir
    )
