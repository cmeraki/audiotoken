import os
import time
import torch
from tqdm import tqdm
from queue import Queue
from pathlib import Path
from torch.utils.data import DataLoader

from .encoder import VoiceEncoder
from .configs import VoiceEncoderConfig
from .utils import find_audio_files
from .datasets import AudioDataset

def collate_fn_naive(batch):
    waveforms, _ = zip(*batch)
    sizes = [waveform.size(1) for waveform in waveforms]
    max_length = max(sizes)

    padded_waveforms = []
    for waveform in waveforms:
        padding = max_length - waveform.size(1)
        padded_waveform = torch.nn.functional.pad(waveform, (-1, padding))
        padded_waveforms.append(padded_waveform)

    padded_waveforms = torch.stack(padded_waveforms)
    return padded_waveforms, sizes

def collate_fn_batched(batch):
    return batch

def save_audio_tokens(tokens, file_pointer, root_dir):
    filename, start_idx, end_idx = file_pointer

    filename = filename.split('/')[-1].split('.')[0]
    save_path = os.path.join(root_dir, f'{filename}.pt')
    tokens_to_save = tokens[start_idx:end_idx]
    B, K, T = tokens_to_save.size()
    tokens_to_save = tokens_to_save.permute(1, 0, 2).reshape(K, B*T)

    if os.path.exists(save_path):
        prev_tokens = torch.load(save_path)
        prev_tokens = torch.hstack([prev_tokens, tokens_to_save])
        torch.save(prev_tokens, save_path)

    else:
        torch.save(tokens_to_save, save_path)


@torch.inference_mode()
def encode(voice_encoder, files, batch_size=1, outdir=None):
    dataset = AudioDataset(
        files,
        sample_rate=24000,
        channels=1
    )

    dataloader_batched = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_batched,
        num_workers=12,
        prefetch_factor=8,
        pin_memory=True
    )

    start_time = time.time()

    for batch in tqdm(dataloader_batched, total=len(dataloader_batched)):
        audio_q = Queue()
        for waveform, filename in batch:
            audio_q.put((waveform, filename))

        encoded_audio = voice_encoder(audio_q)
        for tokens_batch, file_pointers in encoded_audio:
            for fp in file_pointers:
                save_audio_tokens(tokens_batch, fp, outdir)

    print(f"Fixed batching encoding time: {time.time() - start_time:.2f}s")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Encode audio files.')
    parser.add_argument('--indir', type=str, required=True, help='Input directory for audio files.')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory for encoded audio.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for encoding.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for encoding.')

    global DEVICE

    args = parser.parse_args()
    DEVICE = args.device

    voice_encoder = VoiceEncoder(
        bandwidth=VoiceEncoderConfig.bandwidth,
        single_segment_duration=VoiceEncoderConfig.single_segment_duration,
        batch_size=VoiceEncoderConfig.batch_size,
        overlap=VoiceEncoderConfig.overlap,
        device=DEVICE,
    )
    files = find_audio_files(args.indir)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    encode(
        voice_encoder,
        files,
        batch_size=args.batch_size,
        outdir=outdir
    )
