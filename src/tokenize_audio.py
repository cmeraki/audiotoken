import time
import torch
import numpy as np
from tqdm import tqdm
from queue import Queue
from torch.utils.data import DataLoader

from .encoder import VoiceEncoder
from .configs import VoiceEncoderConfig
from .utils import find_audio_files
from .datasets import AudioDataset

DEVICE = 'cuda:0'
START_TOKEN = 0


def collate_fn_naive(batch):
    waveforms = batch
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

def codebook_encoding(arr):
    c, n = arr.shape
    i_values = np.arange(c) * 1024
    arr += i_values.reshape(c, 1)
    return arr

def flatten_codebook(arr):
    # give a batch of audio tokens to flatten
    # new_tokenid = old_tokenid + 1024 * codebook_idx
    assert len(arr.shape) == 2
    assert arr.shape[0] < 8

    c, n = arr.shape
    flat_arr = arr.reshape(c*n, order='F')
    return flat_arr

def add_start_token(arr):
    arr = np.insert(arr, 0, START_TOKEN)
    return arr

@torch.inference_mode()
def test_encode(voice_encoder, files, batch_size=1):

    model = voice_encoder.model

    dataset = AudioDataset(
        files,
        sample_rate=24000,
        channels=1
    )

    dataloader_naive = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_naive,
        num_workers=12,
        prefetch_factor=8,
        pin_memory=True
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
    for batch_index, batch in enumerate(tqdm(dataloader_batched)):

        audio_q = Queue(len(batch))
        for waveform in batch:
            audio_q.put(waveform)

        encoded_audio = voice_encoder(audio_q)
        for _ in encoded_audio:
            pass

    print(f"Fixed batching encoding time: {time.time() - start_time:.2f}s")

    start_time = time.time()

    for batch_index, (batch, sizes) in enumerate(tqdm(dataloader_naive)):
        batch = batch.to(DEVICE)
        _ = model.encode(batch)

    naive_time = time.time() - start_time
    print(f"Naive encoding time: {naive_time:.2f}s")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Encode audio files.')
    parser.add_argument('--indir', type=str, required=True, help='Input directory for audio files.')
    parser.add_argument('--outdir', type=str, required=False, help='Output directory for encoded audio.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for encoding.')

    args = parser.parse_args()
    voice_encoder = VoiceEncoder(
        bandwidth=VoiceEncoderConfig.bandwidth,
        single_segment_duration=VoiceEncoderConfig.single_segment_duration,
        batch_size=VoiceEncoderConfig.batch_size,
        overlap=VoiceEncoderConfig.overlap,
        device=DEVICE,
    )
    files = find_audio_files(args.indir)

    test_encode(voice_encoder, files, batch_size=args.batch_size)
