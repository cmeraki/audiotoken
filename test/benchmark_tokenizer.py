import time
import torch
from tqdm import tqdm
from queue import Queue
from torch.utils.data import DataLoader

from ..src.encoder import VoiceEncoder
from ..src.configs import AcousticEncoderConfig
from ..src.utils import find_audio_files
from ..src.datasets import AudioDataset

DEVICE = 'cuda:0'
START_TOKEN = 0


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
    audio_q = Queue()

    for batch_index, batch in enumerate(dataloader_batched):
        for waveform, filename in batch:
            audio_q.put((waveform, filename))

    encoded_audio = voice_encoder(audio_q)
    for op in tqdm(encoded_audio):
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
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for encoding.')

    args = parser.parse_args()
    voice_encoder = VoiceEncoder(
        bandwidth=AcousticEncoderConfig.bandwidth,
        single_segment_duration=AcousticEncoderConfig.single_segment_duration,
        batch_size=AcousticEncoderConfig.batch_size,
        overlap=AcousticEncoderConfig.overlap,
        device=DEVICE,
    )
    files = find_audio_files(args.indir)

    test_encode(
        voice_encoder,
        files,
        batch_size=args.batch_size,
    )
