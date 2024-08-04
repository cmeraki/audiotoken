import os
import torch
import numpy as np
from queue import Queue

from encodec.utils import save_audio

from .decoder import VoiceDecoder
from .configs import 
from .utils import find_files

if __name__ == '__main__':
    """
    python -m src.detokenize_audio --indir ./data/tokens --outdir ./data/decoded_audio --device cuda:0
    """
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--indir', type=str, required=True, help='Input directory or filenames for audio tokens.')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory for decoded audio.')
    parser.add_argument('--device', type=str, required=True, help='Device to run the model.')

    args = parser.parse_args()

    voice_decoder = VoiceDecoder(
        bandwidth=.bandwidth,
        single_segment_duration=.single_segment_duration,
        overlap=.overlap,
        device=args.device
    )

    if os.path.isdir(args.indir):
        audio_tokens = find_files(args.indir, ('.npy'))

    else:
        audio_tokens = [args.indir]

    os.makedirs(args.outdir, exist_ok=True)

    for idx, a in enumerate(audio_tokens):
        encoded_audio_q: Queue[torch.Tensor] = Queue()

        tokens = np.load(a)
        tokens = torch.tensor(tokens, device=args.device, dtype=torch.int64).unsqueeze(0)

        encoded_audio_q.put(tokens)

        decoder = voice_decoder(encoded_audio_q)
        decoded_audio = next(iter(decoder))

        save_filename = a.split('/')[-1].replace('.npy', '.wav')
        temp = decoded_audio.detach().cpu().unsqueeze(0)

        print(f'Decoded audio shape: {idx}\t{temp.shape}, {save_filename}')

        save_audio(temp, os.path.join(args.outdir, f'{save_filename}'), 24_000)
