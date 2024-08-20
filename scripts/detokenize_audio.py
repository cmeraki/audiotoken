import os
import torch

from .decoder import AcousticDecoder
from .utils import find_files, save_audio

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

    decoderer = AcousticDecoder(device=args.device)

    if os.path.isdir(args.indir):
        audio_tokens = find_files(args.indir, ('.npy'))

    else:
        audio_tokens = [args.indir]

    os.makedirs(args.outdir, exist_ok=True)

    for idx, a in enumerate(audio_tokens):
        tokens = torch.load(audio_tokens, map_location=args.device).unsqueeze(0)
        tokens = tokens.to(dtype=torch.int64)

        decoded_audio = decoderer(tokens)
        save_filename = os.path.basename(a).replace('.npy', '.wav')

        print(f'Decoded audio shape: {idx}\t{decoded_audio.shape}, {save_filename}')

        save_audio(decoded_audio.cpu(), os.path.join(args.outdir, f'{save_filename}'), 24_000)
