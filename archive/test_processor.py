import pdb
import torch

from transformers import SeamlessM4TFeatureExtractor
from .configs import Wav2VecBertConfig
from .optim_impl import OptimizedSeamlessM4TFeatureExtractor
from .faster_impl import W2VBert2Processor

def naive_impl(a):
    processor = SeamlessM4TFeatureExtractor.from_pretrained(
        Wav2VecBertConfig.model_id)

    out = processor(
        a,
        sampling_rate=Wav2VecBertConfig.model_sample_rate,
        return_attention_masks=True,
        return_tensors='pt',
        padding=True,
        truncation=False,
        pad_to_multiple_of=500,
    )

    return out['input_features'][0], out['attention_mask'][0]

def optim_impl(a):
    a = a.numpy()
    feature_extractor = OptimizedSeamlessM4TFeatureExtractor(
        feature_size=80,
        num_mel_bins=80,
        padding_value=1,
        return_attention_mask=True,
        sampling_rate=16000,
        pad_to_multiple_of=500,
        stride=2
    )
    out = feature_extractor(a)

    return torch.from_numpy(out['input_features']), torch.from_numpy(out['attention_mask'])

def faster_impl(a):
    feature_extractor = W2VBert2Processor(
        feature_size=80,
        num_mel_bins=80,
        padding_value=1,
        sampling_rate=16000,
        stride=2,
        pad_to_multiple_of=500,
        device='cuda'
    )

    out = feature_extractor(a.to('cuda'))

    return out['input_features'].detach().cpu(), out['attention_mask'].detach().cpu()

if __name__ == '__main__':
    import os
    import time
    import argparse
    import numpy as np
    from tqdm import tqdm

    from .utils import read_audio, find_audio_files

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default='data/test-clean/LibriSpeech/test-clean/1089')

    args = parser.parse_args()

    if os.path.isfile(args.indir):
        audio_files = [args.indir]

    else:
        audio_files = find_audio_files(args.indir)

    print(f'Found {len(audio_files)} audio files')

    hf_speed = []
    cpu_speed = []
    gpu_speed = []

    for audio_path in tqdm(audio_files):
        audio = read_audio(audio_path, 16_000) # type: ignore

        try:
            start_time = time.time()
            i1, a1 = naive_impl(audio)
            hf_speed.append(time.time() - start_time)

            start_time = time.time()
            i2, a2 = optim_impl(audio[0])
            cpu_speed.append(time.time() - start_time)

            start_time = time.time()
            i3, a3 = faster_impl(audio[0])
            torch.cuda.synchronize()
            gpu_speed.append(time.time() - start_time)

            diff = torch.abs(i1 - i2)

            if diff.mean().item():
                print(f'Diff b/w i1, i2: {diff.mean()} {np.percentile(diff, 99)}')

            diff = torch.abs(a1 - a2).to(torch.float32)
            if diff.mean().item():
                print(f'Diff b/w a1, a2: {diff.mean()} {np.percentile(diff, 99)}')

            diff = torch.abs(i2 - i3)
            if diff.mean().item() > 2e-6:
                print(f'Diff b/w i2, i3: {diff.mean()} {np.percentile(diff, 99)}')

            diff = torch.abs(a2 - a3).to(torch.float32)
            if diff.mean().item():
                print(f'Diff b/w a2, a3: {diff.mean()} {np.percentile(diff, 99)}')

        except Exception as e:
            print(f'Error in {audio_path}: {e}')

    pdb.set_trace()
    print(f'HF: {np.mean(hf_speed)}, CPU: {np.mean(cpu_speed)}, GPU: {np.mean(gpu_speed)}')
