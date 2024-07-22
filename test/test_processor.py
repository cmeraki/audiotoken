import pdb
import torch

from transformers import SeamlessM4TFeatureExtractor
from src.configs import Wav2VecBertConfig
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
        pad_to_multiple_of=2,
    )

    return out['input_features'], out['attention_mask']

def optim_impl(a):
    a = a.numpy()
    feature_extractor = OptimizedSeamlessM4TFeatureExtractor(
        feature_size=80,
        num_mel_bins=80,
        padding_value=1,
        return_attention_mask=True,
        sampling_rate=16000,
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
        device='cuda'
    )

    out = feature_extractor(a.to('cuda'))

    return out['input_features'].detach().cpu(), out['attention_mask'].detach().cpu()

if __name__ == '__main__':
    import time
    import argparse
    import numpy as np

    from src.utils import read_audio, find_audio_files

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default='data/test-clean/LibriSpeech/test-clean/1089')

    args = parser.parse_args()

    audio_files = find_audio_files(args.indir)

    for audio_path in audio_files:
        audio = read_audio(audio_path, 16_000) # type: ignore

        start_time = time.time()
        i1, a1 = naive_impl(audio)
        print(f'Naive: {time.time() - start_time}')

        start_time = time.time()
        i2, a2 = optim_impl(audio)
        print(f'Optim: {time.time() - start_time}')

        start_time = time.time()
        i3, a3 = faster_impl(audio)
        torch.cuda.synchronize()
        print(f'Faster: {time.time() - start_time}')

        diff = torch.abs(i1[0] - i2)
        print(f'Diff b/w i1, i2: {diff.mean()} {np.percentile(diff, 99)}')

        diff = torch.abs(a1[0] - a2).to(torch.float32)
        print(f'Diff b/w a1, a2: {diff.mean()} {np.percentile(diff, 99)}')

        diff = torch.abs(i2 - i3)
        print(f'Diff b/w i2, i3: {diff.mean()} {np.percentile(diff, 99)}')

        diff = torch.abs(a2 - a3).to(torch.float32)
        print(f'Diff b/w a2, a3: {diff.mean()} {np.percentile(diff, 99)}')
