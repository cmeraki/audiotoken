from transformers import AutoFeatureExtractor, SeamlessM4TFeatureExtractor

from src.configs import Wav2VecBertConfig
from src.utils import read_audio
from .optim_impl import OptimizedSeamlessM4TFeatureExtractor
from .faster_impl import FasterSeamlessM4TFeatureExtractor

import torch
import torch.nn.functional as F
import torchaudio
import math


def naive_impl(a):
    processor = SeamlessM4TFeatureExtractor.from_pretrained(
        Wav2VecBertConfig.model_id)

    proc = processor(
        a,
        sampling_rate=Wav2VecBertConfig.model_sample_rate,
        return_attention_masks=True,
        return_tensors='pt',
        padding=True,
        truncation=False,
        pad_to_multiple_of=160,
    )

    return proc

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

    return out

def faster_impl(a):
    feature_extractor = FasterSeamlessM4TFeatureExtractor(
        feature_size=80,
        num_mel_bins=80,
        padding_value=1,
        return_attention_mask=True,
        sampling_rate=16000,
        stride=2,
        device='cuda'
    )

    out = feature_extractor(a.to('cuda'))

    return out.cpu()

def normalize_feats(features):
    mean = features.mean(dim=0, keepdim=True)
    var = features.var(dim=0, keepdim=True, unbiased=True)
    features = (features - mean) / torch.sqrt(var + 1e-7)

    return features

if __name__ == '__main__':
    import numpy as np
    audio = read_audio('data/test-clean/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac', 16_000) # type: ignore

    o1 = naive_impl(audio)
    o2 = optim_impl(audio)
    o3 = faster_impl(audio)

    import pdb
    pdb.set_trace()

    diff = torch.abs(o1 - o2)
    print(f'Diff b/w o1, o2: {diff.mean()} {np.percentile(diff, 99)}')

    diff = torch.abs(o2 - o3)
    print(f'Diff b/w o2, o3: {diff.mean()} {np.percentile(diff, 99)}')

    # import pdb
    # pdb.set_trace()

    # assert o1 == o2
