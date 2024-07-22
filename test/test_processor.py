from transformers import AutoFeatureExtractor, SeamlessM4TFeatureExtractor

from src.configs import Wav2VecBertConfig
from src.utils import read_audio
from optim_impl import OptimizedSeamlessM4TFeatureExtractor
from faster_impl import FasterSeamlessM4TFeatureExtractor

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
        return_tensors='pt'
    )

    # return torch.from_numpy(proc[0])

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

    return torch.from_numpy(out[0])

def faster_impl(a):
    feature_extractor = FasterSeamlessM4TFeatureExtractor(
        feature_size=80,
        num_mel_bins=80,
        padding_value=1,
        return_attention_mask=True,
        sampling_rate=16000,
        stride=2
    )

    out = feature_extractor(a.to('cuda'))

    return out

if __name__ == '__main__':
    audio = read_audio('data/test-clean/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac', 16_000) # type: ignore

    o1 = naive_impl(audio)
    o2 = optim_impl(audio)
    print('**********')
    o3 = faster_impl(audio)

    import pdb
    pdb.set_trace()

    assert o1 == o2
