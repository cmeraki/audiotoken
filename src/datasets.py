import torch
import math
from typing import List, Optional
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info

from .configs import AudioConfig
from .utils import read_audio
from .logger import logger


def collate_fn(batch):
    segments, attention_masks, file_names = zip(*batch)
    return torch.stack(segments), torch.stack(attention_masks), file_names

class AudioBatchDataset(IterableDataset):
    def __init__(
            self,
            audio_files: List[str],
            sample_rate: int,
            single_segment_duration: int,
            model_token_rate: int,
            transform=None,
            post_transform=None,
            pad_token: Optional[int] = 0,
            overlap: float = 0
        ):

        # From: https://github.com/pytorch/pytorch/issues/13246#issuecomment-715050814
        # If the list of audio files is Python list, the memory usage is very high when using multiple works
        # It is better to convert the list to numpy array
        # More here: https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        self.audio_files = np.array(audio_files).astype(np.string_)

        # self.audio_files = audio_files

        self.sample_rate = sample_rate
        self.model_token_rate = model_token_rate
        self.transform = transform
        self.post_transform = post_transform
        self.pad_token = pad_token

        self.single_segment_duration = single_segment_duration
        self.segment_length = single_segment_duration*sample_rate
        self.stride = int(self.segment_length - overlap * sample_rate)

        # TODO: Implement overlap
        assert self.stride == self.segment_length, "Overlap not supported yet"

        self.pbar = tqdm(total=len(self.audio_files), desc="Processing audio files", position=-1, leave=True)
        self.files_processed = torch.zeros(1, dtype=torch.long).share_memory_()

    def __iter__(self):
        worker_info = get_worker_info()

        iter_start = 0
        iter_end = len(self.audio_files)

        if worker_info is not None:
            per_worker = int(
                math.ceil(len(self.audio_files) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.audio_files))

            logger.info(f"Worker {worker_id} processing files {iter_start} to {iter_end}")

        self.files_processed += 1
        self.pbar.n = self.files_processed.item()
        self.pbar.refresh()

        for idx in range(iter_start, iter_end):
            file_path = str(self.audio_files[idx], encoding='utf-8')
            # file_path = self.audio_files[idx]
            waveform = read_audio(file_path, self.sample_rate)
            length = waveform.shape[-1]

            if self.transform:
                waveform = self.transform(waveform)

            audio_config = AudioConfig(
                file_name=file_path,
                length_seconds=length/ self.sample_rate,
                length_samples=length,
                length_tokens=self.model_token_rate
            )

            # If post transform is provided it is assumed that the transform
            # will take in the audio and produce (N, D) tensor where N 
            # is the number od tokens and D is the dimension of the token
            if self.post_transform:
                input_ids, attention_mask = self.post_transform(waveform)
                input_ids, attention_mask = input_ids.squeeze(0), attention_mask.squeeze(0)
                stride = self.model_token_rate * self.single_segment_duration

                idx = 0
                for i in range(0, input_ids.shape[0], stride):
                    segment = input_ids[i:i+stride, :]
                    mask = attention_mask[i:i+stride]

                    # Store audio configs start idx, end idx in seconds
                    audio_config.start_idx = idx
                    audio_config.end_idx = min(idx + self.segment_length, length)
                    idx += self.segment_length

                    if self.pad_token is None:
                        yield segment, mask, deepcopy(audio_config)
                        continue

                    padded_len = stride - segment.shape[0]
                    segment = F.pad(segment, (0, 0, 0, padded_len), "constant", value=self.pad_token)
                    mask = F.pad(mask, (0, padded_len), "constant", value=0)

                    yield segment, mask, deepcopy(audio_config)

                continue


            for i in range(0, length, self.stride):
                segment = waveform[0, i:i+self.segment_length]
                attention_mask = torch.ones(segment.shape[0])
                audio_config.start_idx = i
                audio_config.end_idx = min(i + self.segment_length, length)

                if segment.shape[0] < self.segment_length:
                    padded_segment_len = self.segment_length - segment.shape[0]

                    attention_mask = F.pad(attention_mask, (0, padded_segment_len), value=0)
                    segment = F.pad(segment, (0, padded_segment_len), value=self.pad_token)

                yield segment, attention_mask, deepcopy(audio_config)
    def __del__(self):
        self.pbar.close()



if __name__ == '__main__':
    import pdb
    from tqdm import tqdm
    from src.utils import find_audio_files
    from torch.utils.data import DataLoader
    from argparse import ArgumentParser
    from functools import partial
    from transformers import AutoFeatureExtractor, WhisperFeatureExtractor

    from .encoder import w2vbert2_processor, whisper_processor
    from .configs import Wav2VecBertConfig, WhisperEncoderConfig

    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/test-clean/LibriSpeech/test-clean/121/121726")

    args = parser.parse_args()
    fns = find_audio_files(args.data_dir)

    # Trying out w2vbert2 pipeline
    processor = AutoFeatureExtractor.from_pretrained(Wav2VecBertConfig.model_id)
    post_transform_func = partial(w2vbert2_processor, processor=processor)

    ds = AudioBatchDataset(
        audio_files=fns,
        sample_rate=Wav2VecBertConfig.model_sample_rate,
        single_segment_duration=Wav2VecBertConfig.single_segment_duration,
        post_transform=post_transform_func,
        model_token_rate=Wav2VecBertConfig.model_token_rate,
        pad_token=Wav2VecBertConfig.pad_token
    )

    for f in ds:
        print(f'Shape of segment {f[0].shape}, Shape of attention mask {f[1].shape}, File name: {f[2].file_name}')


    dataloader = DataLoader(
        ds,
        batch_size=4,
        collate_fn=collate_fn,
        num_workers=4
    )

    for segments, attention_masks, file_names in tqdm(dataloader):
        print(f"Segments shape: {segments.shape}, Attention masks shape: {attention_masks.shape}")

    # Trying out whisper pipeline
    processor = WhisperFeatureExtractor.from_pretrained(WhisperEncoderConfig.model_id)
    post_transform_func = partial(whisper_processor, processor=processor)

    ds = AudioBatchDataset(
        audio_files=fns,
        sample_rate=WhisperEncoderConfig.model_sample_rate,
        single_segment_duration=WhisperEncoderConfig.single_segment_duration,
        post_transform=post_transform_func,
        model_token_rate=WhisperEncoderConfig.model_token_rate,
        pad_token=WhisperEncoderConfig.pad_token
    )

    for f in ds:
        print(f'Shape of segment {f[0].shape}, Shape of attention mask {f[1].shape}, File name: {f[2].file_name}')


    dataloader = DataLoader(
        ds,
        batch_size=4,
        collate_fn=collate_fn,
        num_workers=4
    )

    for segments, attention_masks, file_names in tqdm(dataloader):
        print(f"Segments shape: {segments.shape}, Attention masks shape: {attention_masks.shape}")
