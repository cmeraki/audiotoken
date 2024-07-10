import torch
import math
from typing import List
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
            pad_token: int = 0,
            overlap: float = 0
        ):

        self.audio_files = audio_files
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

    def __iter__(self):
        worker_info = get_worker_info()

        if worker_info is None:
            iter_start = 0
            iter_end = len(self.audio_files)

        else:
            per_worker = int(
                math.ceil(len(self.audio_files) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.audio_files))

            logger.info(f"Worker {worker_id} processing files {iter_start} to {iter_end}")

        for idx in range(iter_start, iter_end):
            file_path = self.audio_files[idx]

            try:
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

                        padded_len = stride - segment.shape[0]
                        segment = F.pad(segment, (0, 0, 0, padded_len), "constant", value=self.pad_token)
                        mask = F.pad(mask, (0, padded_len), "constant", value=self.pad_token)

                        # Store audio configs start idx, end idx in seconds
                        audio_config.start_idx = idx
                        audio_config.end_idx = min(idx + self.segment_length, length)
                        idx += self.segment_length

                        yield segment, mask, deepcopy(audio_config)

                    continue


                for i in range(0, length, self.stride):
                    segment = waveform[0, i:i+self.segment_length]
                    attention_mask = torch.ones(segment.shape[0])
                    audio_config.start_idx = i
                    audio_config.end_idx = min(i + self.segment_length, length)

                    if segment.shape[0] < self.segment_length:
                        padded_segment_len = self.segment_length - segment.shape[0]

                        attention_mask = F.pad(attention_mask, (0, padded_segment_len), value=self.pad_token)
                        segment = F.pad(segment, (0, padded_segment_len), value=self.pad_token)

                    yield segment, attention_mask, deepcopy(audio_config)

            except Exception as e:
                logger.error(f"Error reading audio file {file_path}, error: {e}")
                continue


if __name__ == '__main__':

    from tqdm import tqdm
    from src.utils import find_audio_files
    from torch.utils.data import DataLoader
    from argparse import ArgumentParser
    from functools import partial
    from transformers import AutoFeatureExtractor

    from .encoder import wav2vec_processor
    from .configs import Wav2VecBertConfig

    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/test-clean/LibriSpeech/test-clean/121/121726")

    args = parser.parse_args()
    fns = find_audio_files(args.data_dir)


    processor = AutoFeatureExtractor.from_pretrained(Wav2VecBertConfig.model_id)
    post_transform_func = partial(wav2vec_processor, processor=processor)

    ds = AudioBatchDataset(
        audio_files=fns,
        sample_rate=Wav2VecBertConfig.model_sample_rate,
        single_segment_duration=Wav2VecBertConfig.single_segment_duration,
        post_transform=post_transform_func,
        model_token_rate=Wav2VecBertConfig.model_token_rate
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
        # pdb.set_trace()
        print(f"Files in this batch: {file_names} Segments shape: {segments.shape}, Attention masks shape: {attention_masks.shape}")
        pass
