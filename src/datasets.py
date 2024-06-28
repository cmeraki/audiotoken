import torch
import math
from typing import List
from copy import deepcopy
from torch.utils.data import IterableDataset, get_worker_info

from .configs import AudioConfig
from .utils import read_audio
from .logger import logger

class AudioBatchDataset(IterableDataset):
    def __init__(
            self,
            audio_files: List[str],
            sample_rate: int,
            single_segment_duration: int,
            model_token_rate: int,
            transform=None,
            overlap: float = 0
        ):

        self.audio_files = audio_files
        self.sample_rate = sample_rate
        self.model_token_rate = model_token_rate
        self.transform = transform

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

                num_segments = math.ceil(length / self.segment_length)

                for i in range(0, length, self.stride):
                    segment = waveform[0, i:i+self.segment_length]
                    attention_mask = torch.ones(segment.shape[0])
                    audio_config.start_idx = i
                    audio_config.end_idx = min(i + self.segment_length, length)

                    if segment.shape[0] < self.segment_length:
                        padded_segment_len = self.segment_length - segment.shape[0]

                        attention_mask = torch.nn.functional.pad(
                            attention_mask, (0, padded_segment_len), value=0)
                        segment = torch.nn.functional.pad(
                            segment, (0, padded_segment_len), value=0)

                    yield segment, attention_mask, deepcopy(audio_config)

            except Exception as e:
                logger.error(f"Error reading audio file {file_path}, error: {e}")
                yield None, None, None


if __name__ == '__main__':

    import pdb
    from tqdm import tqdm
    from src.utils import find_audio_files
    from torch.utils.data import DataLoader
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/test-clean/LibriSpeech/test-clean/121/121726")

    args = parser.parse_args()
    fns = find_audio_files(args.data_dir)

    ds = AudioBatchDataset(
        audio_files=fns,
        sample_rate=16000,
        single_segment_duration=5,
        model_token_rate=50
    )

    def collate_fn(batch):
        segments, attention_masks, file_names = zip(*batch)
        return torch.stack(segments), torch.stack(attention_masks), file_names

    def batch_generator(dataloader):
        for batch in dataloader:
            yield batch

    dataloader = DataLoader(
        ds,
        batch_size=5,
        collate_fn=collate_fn,
        num_workers=4
    )

    for segments, attention_masks, file_names in tqdm(batch_generator(dataloader)):
        # pdb.set_trace()
        logger.info(
            f"Files in this batch: {file_names} Segments shape: {segments.shape}, Attention masks shape: {attention_masks.shape}"
        )
        pass
