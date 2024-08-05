import torch
import math
import zipfile
import tarfile
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Optional
from copy import deepcopy
from torch.utils.data import IterableDataset, get_worker_info

from .configs import AudioConfig, AUDIO_EXTS, TAR_EXTS, ZIP_EXTS
from .utils import read_audio, iterate_tar, iterate_zip
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
        # If the list of audio files is Python list, the memory usage is very high when using multiple workers
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

        """
        self.pbar = tqdm(total=len(self.audio_files), desc="Processing audio files", position=-1, leave=True)
        self.files_processed = torch.zeros(1, dtype=torch.long).share_memory_()

        # TODO: There needs to a better approach to handle tar files
        self.pbar.total = 0

        for f in audio_files:
            if f.endswith(TAR_EXTS):
                with tarfile.open(f, 'r') as tar:
                    total_files = len(tar.getnames())
                    logger.info(f"Found {total_files} files in {f}")

                    self.pbar.total += total_files

            elif f.endswith(AUDIO_EXTS):
                self.pbar.total += 1

            elif f.endswith(ZIP_EXTS):
                with zipfile.ZipFile(f, 'r') as z:
                    total_files = len(z.namelist())
                    logger.info(f"Found {total_files} files in {f}")

                    self.pbar.total += total_files
        """

    def _iter_chunk(self, waveform: torch.Tensor, file_name: str):
        length = waveform.shape[-1]

        if self.transform:
            waveform = self.transform(waveform)

        audio_config = AudioConfig(
            file_name=file_name,
            length_seconds=length/ self.sample_rate,
            length_samples=length,
            length_tokens=self.model_token_rate
        )

        # If post transform is provided it is assumed that the transform
        # will take in the audio and produce (N, D) tensor where N = number of tokens
        # and D = dimension of the token
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

        else:
            for i in range(0, length, self.stride):
                segment = waveform[0, i:i+self.segment_length]
                attention_mask = torch.ones(segment.shape[0])
                audio_config.start_idx = i
                audio_config.end_idx = min(i + self.segment_length, length)

                # Make sure that a segment is at least 1 second long
                if segment.shape[-1] < 16000:
                    logger.warning(f'File segment {file_name} is too short. Skipping')
                    continue

                if segment.shape[0] < self.segment_length:
                    padded_segment_len = self.segment_length - segment.shape[0]

                    attention_mask = F.pad(attention_mask, (0, padded_segment_len), value=0)
                    segment = F.pad(segment, (0, padded_segment_len), value=self.pad_token)

                yield segment, attention_mask, deepcopy(audio_config)

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

        for idx in range(iter_start, iter_end):

            file_path = str(self.audio_files[idx], encoding='utf-8')
            logger.info(f'Worker {worker_id} is processing {file_path}')

            if file_path.endswith(AUDIO_EXTS):
                waveform = read_audio(file_path, self.sample_rate)
                yield from self._iter_chunk(waveform, file_path)

                # self.files_processed += 1
                # self.pbar.n = self.files_processed.item()
                # self.pbar.refresh()

            elif file_path.endswith(TAR_EXTS):
                for waveform, file_name in iterate_tar(file_path, self.sample_rate):
                    yield from self._iter_chunk(waveform, file_name)

                    # self.files_processed += 1
                    # self.pbar.n = self.files_processed.item()
                    # self.pbar.refresh()

            elif file_path.endswith(ZIP_EXTS):
                for waveform, file_name in iterate_zip(file_path, self.sample_rate):
                    yield from self._iter_chunk(waveform, file_name)

                    # self.files_processed += 1
                    # self.pbar.n = self.files_processed.item()
                    # self.pbar.refresh()

            else:
                logger.error(f"File {file_path} not supported for processing. Only {AUDIO_EXTS + TAR_EXTS + ZIP_EXTS} supported")

            # with open('logs/processed_mix.txt', 'a') as fp:
            #     fp.write(file_path)
            #     fp.write('\n')

            logger.info(f"Processed complete file at {file_path}")

    def __del__(self):
        pass
        # self.pbar.close()

if __name__ == '__main__':
    import pdb
    from torch.utils.data import DataLoader
    from argparse import ArgumentParser
    from functools import partial
    from torch.cuda import empty_cache

    from .utils import find_files

    DEVICE = 'cuda:0'

    parser = ArgumentParser()
    parser.add_argument('--indir', type=str, required=True, help='Folder to load')
    parser.add_argument('--tokenizer', choices=['encodec', 'hubert', 'w2vbert2', 'whisper'], type=str, required=True, help='Encoder to run.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for encoding.')
    parser.add_argument('--workers', type=int, default=4, help='Batch size for encoding.')

    args = parser.parse_args()
    files = find_files(args.indir, AUDIO_EXTS + TAR_EXTS + ZIP_EXTS)
    files = sorted(files)

    print('Found files:', len(files))

    if args.tokenizer == 'encodec':
        from .configs import VoiceEncoderConfig

        dataset = AudioBatchDataset(
            files,
            sample_rate=VoiceEncoderConfig.model_sample_rate,
            single_segment_duration=VoiceEncoderConfig.single_segment_duration,
            model_token_rate=VoiceEncoderConfig.model_token_rate,
            pad_token=VoiceEncoderConfig.pad_token
        )

    elif args.tokenizer == 'hubert':
        from transformers import Wav2Vec2FeatureExtractor
        from .encoder import hubert_processor
        from .configs import HubertEncoderConfig

        processor = Wav2Vec2FeatureExtractor.from_pretrained(HubertEncoderConfig.model_id)
        tranform_func = partial(hubert_processor, processor=processor)

        dataset = AudioBatchDataset(
            files,
            sample_rate=HubertEncoderConfig.model_sample_rate,
            single_segment_duration=HubertEncoderConfig.single_segment_duration,
            transform=tranform_func,
            model_token_rate=HubertEncoderConfig.model_token_rate,
            pad_token=HubertEncoderConfig.pad_token
        )

    elif args.tokenizer == 'w2vbert2':
        from .configs import Wav2VecBertConfig

        dataset = AudioBatchDataset(
            files,
            sample_rate=Wav2VecBertConfig.model_sample_rate,
            single_segment_duration=Wav2VecBertConfig.single_segment_duration,
            model_token_rate=Wav2VecBertConfig.model_token_rate,
            pad_token=Wav2VecBertConfig.pad_token
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.workers,
        prefetch_factor=4,
        pin_memory=True
    )

    for idx, (segments, attention_masks, file_names) in tqdm(enumerate(dataloader)):
        segments = segments.to(DEVICE)
        attention_masks = attention_masks.to(DEVICE)

        if idx % 100:
            empty_cache()
