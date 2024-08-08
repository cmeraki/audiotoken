import glob
import torch
import itertools
import numpy as np
import multiprocessing as mp
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Optional
from copy import deepcopy
from torch.utils.data import IterableDataset, get_worker_info

from .configs import AudioConfig, AUDIO_EXTS, TAR_EXTS, ZIP_EXTS
from .utils import read_audio, iterate_tar, iterate_zip, process_audio_chunks
from .logger import get_logger

logger = get_logger(__name__)

def collate_fn(batch):
    segments, attention_masks, file_names = zip(*batch)
    return torch.stack(segments), torch.stack(attention_masks), file_names


class AudioBatchDataset(IterableDataset):
    def __init__(
            self,
            sample_rate: int,
            model_token_rate: int,
            chunk_size: int,
            transform=None,
            pad_token: Optional[int] = 0,
            audio_files: Optional[List[str]] = None,
            audio_dir: Optional[str] = None,
            exts: tuple = AUDIO_EXTS + TAR_EXTS + ZIP_EXTS
        ):

        assert audio_files or audio_dir, "Either audio_files or audio_dir must be provided"

        self.decode_path = False
        if audio_files:
            # From: https://github.com/pytorch/pytorch/issues/13246#issuecomment-715050814
            # If the list of audio files is Python list, the memory usage is very high when using multiple workers
            # It is better to convert the list to numpy array
            # More here: https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
            self.audio_files = np.array(audio_files).astype(np.string_)
            self.decode_path = True

        elif audio_dir:
            self.audio_files = itertools.chain.from_iterable(
                glob.iglob(f"{audio_dir}/**/*{ext}", recursive=True) for ext in exts
            ) # type: ignore

        self.audio_q: mp.Queue = mp.Queue(maxsize=10000)

        self.sample_rate = sample_rate
        self.model_token_rate = model_token_rate
        self.transform = transform
        self.pad_token = pad_token

        self.chunk_size = chunk_size
        self.segment_length = chunk_size*sample_rate
        self.stride = int(self.segment_length)

        # Run in a different process to avoid blocking the main process
        process = mp.Process(target=self._populate_q)
        process.start()

    def _populate_q(self):
        for file_path in self.audio_files:
            file_path = str(file_path, encoding='utf-8') if self.decode_path else file_path
            self.audio_q.put(file_path)
            logger.debug(f'Putting in q: {file_path}, {self.audio_q.qsize()}')

        self.audio_q.put(None)

    def _iter_chunk(self, waveform: torch.Tensor, file_name: str):
        length = waveform.shape[-1]

        if self.transform:
            waveform = self.transform(waveform)

        audio_config = AudioConfig(
            file_name=file_name,
            length_seconds=length/ self.sample_rate,
            length_samples=length,
            model_token_rate=self.model_token_rate
        )

        for i in range(0, length, self.stride):
            segment = waveform[0, i:i+self.segment_length]
            attention_mask = torch.ones(segment.shape[0])
            audio_config.start_idx = i
            audio_config.end_idx = min(i + self.segment_length, length)

            # Make sure that a segment is at least 0.2 second long
            if segment.shape[-1] < 3200:
                logger.warning(f'File segment {i//self.sample_rate} of {file_name} is too short. Skipping')
                continue

            if segment.shape[0] < self.segment_length:
                padded_segment_len = self.segment_length - segment.shape[0]

                attention_mask = F.pad(attention_mask, (0, padded_segment_len), value=0)
                segment = F.pad(segment, (0, padded_segment_len), value=self.pad_token)

            yield segment, attention_mask, deepcopy(audio_config)

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        logger.info(f"Starting worker {worker_id}")

        while self.audio_q.qsize() > 0:
            file_path = self.audio_q.get()

            if file_path is None:
                logger.info(f"Worker {worker_id} is done processing")
                break

            logger.info(f'Worker {worker_id} is processing {file_path}')

            if file_path.endswith(AUDIO_EXTS):
                with open(file_path, "rb") as file_stream:
                    for waveform, file_name in process_audio_chunks(
                        file_path, file_stream, self.sample_rate, self.chunk_size
                    ):
                        yield from self._iter_chunk(waveform, file_name)

            elif file_path.endswith(TAR_EXTS):
                for waveform, file_name in iterate_tar(file_path, self.sample_rate, self.chunk_size):
                    yield from self._iter_chunk(waveform, file_name)

            elif file_path.endswith(ZIP_EXTS):
                for waveform, file_name in iterate_zip(file_path, self.sample_rate, self.chunk_size):
                    yield from self._iter_chunk(waveform, file_name)

            else:
                logger.error(f"File {file_path} not supported for processing. Only {AUDIO_EXTS + TAR_EXTS + ZIP_EXTS} supported")

            logger.info(f"Processed complete file at {file_path}")

    def __del__(self):
        pass

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

    single_segment_duration = 10

    args = parser.parse_args()
    files = find_files(args.indir, AUDIO_EXTS + TAR_EXTS + ZIP_EXTS)
    files = sorted(files)

    print('Found files:', len(files))

    if args.tokenizer == 'encodec':
        from .configs import AcousticEncoderConfig

        dataset = AudioBatchDataset(
            audio_files=files,
            sample_rate=AcousticEncoderConfig.model_sample_rate,
            chunk_size=single_segment_duration,
            model_token_rate=AcousticEncoderConfig.model_token_rate,
            pad_token=AcousticEncoderConfig.pad_token
        )

    elif args.tokenizer == 'hubert':
        from transformers import Wav2Vec2FeatureExtractor
        from .encoder import hubert_processor
        from .configs import HubertEncoderConfig

        processor = Wav2Vec2FeatureExtractor.from_pretrained(HubertEncoderConfig.model_id)
        tranform_func = partial(hubert_processor, processor=processor)

        dataset = AudioBatchDataset(
            audio_files=files,
            sample_rate=HubertEncoderConfig.model_sample_rate,
            chunk_size=single_segment_duration,
            transform=tranform_func,
            model_token_rate=HubertEncoderConfig.model_token_rate,
            pad_token=HubertEncoderConfig.pad_token
        )

    elif args.tokenizer == 'w2vbert2':
        from .configs import Wav2VecBertConfig

        dataset = AudioBatchDataset(
            audio_files=files,
            sample_rate=Wav2VecBertConfig.model_sample_rate,
            chunk_size=single_segment_duration,
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
