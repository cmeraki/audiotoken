import bark
import torch
import numpy as np
from encodec import EncodecModel

from .logger import get_logger
from .gpt2_model import get_model
from .configs import AcousticDecoderConfig, HubertDecoderConfig, Wav2VecBertDecoderConfig, COMMONS
from .utils import ctx

logger = get_logger(__name__)

def _prepare_source(
    source_arr: torch.Tensor,
    source_offset: int,
    max_source_tokens: int
):
    source_arr = source_arr + source_offset
    source_arr = source_arr.reshape(1, -1)
    source_arr = source_arr[:, :max_source_tokens]

    return source_arr


def _extract_new_tokens(
        y,
        infer_token,
        stop_token
    ):

    start_idx = np.where(y == infer_token)[0]
    end_idx = np.where(y == stop_token)[0]

    if end_idx.any():
        y = y[start_idx[0] + 1: end_idx[0]]
    else:
        y = y[start_idx[0] + 1:]

    return y


def _deserialize_acoustic_tokens(tokens):
    cb1 = tokens[::2]
    cb2 = tokens[1::2]
    acoustic_tokens = np.stack([cb1, cb2 - 1024])

    return acoustic_tokens


class AcousticDecoder(torch.nn.Module):
    def __init__(
            self,
            config: 'AcousticDecoderConfig' = AcousticDecoderConfig(),
            device: str = 'cpu'
        ):

        super().__init__()

        self.device = device
        self.model = EncodecModel.encodec_model_24khz()
        self.model.to(device)
        self.model.set_target_bandwidth(config.bandwidth)

        self.model.eval()

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        with ctx:
            with torch.no_grad():
                tokens_batch = input_batch.to(self.device).transpose(0, 1)

                out = self.model.quantizer.decode(tokens_batch)
                out = self.model.decoder(out) # B, 1, L

                logger.info(f'Input shape: {input_batch.shape} Output shape: {out.shape}')

                return out.reshape(-1, ).detach().to(torch.float32).unsqueeze(0)


class HubertDecoder(torch.nn.Module):
    def __init__(
            self,
            config: 'HubertDecoderConfig' = HubertDecoderConfig(),
            lanugage: str = COMMONS.EN,
            device: str = 'cpu'
        ):

       super().__init__()

       self.device = device
       self.config = config
       assert lanugage in self.config.suported_languages, f'{lanugage} language not supported for the decoder. Only {self.config.suported_languages} are supported.'
       model_id = self.config.en_model_id

       self.model = get_model(
           vocab_size=self.config.VOCAB_SIZE,
           path=model_id,
           device=self.device
       )
       self.model.to(self.device)
       self.model.eval()

       # Load the model for bark
       _ = bark.generation.load_model(
           use_gpu=True if 'cuda' in self.device else False,
           model_type="fine"
       )

    def nar_bark(self, tokens_02):
        tokens_02 = _deserialize_acoustic_tokens(tokens_02)

        with ctx:
            with torch.no_grad():
                tokens = bark.api.generate_fine(
                    x_coarse_gen=tokens_02[0:2, :],
                    silent=False
                )
        tokens = np.expand_dims(tokens, axis=0)
        tokens = torch.from_numpy(tokens)

        return tokens

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        source_tokens = _prepare_source(
            source_arr=input_batch,
            source_offset=self.config.OFFSET[COMMONS.SEMANTIC],
            max_source_tokens=self.config.max_source_tokens
        )

        infer_tensor = torch.tensor(self.config.INFER_TOKEN[COMMONS.ACOUSTIC]).reshape(1, -1)
        source_tokens = torch.hstack(
            [source_tokens, infer_tensor]
        ).to(self.device)

        with ctx:
            with torch.no_grad():
                target_tokens = self.model.generate(
                    source_tokens,
                    max_new_tokens=1024,
                    temperature=0.8,
                    top_k=100,
                    stop_token=self.config.STOP_TOKEN[COMMONS.ACOUSTIC]
                )

        target_tokens = target_tokens.detach().cpu().numpy()[0]

        target_tokens = _extract_new_tokens(
            y=target_tokens,
            infer_token=self.config.INFER_TOKEN[COMMONS.ACOUSTIC],
            stop_token=self.config.STOP_TOKEN[COMMONS.ACOUSTIC]
        )
        target_tokens = target_tokens - self.config.OFFSET[COMMONS.ACOUSTIC]

        acoustic_tokens = self.nar_bark(target_tokens)

        return acoustic_tokens


class Wav2VecBertDecoder(torch.nn.Module):
    """
    Generates acoustic tokens from semantic tokens
    Acoustic tokens are generated in two steps:
        1. A pretrained autoregressive GPT2 model is used to generate 2 acosutic tokens for each semantic token
        2. Bark's non autoregressive pretrained model is used to generate remaining 6 acoustic tokens
    """
    def __init__(
            self,
            config: 'Wav2VecBertDecoderConfig' = Wav2VecBertDecoderConfig(),
            lanugage: str = COMMONS.HI,
            device: str = 'cpu'
        ):
        super().__init__()

        self.device = device
        self.config = config
        assert lanugage in self.config.suported_languages, f'{lanugage} language not supported for the decoder. Only {self.config.suported_languages} are supported.'
        model_id = self.config.hi_model_id if lanugage == COMMONS.HI else self.config.en_model_id

        self.model = get_model(
            vocab_size=self.config.VOCAB_SIZE,
            path=model_id,
            device=self.device
        )
        self.model.to(self.device)
        self.model.eval()

        # Load the model for bark
        _ = bark.generation.load_model(
            use_gpu=True if 'cuda' in self.device else False,
            model_type="fine"
        )

    def nar_bark(self, tokens_02):
        tokens_02 = _deserialize_acoustic_tokens(tokens_02)

        with ctx:
            with torch.no_grad():
                tokens = bark.api.generate_fine(
                    x_coarse_gen=tokens_02[0:2, :],
                    silent=False
                )
        tokens = np.expand_dims(tokens, axis=0)
        tokens = torch.from_numpy(tokens)

        return tokens

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        source_tokens = _prepare_source(
            source_arr=input_batch,
            source_offset=self.config.OFFSET[COMMONS.SEMANTIC],
            max_source_tokens=self.config.max_source_tokens
        )

        infer_tensor = torch.tensor(self.config.INFER_TOKEN[COMMONS.ACOUSTIC]).reshape(1, -1)
        source_tokens = torch.hstack(
            [source_tokens, infer_tensor]
        ).to(self.device)

        with ctx:
            with torch.no_grad():
                target_tokens = self.model.generate(
                    source_tokens,
                    max_new_tokens=1024,
                    temperature=0.8,
                    top_k=100,
                    stop_token=self.config.STOP_TOKEN[COMMONS.ACOUSTIC]
                )

        target_tokens = target_tokens.detach().cpu().numpy()[0]

        target_tokens = _extract_new_tokens(
            y=target_tokens,
            infer_token=self.config.INFER_TOKEN[COMMONS.ACOUSTIC],
            stop_token=self.config.STOP_TOKEN[COMMONS.ACOUSTIC]
        )
        target_tokens = target_tokens - self.config.OFFSET[COMMONS.ACOUSTIC]

        acoustic_tokens = self.nar_bark(target_tokens)

        return acoustic_tokens
