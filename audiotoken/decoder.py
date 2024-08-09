import torch
from encodec import EncodecModel

from .logger import get_logger
from .configs import AcousticDecoderConfig, HubertDecoderConfig

logger = get_logger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
torch.set_float32_matmul_precision("high") # set matmul precision to use either bfloat16 or tf32
# torch.backends.cudnn.benchmark = True  # Selects the best conv algo

class AcousticDecoder(torch.nn.Module):
    def __init__(
            self,
            config: 'AcousticDecoderConfig' = AcousticDecoderConfig(),
            device: str = 'cpu'
        ):

        super().__init__()

        self.model = EncodecModel.encodec_model_24khz()
        self.model.to(device)
        self.model.set_target_bandwidth(config.bandwidth)

        self.model.eval()

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                tokens_batch = input_batch.transpose(0, 1)

                out = self.model.quantizer.decode(tokens_batch)
                out = self.model.decoder(out) # B, 1, L

                logger.info(f'Input shape: {input_batch.shape} Output shape: {out.shape}')

                return out.reshape(-1, ).detach()


class HubertDecoder(torch.nn.Module):
    def __init__(
            self,
            config: 'HubertDecoderConfig' = HubertDecoderConfig(),
            device: str = 'cpu'
        ):

       raise NotImplementedError('HubertDecoder is not implemented yet')

    def forward(self, input_batch: torch.Tensor):
        pass
