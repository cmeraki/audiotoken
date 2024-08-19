from .core import AudioToken
from .configs import AUDIO_EXTS, TAR_EXTS, ZIP_EXTS, Tokenizers

import torch
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    torch.set_float32_matmul_precision("high")  # set matmul precision to use either bfloat16 or tf32
