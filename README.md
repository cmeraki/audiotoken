# Audiotoken

Tokenize audio to get acoustic and semantic tokens.

## Installation

```bash
pip install audiotoken
```

## Usage

You can get either acoustic or semantic tokens.

```python
from pathlib import Path
from audiotoken import AudioToken, Tokenizers
encoder = AudioToken(tokenizer=Tokenizers.acoustic, device='cuda:0')
encoded_audio = encoder.encode(Path('path/to/audio.wav'))
```

There are 1 acoustic and 2 semantic tokenizers available:

1. `Tokenizers.acoustic`
2. `Tokenizers.semantic_s` (Small)
3. `Tokenizers.semantic_m` (Medium)

You can also decode acoustic tokens

```python
from pathlib import Path
from audiotoken import AudioToken, Tokenizers
encoder = AudioToken(tokenizer=Tokenizers.acoustic, device='cuda:0')
encoded_audio = encoder.encode(Path('path/to/audio.wav'))
decoded_audio = encoder.decode(encoded_audio)

# Save the decoded audio and compare it with the original audio
import torch
import torchaudio
torchaudio.save(
    'test.wav',
    decoded_audio.to(torch.float32).unsqueeze(0),
    sample_rate=24000
)
```

Currently, only the acoustic decoder is supported. We are working to add a semantic decoder as well.
