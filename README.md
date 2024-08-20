# Audiotoken

Tokenize audio to get acoustic and semantic tokens.

## Installation

```bash
pip install audiotoken
```

## Usage

### Encoding

You can either use an acoustic or semantic encoder to encode audio and get tokens.

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

### Decoding

You can decode acoustic tokens like this:

```python
from pathlib import Path
from audiotoken import AudioToken, Tokenizers

tokenizer = AudioToken(tokenizer=Tokenizers.acoustic, device='cuda:0')
encoded_audio = tokenizer.encode(Path('path/to/audio.wav'))
decoded_audio = tokenizer.decode(encoded_audio)

# Save the decoded audio and compare it with the original audio
import torch
import torchaudio
torchaudio.save(
    'reconstructed.wav',
    decoded_audio,
    sample_rate=24000
)
```

You can decode semantic tokens like this:

```python
from pathlib import Path
from audiotoken import AudioToken, Tokenizers

semantic_tokenizer = AudioToken(tokenizer=Tokenizers.semantic_s, device='cuda:0')
semantic_toks = semantic_tokenizer.encode(Path('path/to/audio.wav'))
acoustic_toks = semantic_tokenizer.decode(semantic_toks)

acoustic_tokenizer = AudioToken(tokenizer=Tokenizers.acoustic, device='cuda:0')
decoded_audio = acoustic_tokenizer.decode(acoustic_toks)

# Save the decoded audio and compare it with the original audio
import torch
import torchaudio
torchaudio.save(
    'reconstructed.wav',
    decoded_audio,
    sample_rate=24000
)
```

See [examples/usage.ipynb](examples/usage.ipynb) for more usage examples.

## APIs

Core class

```python
from audiotoken import AudioToken, Tokenizers
tokenizer = AudioToken(tokenizer=Tokenizers.semantic_m, device='cuda:0')
```

See [audiotoken/core.py](audiotoken/core.py) for complete documentation of APIs.

There are 3 APIs provided:

1. `tokenizer.encode`: Encode single audio files/arrays at a time
2. `tokenizer.encode_batch_files`: Encode multiple audio files in batches and save them to disk directly
   1. **NOTE**: `encode_batch_files` is not safe to run multiple times on the same list of files as it can result in incorrect data.
   This will be fixed in a future release.
3. `tokenizer.decode`: Decode acoustic/semantic tokens. Note: Semantic tokens are decoded to acoustic tokens
