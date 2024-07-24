"""
This is modification of the modeling_wav2vec2_bert.py from Huggingface Transformers library
https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2_bert/modeling_wav2vec2_bert.py#L448

This updated forward pass adds torch.nn.functional.scaled_dot_product_attention function for attention calculation
which is approximately 10x faster than torch.matmul(probs, value) calculation.
"""

__author__      = "Romit Jain"
__email__       = "romit@merakilabs.com"
__maintainer__  = "Romit Jain"
__status__      = "Development"


import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    relative_position_embeddings: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    assert self.position_embeddings_type == "relative_key", "Only `relative_key` is supported for `self.position_embeddings_type` now"

    # self-attention mechanism
    batch_size, sequence_length, hidden_size = hidden_states.size()

    # make sure query/key states can be != value states
    query_key_states = hidden_states
    value_states = hidden_states

    # project query_key_states and value_states
    query = self.linear_q(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
    key = self.linear_k(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
    value = self.linear_v(value_states).view(batch_size, -1, self.num_heads, self.head_size)

    # => (batch, head, time1, d_k)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    if self.position_embeddings_type == "relative_key":
        query_length, key_length = query.shape[2], key.shape[2]

        position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
        position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
        distance = position_ids_r - position_ids_l
        distance = torch.clamp(distance, -self.left_max_position_embeddings, self.right_max_position_embeddings)

        positional_embedding = self.distance_embedding(distance + self.left_max_position_embeddings)
        positional_embedding = positional_embedding.to(dtype=query.dtype)  # fp16 compatibility

        relative_position_attn_weights = torch.einsum("bhld,lrd->bhlr", query, positional_embedding)
        relative_position_attn_weights = (relative_position_attn_weights / math.sqrt(self.head_size))

    # Create `attn_mask` that will be passed to the spda
    attn_mask = relative_position_attn_weights
    # Apply attention_mask if necessary
    if attention_mask is not None:
        attn_mask += attention_mask

    # => (batch, head, time1, time2)
    hidden_states = nn.functional.scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        scale=1/math.sqrt(self.head_size)
    )

    # => (batch, time1, hidden_size)
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_size)
    hidden_states = self.linear_out(hidden_states)

    # Passing None for probs
    return hidden_states, None # type: ignore
