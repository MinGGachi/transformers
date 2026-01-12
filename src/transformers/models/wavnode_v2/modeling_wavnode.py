# coding=utf-8
# Copyright 2021 The Fairseq Authors and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
####
# written by Min Jun Choi
# Ph.D. Student, Music & Audio Research Group, Seoul Nat'l Univ.
# Last modified: 25.09.25
####
"""PyTorch WavNode model."""
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import (
    _prepare_4d_attention_mask,
    _prepare_4d_attention_mask_for_sdpa,
)
from ...integrations.fsdp import is_fsdp_managed_module
from ...modeling_outputs import (
    BaseModelOutput,
    CausalLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    Wav2Vec2BaseModelOutput,
    XVectorOutput,
)
WavNodeBaseModelOutput = Wav2Vec2BaseModelOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...utils import (
    ModelOutput, 
    auto_docstring, 
    logging
)
from .configuration_wavnode import WavNodeConfig

# from torchdiffeq import odeint
# from torchdiffeq import odeint_adjoint as odeint
from .model import *


logger = logging.get_logger(__name__)


@dataclass
class WavNodeForPretrainingOutput(ModelOutput):
    """
    Output type of [`Wav2Vec2ForPreTraining`], with potential hidden states and attentions.

    Args:
        loss (*optional*, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the contrastive loss (L_m) and the diversity loss (L_d) as stated in the [official
            paper](https://arxiv.org/pdf/2006.11477.pdf) . (classification) loss.
        projected_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`):
            Hidden-states of the model projected to *config.proj_codevector_dim* that can be used to predict the masked
            projected quantized states.
        projected_quantized_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`):
            Quantized extracted feature vectors projected to *config.proj_codevector_dim* representing the positive
            target vectors for contrastive loss.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        contrastive_loss (*optional*, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`):
            The contrastive loss (L_m) as stated in the [official paper](https://arxiv.org/pdf/2006.11477.pdf) .
        diversity_loss (*optional*, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`):
            The diversity loss (L_d) as stated in the [official paper](https://arxiv.org/pdf/2006.11477.pdf) .
    """
    loss: Optional[torch.FloatTensor] = None
    projected_states: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    nce_loss: Optional[torch.FloatTensor] = None
    stiffness_loss: Optional[torch.FloatTensor] = None


def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.LongTensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape. Used to implement [SpecAugment: A Simple Data Augmentation Method for
    ASR](https://arxiv.org/abs/1904.08779). Note that this method is not optimized to run on TPU and should be run on
    CPU as part of the preprocessing during training.

    Args:
        shape: The shape for which to compute masks. This should be of a tuple of size 2 where
               the first element is the batch size and the second element is the length of the axis to span.
        mask_prob:  The percentage of the whole axis (between 0 and 1) which will be masked. The number of
                    independently generated mask spans of length `mask_length` is computed by
                    `mask_prob*shape[1]/mask_length`. Note that due to overlaps, `mask_prob` is an upper bound and the
                    actual percentage will be smaller.
        mask_length: size of the mask
        min_masks: minimum number of masked spans
        attention_mask: A (right-padded) attention mask which independently shortens the feature axis of
                        each batch dimension.
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon is used for probabilistic rounding
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # make sure num masked span <= sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # make sure num_masked span is also <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # compute number of masked spans in batch
    input_lengths = (
        attention_mask.detach().sum(-1).tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)

    if max_num_masked_span == 0:
        return spec_aug_mask

    for input_length in input_lengths:
        # compute num of masked spans for this input
        num_masked_span = compute_num_masked_span(input_length)

        # get random indices to mask
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # pick first sampled index that will serve as a dummy index to pad vector
        # to ensure same dimension for all batches due to probabilistic rounding
        # Picking first sample just pads those vectors twice.
        if len(spec_aug_mask_idx) == 0:
            # this case can only happen if `input_length` is strictly smaller then
            # `sequence_length` in which case the last token has to be a padding
            # token which we can use as a dummy mask id
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # add offset to the starting indexes so that indexes now create a span
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # ensure that we cannot have indices larger than sequence_length
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # scatter indices to mask
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask


class WavNodeSamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


class WavNodePositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )

        self.batch_norm = None
        if config.conv_pos_batch_norm:
            self.batch_norm = nn.BatchNorm1d(config.hidden_size)
        else:
            weight_norm = nn.utils.weight_norm
            if hasattr(nn.utils.parametrizations, "weight_norm"):
                weight_norm = nn.utils.parametrizations.weight_norm

            if is_deepspeed_zero3_enabled():
                import deepspeed

                with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):
                    self.conv = weight_norm(self.conv, name="weight", dim=2)
                if hasattr(self.conv, "parametrizations"):
                    weight_g = self.conv.parametrizations.weight.original0
                    weight_v = self.conv.parametrizations.weight.original1
                else:
                    weight_g = self.conv.weight_g
                    weight_v = self.conv.weight_v
                deepspeed.zero.register_external_parameter(self, weight_v)
                deepspeed.zero.register_external_parameter(self, weight_g)
            else:
                self.conv = weight_norm(self.conv, name="weight", dim=2)

        self.padding = WavNodeSamePadLayer(config.num_conv_pos_embeddings)
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)
        if self.batch_norm is not None:
            hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class WavNodeFeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feat_proj_layer_norm = config.feat_proj_layer_norm
        if self.feat_proj_layer_norm:
            self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states):
        # non-projected hidden states are needed for quantization
        if self.feat_proj_layer_norm:
            hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class WavNodeNoLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class WavNodeLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)

        hidden_states = hidden_states.transpose(-2, -1)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(-2, -1)

        hidden_states = self.activation(hidden_states)
        return hidden_states


class WavNodeGroupNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.activation = ACT2FN[config.feat_extract_activation]

        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class WavNodeFeatureEncoder(nn.Module):
    """Construct the features from raw audio waveform"""

    def __init__(self, config):
        super().__init__()

        if config.feat_extract_norm == "group":
            conv_layers = [WavNodeGroupNormConvLayer(config, layer_id=0)] + [
                WavNodeNoLayerNormConvLayer(config, layer_id=i + 1) for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            conv_layers = [WavNodeLayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)]
        else:
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        self.conv_layers = nn.ModuleList(conv_layers)
        self.gradient_checkpointing = False
        self._requires_grad = True

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def forward(self, input_values):
        hidden_states = input_values[:, None]

        # make sure hidden_states require grad for gradient_checkpointing
        if self._requires_grad and self.training:
            hidden_states.requires_grad = True

        for conv_layer in self.conv_layers:
            if self._requires_grad and self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    conv_layer.__call__,
                    hidden_states,
                )
            else:
                hidden_states = conv_layer(hidden_states)

        return hidden_states


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    head_mask: Optional[torch.Tensor] = None,
    **kwargs,
):
    if scaling is None:
        scaling = query.size(-1) ** -0.5

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    if head_mask is not None:
        attn_weights = attn_weights * head_mask.view(1, -1, 1, 1)

    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class HuBERTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        dropout: float = 0.0,
        use_sdpa: bool = False,
        config: Optional[WavNodeConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if use_sdpa:
            print("If you want to get attention matrix, please set sdpa to False.")
            self.attention_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]
        else:
            self.attention_interface = eager_attention_forward

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Attention layer with relative attention"""
        bsz, tgt_len, _  = hidden_states.size()
        input_shape = (bsz, tgt_len, -1, self.head_dim)

        # 6) project hidden states
        query_states = self.q_proj(hidden_states).reshape(*input_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).reshape(*input_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).reshape(*input_shape).transpose(1, 2)

        # 7) apply attention
        attn_output, attn_weights = self.attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=self.scaling,
            output_attentions=output_attentions,
            head_mask=None,
            is_causal=False,
        )

        # 8) apply output projection
        attn_output = attn_output.reshape(bsz, tgt_len, -1)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights
    

class HubertFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)

        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, 
                hidden_states: torch.Tensor,
                ):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class HubertEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = HuBERTAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
            config=config,
        )

        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = HubertFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        attn_residual = hidden_states
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        if output_attentions:
            return hidden_states, attn_weights
        else:
            return hidden_states, None
    

class WavNodeAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        dropout: float = 0.0,
        use_sdpa: bool = False,
        rank: int = None,
        time_dim: int = 128,
        hidden_dim: int = 128,
        time_activation: nn.Module = nn.SiLU(),
        config: Optional[WavNodeConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.q_proj = Linear(in_features=embed_dim, out_features=embed_dim, bias=bias,
                             rank=rank, time_dim=time_dim, hidden_dim=hidden_dim, activation=time_activation)
        self.k_proj = Linear(in_features=embed_dim, out_features=embed_dim, bias=bias,
                             rank=rank, time_dim=time_dim, hidden_dim=hidden_dim, activation=time_activation)
        self.v_proj = Linear(in_features=embed_dim, out_features=embed_dim, bias=bias,
                             rank=rank, time_dim=time_dim, hidden_dim=hidden_dim, activation=time_activation)
        self.out_proj = Linear(in_features=embed_dim, out_features=embed_dim, bias=bias,
                               rank=rank, time_dim=time_dim, hidden_dim=hidden_dim, activation=time_activation)

        if use_sdpa:
            print("If you want to get attention matrix, please set sdpa to False.")
            self.attention_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]
        else:
            self.attention_interface = eager_attention_forward

    def forward(
        self,
        t_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Attention layer with relative attention"""
        bsz, tgt_len, _  = hidden_states.size()
        input_shape = (bsz, tgt_len, -1, self.head_dim)

        # 6) project hidden states
        query_states = self.q_proj(t_emb, hidden_states).reshape(*input_shape).transpose(1, 2)
        key_states = self.k_proj(t_emb, hidden_states).reshape(*input_shape).transpose(1, 2)
        value_states = self.v_proj(t_emb, hidden_states).reshape(*input_shape).transpose(1, 2)

        # 7) apply attention
        attn_output, attn_weights = self.attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=self.scaling,
            output_attentions=output_attentions,
            head_mask=None,
            is_causal=False,
        )

        # 8) apply output projection
        attn_output = attn_output.reshape(bsz, tgt_len, -1)
        attn_output = self.out_proj(t_emb, attn_output)

        return attn_output, attn_weights


class WavNodeFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        # activation function for the time-varying modules
        if isinstance(config.time_activation, str):
            time_activation = ACT2FN[config.time_activation]
        else:
            time_activation = config.time_activation
        
        self.intermediate_dense = Linear(config.hidden_size, config.intermediate_size,
                                         rank=config.hidden_size, time_dim=config.time_dim,
                                         hidden_dim=config.hidden_dim, activation=time_activation)
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)
        
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        self.output_dense = Linear(config.intermediate_size, config.hidden_size,
                                   rank=config.hidden_size, time_dim=config.time_dim,
                                   hidden_dim=config.hidden_dim, activation=time_activation)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, 
                t_emb: torch.Tensor,
                hidden_states: torch.Tensor,
                ):
        hidden_states = self.intermediate_dense(t_emb, hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(t_emb, hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class WavNodeEncoderLayer(nn.Module):
    def __init__(self, config: WavNodeConfig,):
        super().__init__()

        if isinstance(config.time_activation, str):
            time_activation = ACT2FN[config.time_activation]
        else:
            time_activation = config.time_activation
        self.attn = WavNodeAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            bias=True,
            dropout=config.attention_dropout,
            use_sdpa=config.use_sdpa,
            rank=config.rank, 
            time_dim=config.time_dim, 
            hidden_dim=config.hidden_dim,
            time_activation=time_activation,
            config=config,
        )
        self.attn_ln = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, 
                                 time_dim=config.time_dim, hidden_dim=config.hidden_dim, 
                                 activation=time_activation)
        self.ffn = WavNodeFeedForward(config)
        self.ffn_ln = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, 
                                time_dim=config.time_dim, hidden_dim=config.hidden_dim, 
                                activation=time_activation)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.time_embed = SinusoidalEmbedding(time_dim=config.time_dim)

        # Pre-collect layers that need orthonormalization to avoid traversal overhead at every step
        self.ortho_layers = [m for m in self.modules() if isinstance(m, Linear) and hasattr(m, "_orthonormalize")]

    def _orthonormalize(self):
        for layer in self.ortho_layers:
            layer._orthonormalize()

    def forward(self, 
                time, 
                hidden_states, 
                attention_mask=None, 
                output_attentions=False, 
            ):
        t_emb = self.time_embed(time)
        # attn
        hidden_states_attn = self.attn_ln(t_emb, hidden_states)
        hidden_states_attn, attn_weights = self.attn(
            t_emb,
            hidden_states_attn,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states_attn = self.dropout(hidden_states_attn)

        # ffn
        hidden_states_ffn = self.ffn_ln(t_emb, hidden_states)
        hidden_states_ffn = self.ffn(t_emb, hidden_states_ffn)
        hidden_states_ffn = self.dropout(hidden_states_ffn)
        
        hidden_states = hidden_states_ffn + hidden_states_attn

        if output_attentions:
            return hidden_states, attn_weights
        else:
            return hidden_states, None


class WavNodeEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_conv_embed = WavNodePositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.hubert_layer = HubertEncoderLayer(config)
        self.ode_layer = WavNodeEncoderLayer(config)
        self.solver = Solver(f=self.ode_layer, 
                             step_method=config.step_method, 
                             use_checkpoint=config.use_checkpoint,)

    def forward(
        self,
        hidden_states: torch.tensor,
        times: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens output 0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0

        attention_mask = self._update_full_mask(
            attention_mask,
            hidden_states,
        )

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states, self_attentions = self.hubert_layer(hidden_states, 
                                                           attention_mask=attention_mask, 
                                                           output_attentions=output_attentions,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if output_attentions:
            all_self_attentions = all_self_attentions + (self_attentions,)

        # Solver returns (trajectory, aux_outputs_list)
        # aux_outputs_list is a list of tuples, e.g., [(attn_w_step1,), (attn_w_step2,), ...]
        stacked_hidden_states, aux_outputs = self.solver(times, hidden_states,
                                                         return_trajectory=output_hidden_states,
                                                         attention_mask=attention_mask,
                                                         output_attentions=output_attentions,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + tuple(torch.unbind(stacked_hidden_states, dim=0))

        if output_attentions:
            # Extract attention weights from aux_outputs list of tuples
            # Each aux_output is expected to be (attn_weights,)
            step_attentions = tuple(aux[0] for aux in aux_outputs)
            all_self_attentions = all_self_attentions + step_attentions

        if not return_dict:
            return tuple(v for v in [stacked_hidden_states[-1], all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=stacked_hidden_states[-1],
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    # Copied from transformers.models.bart.modeling_bart.BartPreTrainedModel._update_full_mask
    def _update_full_mask(
        self,
        attention_mask: Union[torch.Tensor, None],
        inputs_embeds: torch.Tensor,
    ):
        if attention_mask is not None:
            if self.config.use_sdpa:
                # output_attentions=True & head_mask can not be supported when using SDPA, fall back to
                # the manual implementation that requires a 4D causal mask in all cases.
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _prepare_4d_attention_mask_for_sdpa(attention_mask, inputs_embeds.dtype)
            else:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

        return attention_mask


@auto_docstring
class WavNodePreTrainedModel(PreTrainedModel):
    config_class = WavNodeConfig
    base_model_prefix = "wavnode"
    main_input_name = "input_values"
    supports_gradient_checkpointing = False
    _supports_flash_attn_2 = False
    _supports_sdpa = True
    _supports_flex_attn = False

    # TODO: check initialization for time dependent layers
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, WavNodePositionalConvEmbedding):
            nn.init.normal_(
                module.conv.weight,
                mean=0,
                std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
            )
            nn.init.constant_(module.conv.bias, 0)
        elif isinstance(module, WavNodeFeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)

            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[torch.LongTensor, int],
    ):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor,
    ):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths)
        output_lengths = output_lengths.to(torch.long)

        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask


@auto_docstring
class WavNodeModel(WavNodePreTrainedModel):
    def __init__(self, config: WavNodeConfig):
        super().__init__(config)
        self.config = config
        self.feature_encoder = WavNodeFeatureEncoder(config)
        self.feature_projection = WavNodeFeatureProjection(config)

        # model only needs masking vector if mask prob is > 0.0
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.Tensor(config.hidden_size).uniform_())

        self.encoder = WavNodeEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.feature_encoder._freeze_parameters()

    def _mask_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """

        # `config.apply_spec_augment` can set masking to False
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        # generate indices & apply SpecAugment along time axis
        batch_size, sequence_length, hidden_size = hidden_states.size()

        if mask_time_indices is not None:
            # apply SpecAugment along time axis with given mask_time_indices
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and self.training:
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=hidden_states.device, dtype=torch.bool)
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)

        if self.config.mask_feature_prob > 0 and self.training:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(mask_feature_indices, device=hidden_states.device, dtype=torch.bool)
            mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
            hidden_states[mask_feature_indices] = 0

        return hidden_states

    @auto_docstring
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, WavNodeBaseModelOutput]:
        r"""
        mask_time_indices (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices to mask extracted features for contrastive loss. When in training mode, model learns to predict
            masked extracted features in *config.proj_codevector_dim* space.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_encoder(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        return WavNodeBaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    WavNode Model with a learnable codebook on top.
    """
)
class WavNodeForPreTraining(WavNodePreTrainedModel):
    def __init__(self, config: WavNodeConfig):
        super().__init__(config)
        # wavnode model
        self.wavnode = WavNodeModel(config)

        # for pretraining head
        self.final_proj = nn.Linear(config.hidden_size, config.codevector_dim)
        self.label_embs = nn.Parameter(torch.FloatTensor(config.num_classes, config.codevector_dim))
        nn.init.uniform_(self.label_embs)

        if getattr(config, "use_target_glu", False):
            self.target_glu = nn.Sequential(
                nn.Linear(config.codevector_dim, 2 * config.codevector_dim),
                nn.GLU(dim=-1),
            )

        feature_ds_rate = np.prod(config.conv_stride)
        self.feat2tar_ratio = config.label_rate * feature_ds_rate / config.sample_rate
        self.logits_temperature = config.logits_temperature

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wavnode.feature_encoder._freeze_parameters()

    def _compute_pred(self, proj_x: torch.Tensor, label_embs: torch.Tensor) -> torch.Tensor:
        """
        Compute class logits for HuBERT-style masked prediction.

        Args:
            proj_x: Tensor of shape (N_mask, D)
            label_embs: Tensor of shape (K, D)
        Returns:
            logits: Tensor of shape (N_mask, K)
        """
        if getattr(self.config, "use_target_glu", False):
            label_embs = self.target_glu(label_embs)

        if getattr(self.config, "use_cosine_similarity", True):
            proj_x = F.normalize(proj_x, dim=-1)
            label_embs = F.normalize(label_embs, dim=-1)
            logits = torch.matmul(proj_x, label_embs.transpose(0, 1))
        else:
            logits = torch.matmul(proj_x, label_embs.transpose(0, 1))

        logits = logits / self.logits_temperature
        return logits

    def _forward_targets(
        self,
        features: torch.Tensor,
        target_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Trim features to ensure labels exist and then get aligned labels
        feat_tsz = features.size(2)
        targ_tsz = min([t.size(1) for t in target_list])
        if self.feat2tar_ratio * feat_tsz > targ_tsz:
            feat_tsz = int(targ_tsz / self.feat2tar_ratio)
            features = features[..., :feat_tsz]
        target_inds = torch.arange(feat_tsz, device=features.device).float() * self.feat2tar_ratio
        target_list = [t[:, target_inds.long()] for t in target_list]
        return features, target_list

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, WavNodeForPretrainingOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, raw_sequence_length = input_values.shape

        # Prepare feature-domain mask if not provided
        if mask_time_indices is None and self.training and self.config.mask_time_prob > 0:
            input_lengths = (
                attention_mask.sum(-1)
                if attention_mask is not None
                else torch.full((batch_size,), raw_sequence_length, device=input_values.device, dtype=torch.long)
            )
            feat_lengths = self.wavnode._get_feat_extract_output_lengths(input_lengths)
            max_feat_len = int(feat_lengths.max().item())
            feat_attn_mask = None
            if attention_mask is not None:
                feat_attn_mask = self.wavnode._get_feature_vector_attention_mask(max_feat_len, attention_mask)
            mask_np = _compute_mask_indices(
                (batch_size, max_feat_len),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=feat_attn_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(mask_np, device=input_values.device, dtype=torch.bool)
        elif mask_time_indices is not None:
            mask_time_indices = mask_time_indices.to(torch.bool)

        # Forward backbone with mask indices
        outputs = self.wavnode(
            input_values,
            attention_mask=attention_mask,
            mask_time_indices=mask_time_indices,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]  # (B, T_feat, H)
        attentions = outputs.attentions if return_dict else None

        # Build feature padding mask
        if attention_mask is not None:
            padding_mask = self.wavnode._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
        else:
            padding_mask = torch.ones(hidden_states.shape[:2], device=hidden_states.device, dtype=torch.bool)

        # Align labels to feature frames if provided
        aligned_labels = None
        if labels is not None:
            features_nc_t, [aligned_labels] = self._forward_targets(hidden_states.transpose(1, 2), [labels])
            new_len = features_nc_t.size(2)
            if hidden_states.size(1) != new_len:
                hidden_states = hidden_states[:, :new_len]
                padding_mask = padding_mask[:, :new_len]
                if mask_time_indices is not None:
                    mask_time_indices = mask_time_indices[:, :new_len]

        # Determine masked positions
        masked_indices = None
        if mask_time_indices is not None:
            masked_indices = mask_time_indices & padding_mask
        else:
            # if no mask provided and skip_nomask is True, do not compute loss
            masked_indices = padding_mask if not getattr(self.config, "skip_nomask", True) else None

        # Project and compute logits/loss
        projected_states = self.final_proj(hidden_states)

        loss = None
        if aligned_labels is not None:
            # if there are masked indices, compute the loss
            if masked_indices is not None and masked_indices.any():
                proj_x_m = projected_states[masked_indices]
                logits_m = self._compute_pred(proj_x_m, self.label_embs)

                targets_m = aligned_labels[masked_indices]
                loss = F.cross_entropy(logits_m, targets_m.long())
            else:
                # no masked indices, set loss to 0
                loss = torch.tensor(0.0, device=projected_states.device, dtype=projected_states.dtype)

        if not return_dict:
            return (loss, projected_states) + outputs[2:]

        return WavNodeForPretrainingOutput(
            loss=loss,
            projected_states=projected_states,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            nce_loss=None,
            stiffness_loss=None,
        )


__all__ = [
    "WavNodeEncoderLayer",
    "WavNodeEncoder",
    "WavNodeModel",
    "WavNodePreTrainedModel"
]
