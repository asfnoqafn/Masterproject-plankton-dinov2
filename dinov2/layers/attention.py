# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

import torch
from einops import rearrange
from torch import Tensor, nn

from dinov2.utils.utils import exists

logger = logging.getLogger("dinov2")


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import (
            memory_efficient_attention,
            unbind,
        )

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_pytorch_attn: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.attn_drop_p = attn_drop
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_pytorch_attn = use_pytorch_attn

    def forward(self, x: Tensor) -> Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)  # 3 b h n d
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        if self.use_pytorch_attn:  # TODO: implement attn mask AND args to use pytorch attn
            x = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,  # TODO: implement attn mask
                dropout_p=self.attn_drop_p,
            )
        else:
            attn = q @ k.transpose(-2, -1)
            # b h n n

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            # b h n n

            x = (attn @ v).transpose(1, 2).reshape(B, N, D)
            x = self.proj(x)
        x = self.proj_drop(x)
        # b n d
        return x


class MemEffAttention(Attention):
    def masked_mem_eff_attn(
        self,
        query,
        key,
        value,
        attn_bias=None,
        attn_mask=None,
        mask=None,
    ):
        scale = 1 / query.shape[-1] ** 0.5
        query = query * scale
        attn = query @ key.transpose(-2, -1)

        # b n n_heads c//n_heads = b n h d
        if exists(mask):
            mask = rearrange(mask, "b j -> b 1 1 j")
            attn = attn.masked_fill(~mask, -torch.finfo(attn.dtype).max)

        if exists(attn_mask):
            attn = attn.masked_fill(~attn_mask, -torch.finfo(attn.dtype).max)

        if attn_bias is not None:
            attn = attn + attn_bias  # invalid types for +: 'Tensor' and 'BlockDiagonalMask'

        attn = attn.softmax(-1)
        # attn = F.dropout(attn, p) # p = 0.0 per default
        return attn @ value

    def forward(self, x: Tensor, attn_bias=None, attn_mask=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            assert attn_bias is None, "xFormers is required for nested tensors usage"
            return super().forward(x, return_attn)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        if exists(attn_mask):
            q, k, v = (
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
            )
            # b n h d -> b h n d, h=self.heads, for our own impl of masked attention
            x = self.masked_mem_eff_attn(
                q,
                k,
                v,
                attn_bias=attn_bias,
                attn_mask=attn_mask,
                mask=None,
            )
        else:
            x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
