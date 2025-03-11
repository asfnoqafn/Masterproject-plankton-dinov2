# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

from typing import Callable, Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
        gray_scale: int = 0,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding
        self.gray_scale = gray_scale

        if self.gray_scale == 1:
            self.channel_adapt = nn.Conv2d(in_channels=1,out_channels=3,kernel_size=1, stride=1, bias=True)
            self.proj = nn.Conv2d(
                3,
                embed_dim,
                kernel_size=patch_HW,
                stride=patch_HW,
            )
        if self.gray_scale == 2:
            self.proj = nn.Conv2d(
                1,
                embed_dim,
                kernel_size=patch_HW,
                stride=patch_HW,
            )
        if self.gray_scale == 0:
            self.proj = nn.Conv2d(
                3,
                embed_dim,
                kernel_size=patch_HW,
                stride=patch_HW,
            )

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        if self.gray_scale == 1:
            x = self.channel_adapt(x)
            x = self.proj(x)  # B D sqrt(np) sqrt(np)
        if self.gray_scale == 2:
            x = self.proj(x)
        if self.gray_scale == 0:
            x = self.proj(x)

        H_p, W_p = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B np D
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H_p, W_p, self.embed_dim)  # B H_p W_p D
        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
