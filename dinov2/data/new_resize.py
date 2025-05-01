import math
import random
from typing import Optional, Sequence, Tuple, List

import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resized_crop, get_dimensions, to_pil_image
from torchvision.transforms import RandomResizedCrop
from torchvision.io import read_image, ImageReadMode
import matplotlib.pyplot as plt
import warnings



class RandomResizedCropForeground(nn.Module):
    """
    Random resize crop focused on foreground regions detected via various mask methods.
    Selects a random foreground pixel and ensures the crop includes it.
    currently only supports sobel filter
    """
    def __init__(
        self,
        size: Sequence[int],
        scale  = [0.08, 1.0],
        ratio = [3.0 / 4.0, 4.0 / 3.0],
        percentile: float = 50.0,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: Optional[bool] = True,
        num_attempts: int = 10, 
    ):
        super().__init__()
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        print(f"size: {self.size}")
        print(f"scale: {scale}")
        print(f"ratio: {ratio}")

        self.scale = scale
        self.ratio = ratio
        self.percentile = percentile
        self.interpolation = interpolation
        self.antialias = antialias
        self.num_attempts = num_attempts


    def _sobel_mask(self, img: torch.Tensor) -> torch.Tensor:
        g_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=img.dtype, device=img.device)
        g_y = g_x.t()
        kernels = torch.stack([g_x, g_y]).unsqueeze(1)  # 2×1×3×3
        grad = F.conv2d(img, kernels, padding=1) # B x 2 x H x W
        mag = torch.sqrt(torch.sum(grad**2, dim=0)) # B x H x W
        k = int((1 - self.percentile / 100.0) * mag.numel())
        k = max(1, min(k, mag.numel()))

        try:
            thresh = mag.flatten().kthvalue(k).values
        except RuntimeError as e:
             print(f"Warning: kthvalue failed ({e}). Falling back to quantile (potentially less precise).")
             thresh = torch.quantile(mag, (1 - self.percentile / 100.0))

        mask = mag > thresh # H x W
        return mask

    def _get_mask(self, img: torch.Tensor) -> torch.Tensor:
        mask = self._sobel_mask(img)
        mask = mask.float().unsqueeze(0).unsqueeze(0) # B C H W = 1 1 H W
        mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        mask = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
        mask = mask.squeeze().bool() # H W
        return mask


    @staticmethod
    def get_params(img: torch.Tensor, scale: Tuple[float, float], ratio: Tuple[float, float], num_attempts: int = 10) -> Tuple[int, int, int, int]:
        """Get parameters for a standard RandomResizedCrop.
        Copied logic from torchvision.transforms.RandomResizedCrop.get_params"""
        _, H, W = get_dimensions(img)
        area = H * W

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(num_attempts):
            target_area = area * random.uniform(scale[0], scale[1])
            aspect_ratio = math.exp(random.uniform(log_ratio[0].item(), log_ratio[1].item()))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= W and 0 < h <= H:
                i = random.randint(0, H - h)
                j = random.randint(0, W - w)
                return i, j, h, w

        # fallback
        print("Warning: Failed to find valid crop parameters after multiple attempts. Falling back to central crop.")
        in_ratio = float(W) / float(H)
        if in_ratio < min(ratio):
            w = W
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = H
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = W
            h = H
        i = (H - h) // 2
        j = (W - w) // 2
        return i, j, h, w


    @staticmethod
    def get_params_foreground(
        img: torch.Tensor,
        scale: Tuple[float, float],
        ratio: Tuple[float, float],
        foreground_pixels: Tuple[torch.Tensor, torch.Tensor],
        num_attempts: int = 10
    ) -> Optional[Tuple[int, int, int, int]]:
        """Get parameters for a crop ensuring a foreground pixel is included."""
        _, H, W = get_dimensions(img)
        area = H * W
        ys, xs = foreground_pixels
        num_foreground = len(ys)


        log_ratio = torch.log(torch.tensor(ratio))

        for _ in range(num_attempts):
            target_area = area * random.uniform(scale[0], scale[1])
            aspect_ratio = math.exp(random.uniform(log_ratio[0].item(), log_ratio[1].item()))

            w_cand = int(round(math.sqrt(target_area * aspect_ratio)))
            h_cand = int(round(math.sqrt(target_area / aspect_ratio)))

            if not (0 < w_cand <= W and 0 < h_cand <= H):
                continue 

            rand_idx = random.randrange(num_foreground)
            center_y, center_x = ys[rand_idx].item(), xs[rand_idx].item()

            i_min = max(0, center_y - h_cand + 1)
            i_max = min(H - h_cand, center_y)

            j_min = max(0, center_x - w_cand + 1)
            j_max = min(W - w_cand, center_x)

            if i_min > i_max or j_min > j_max:
                continue

            i = random.randint(i_min, i_max)
            j = random.randint(j_min, j_max)
            return i, j, h_cand, w_cand

        assert False, "Failed to find valid crop parameters after multiple attempts."


    def forward(self, img: torch.Tensor) -> torch.Tensor:
        mask = self._get_mask(img)
        foreground_pixels = mask.nonzero(as_tuple=True)
        crop_params = None
        
        if len(foreground_pixels[0]) > 0:
            crop_params = self.get_params_foreground(
                img, self.scale, self.ratio, foreground_pixels, self.num_attempts
            )

        if crop_params is None:
            crop_params = self.get_params(img, self.scale, self.ratio, self.num_attempts)

        i, j, h, w = crop_params
        return resized_crop(img, i, j, h, w, list(self.size), self.interpolation, antialias=self.antialias)