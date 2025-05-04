import math
import random
from typing import Optional, Sequence, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resized_crop, get_dimensions
from torchvision.transforms import RandomResizedCrop

import matplotlib.pyplot as plt
import warnings


class RandomResizedCropForeground(nn.Module):
    """
    Random resize crop focused on foreground regions detected via various mask methods.
    Selects a random foreground pixel and ensures the crop includes it.
    Attempts to find a crop in one try based on scale and ratio.
    If not possible, falls back to the largest possible crop centered on the
    foreground pixel respecting the chosen aspect ratio.
    Currently only supports Sobel filter for foreground detection.
    Designed for grayscale images.
    """
    def __init__(
        self,
        size: Sequence[int],
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        percentile: float = 50.0,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: Optional[bool] = False,
        num_attempts: int = 10,
    ):
        super().__init__()
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        # print(f"size: {self.size}")
        # print(f"scale: {scale}")
        # print(f"ratio: {ratio}")

        if not isinstance(scale, Sequence) or len(scale) != 2:
             raise ValueError("scale should be a sequence of length 2.")
        if not isinstance(ratio, Sequence) or len(ratio) != 2:
            raise ValueError("ratio should be a sequence of length 2.")
        if scale[0] > scale[1]:
            warnings.warn(f"scale range [{scale[0]}, {scale[1]}] is reversed.")
        if ratio[0] > ratio[1]:
            warnings.warn(f"ratio range [{ratio[0]}, {ratio[1]}] is reversed.")

        self.scale = scale
        self.ratio = ratio
        self.percentile = percentile
        self.interpolation = interpolation
        self.antialias = antialias

    def _sobel_mask(self, img: torch.Tensor) -> torch.Tensor:
        if img.ndim == 2:
            img = img.unsqueeze(0)
        if img.ndim == 3:
            img = img.unsqueeze(0)

        if not torch.is_floating_point(img):
             img = img.float()

        g_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=img.dtype, device=img.device)
        g_y = g_x.t()
        kernels = torch.stack([g_x, g_y]).unsqueeze(1) # 2x1x3x3 (out_channels, in_channels/groups, kH, kW)

        grad = F.conv2d(img, kernels, padding=1)
        mag = torch.sqrt(torch.sum(grad**2, dim=1))

        numel = mag.numel()
        k = int((1 - self.percentile / 100.0) * numel)
        k = max(1, min(k, numel))

        try:
            mag_flat = mag.flatten().float()
            thresh = mag_flat.kthvalue(k).values
        except RuntimeError as e:
            print(f"Warning: kthvalue failed ({e}). Falling back to quantile (potentially less precise).")

            thresh = torch.quantile(mag.float(), (1 - self.percentile / 100.0))

        mask = mag > thresh # (1, H, W)
        mask = mask.squeeze(0) # Remove batch dim -> (H, W)
        return mask


    def _get_mask(self, img: torch.Tensor) -> torch.Tensor:
        input_img = img if img.ndim >=3 else img.unsqueeze(0)

        mask = self._sobel_mask(input_img)

        mask = mask.float().unsqueeze(0).unsqueeze(0) # B C H W = 1 1 H W
        # Dilation equivalent using max_pool
        mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        # Erosion equivalent using min_pool (implemented via negative max_pool)
        mask = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
        mask = mask.squeeze().bool() # H W
        return mask

    @staticmethod
    def get_params_foreground(
        img: torch.Tensor,
        scale: Tuple[float, float],
        ratio: Tuple[float, float],
        foreground_pixels: Tuple[torch.Tensor, torch.Tensor]
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Get parameters for a crop ensuring a foreground pixel is included, in one attempt.
        Tries to meet scale and ratio. If impossible, falls back to largest crop
        centered on the pixel, maintaining the *chosen* aspect ratio.
        Returns None if foreground_pixels is empty.
        """
        _, H, W = get_dimensions(img)
        area = H * W
        ys, xs = foreground_pixels
        num_foreground = len(ys)

        if num_foreground == 0:
            warnings.warn("No foreground pixels found. Falling back to standard random crop.")
            return None

        target_area = area * random.uniform(scale[0], scale[1])
        log_ratio = torch.log(torch.tensor(ratio))
        target_aspect_ratio = math.exp(random.uniform(log_ratio[0].item(), log_ratio[1].item()))

        h_cand = int(round(math.sqrt(target_area / target_aspect_ratio)))
        w_cand = int(round(math.sqrt(target_area * target_aspect_ratio)))
        
        h_buffer = h_cand // 8
        w_buffer = w_cand // 8

        # rand_idx = random.randrange(num_foreground)
        # center_y, center_x = ys[rand_idx].item(), xs[rand_idx].item()
        valid_idx = ((ys > h_buffer) & (ys < H - h_buffer) & (xs > w_buffer) & (xs < W - w_buffer)).nonzero(as_tuple=False).squeeze(1)

        if len(valid_idx) == 0:
            #fallback to all pixels
            rand_idx = random.randrange(num_foreground)
        else:
            rand_idx = valid_idx[random.randrange(len(valid_idx))].item()

        center_y, center_x = ys[rand_idx].item(), xs[rand_idx].item()


        i_min = max(0, center_y - h_cand + 1)
        i_max = min(H - h_cand, center_y)

        j_min = max(0, center_x - w_cand + 1)
        j_max = min(W - w_cand, center_x)


        if i_min <= i_max and j_min <= j_max:
            # center the crop on (center_y, center_x)
            i = center_y - h_cand // 2
            j = center_x - w_cand // 2
            i = max(0, min(i, H - h_cand))
            j = max(0, min(j, W - w_cand))
            return i, j, h_cand, w_cand

        # --- Fallback Logic ---
        # Executed if:
        # a) w_cand or h_cand were invalid (e.g., 0 or > image dims)
        # b) The valid placement check (i_min <= i_max and j_min <= j_max) failed


        img_ratio = float(W) / float(H)

        if img_ratio < target_aspect_ratio:
            w_fallback = W
            h_fallback = int(round(w_fallback / target_aspect_ratio))
            if h_fallback > H:
                h_fallback = H
                w_fallback = int(round(h_fallback * target_aspect_ratio))
        else:
            h_fallback = H
            w_fallback = int(round(h_fallback * target_aspect_ratio))
            if w_fallback > W:
                w_fallback = W
                h_fallback = int(round(w_fallback / target_aspect_ratio))


        i_fallback = max(0, min(H - h_fallback, center_y - h_fallback // 2))
        j_fallback = max(0, min(W - w_fallback, center_x - w_fallback // 2))

        assert 0 <= i_fallback <= H - h_fallback, f"Fallback i calculation error: i={i_fallback}, H={H}, h={h_fallback}"
        assert 0 <= j_fallback <= W - w_fallback, f"Fallback j calculation error: j={j_fallback}, W={W}, w={w_fallback}"

        return i_fallback, j_fallback, h_fallback, w_fallback


    def forward(self, img: torch.Tensor) -> torch.Tensor:

        mask = self._get_mask(img)
        foreground_pixels = mask.nonzero(as_tuple=True)

        i, j, h, w = self.get_params_foreground(
            img, self.scale, self.ratio, foreground_pixels
        )
        if i is None:
            transform = RandomResizedCrop(
                size=self.size,
                scale=self.scale,
                ratio=self.ratio,
                interpolation=self.interpolation,
                antialias=self.antialias,
            )
            print("Falling back to standard RandomResizedCrop.")
            return transform(img)
        crop = resized_crop(img, i, j, h, w, list(self.size), self.interpolation, antialias=self.antialias)
        return crop
