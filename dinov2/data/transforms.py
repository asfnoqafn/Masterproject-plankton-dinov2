# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Sequence, Union

import torch
from kornia import augmentation
from torchvision.transforms import v2


class KorniaGaussianBlur(augmentation.RandomGaussianBlur):
    """
    Apply Gaussian Blur to the a Tensor using Kornia.
    """

    def __init__(
        self,
        *,
        p: float = 0.5,
        radius_min: float = 0.1,
        radius_max: float = 2.0,
    ):
        super().__init__(
            kernel_size=9,
            sigma=(radius_min, radius_max),
            same_on_batch=False,
            p=p,
        )


class GaussianBlur(v2.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(
        self,
        *,
        p: float = 0.5,
        radius_min: float = 0.1,
        radius_max: float = 2.0,
    ):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = v2.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)


class MaybeToTensor(v2.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
WHOI_DEFAULT_MEAN = 0.68622917
WHOI_DEFAULT_STD = 0.10176649


def make_normalize_transform(
    mean: Sequence[float] = [WHOI_DEFAULT_MEAN],
    std: Sequence[float] = [WHOI_DEFAULT_STD],
    use_kornia=False,
) -> Union[v2.Normalize, augmentation.Normalize]:
    if use_kornia:
        return augmentation.Normalize(mean, std, p=1.0, keepdim=False)
    else:
        return v2.Normalize(mean=mean, std=std)


# This roughly matches torchvision's preset for classification training:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L6-L44
def make_classification_train_transform(
    *,
    crop_size: int = 224,
    interpolation=v2.InterpolationMode.BICUBIC,
    hflip_prob: float = 0.5,
    mean: Sequence[float] = [WHOI_DEFAULT_MEAN],
    std: Sequence[float] = [WHOI_DEFAULT_STD],
):
    transforms_list = [
        v2.RandomResizedCrop(
            crop_size,
            interpolation=interpolation,
            antialias=True,
        )
    ]
    if hflip_prob > 0.0:
        transforms_list.append(v2.RandomHorizontalFlip(hflip_prob))
    transforms_list.extend(
        [
            MaybeToTensor(),
            make_normalize_transform(mean=mean, std=std),
        ]
    )
    return v2.Compose(transforms_list)


# This matches (roughly) torchvision's preset for classification evaluation:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L47-L69
def make_classification_eval_transform(
    *,
    resize_size: int = 256,
    interpolation=v2.InterpolationMode.BICUBIC,
    crop_size: int = 224,
    mean: Sequence[float] = [WHOI_DEFAULT_MEAN],
    std: Sequence[float] = [WHOI_DEFAULT_STD],
) -> v2.Compose:
    transforms_list = [
        v2.Resize(223,max_size= 224, antialias=True),
        v2.Pad(112, fill=255, padding_mode='constant'),
        v2.CenterCrop(crop_size),
        MaybeToTensor(),
        make_normalize_transform(mean=mean, std=std),
    ]
    return v2.Compose(transforms_list)