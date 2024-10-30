# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import sys
from enum import Enum

import numpy as np
import torch
from einops import rearrange
from kornia import augmentation
from kornia.augmentation.container import (
    AugmentationSequential,
)
from kornia.constants import Resample
from skimage.segmentation import (
    felzenszwalb,
    quickshift,
    slic,
)
from torchvision.ops import masks_to_boxes
from torchvision.transforms import v2
from torchvision.transforms.functional import (
    InterpolationMode,
)

from dinov2.utils.utils import exists

from .transforms import (
    GaussianBlur,
    KorniaGaussianBlur,
    MaybeToTensor,
    make_normalize_transform,
)

logger = logging.getLogger("dinov2")
MAX_PATCHES = 1000
MIN_NB_PATCHES_IN_CROP = 10
BASE_LC_NB = 8
PATCH_THRESHOLD = 1e-5
STD_CROP_SIZE = 98


class SegmentationAlgo(Enum):
    FELZENSWALB = "felzenswalb"
    SLIC = "slic"


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        use_kornia=False,
        use_native_res=False,
        patch_size=14,
        do_seg_crops=None,
        do_multi_channel=False,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.use_kornia = use_kornia
        self.use_native_res = use_native_res
        if self.use_native_res:
            self.global_crops_area = self.global_crops_size * self.global_crops_size
        self.patch_size = patch_size
        self.do_seg_crops = do_seg_crops
        self.do_multi_channel = do_multi_channel

        self.patch_maxpool_op = torch.nn.MaxPool2d(
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
            dilation=1,
            return_indices=False,
        )

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        ######## Kornia
        if self.use_kornia:
            global_crops_size = (
                global_crops_size,
                global_crops_size,
            )
            local_crops_size = (
                local_crops_size,
                local_crops_size,
            )

            if self.use_native_res:
                self.random_crop = augmentation.RandomCrop(
                    global_crops_size,
                    resample=Resample.BILINEAR.name,
                    same_on_batch=False,
                    p=1.0,
                    keepdim=False,
                )
                self.geometric_augmentation_global = augmentation.RandomHorizontalFlip(p=0.5, p_batch=1.0)
                self.geometric_augmentation_local = AugmentationSequential(
                    augmentation.RandomHorizontalFlip(p=0.5, p_batch=1.0),
                    data_keys=["input", "input"],
                )

                self.std_augmentation_local = AugmentationSequential(
                    augmentation.RandomResizedCrop(
                        local_crops_size,
                        scale=local_crops_scale,
                        resample=Resample.BICUBIC.name,
                        same_on_batch=False,
                    ),
                    augmentation.RandomHorizontalFlip(p=0.5, p_batch=1.0),
                )
            else:
                self.geometric_augmentation_global = AugmentationSequential(
                    augmentation.RandomResizedCrop(
                        global_crops_size,
                        scale=global_crops_scale,
                        resample=Resample.BICUBIC.name,
                        same_on_batch=False,
                    ),
                    augmentation.RandomHorizontalFlip(p=0.5, p_batch=1.0),
                )
                self.geometric_augmentation_local = AugmentationSequential(
                    augmentation.RandomResizedCrop(
                        local_crops_size,
                        scale=local_crops_scale,
                        resample=Resample.BICUBIC.name,
                        same_on_batch=False,
                    ),
                    augmentation.RandomHorizontalFlip(p=0.5, p_batch=1.0),
                )

            # color distorsions / blurring
            color_jittering = AugmentationSequential(
                augmentation.ColorJiggle(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.2,
                    hue=0.1,
                    same_on_batch=False,
                    p=0.8,
                ),
                augmentation.RandomGrayscale(p=0.2),
            )

            global_transfo1_extra = KorniaGaussianBlur(p=1.0)

            global_transfo2_extra = AugmentationSequential(
                KorniaGaussianBlur(p=0.1),
                augmentation.RandomSolarize(
                    thresholds=0.5,
                    additions=0.1,
                    same_on_batch=False,
                    p=0.2,
                ),
            )

            local_transfo_extra = KorniaGaussianBlur(p=1.0)

            # normalization
            self.normalize = make_normalize_transform(use_kornia=True)

            if not self.do_multi_channel:
                self.global_transfo1 = AugmentationSequential(
                    color_jittering,
                    global_transfo1_extra,
                    self.normalize,
                )
                self.global_transfo2 = AugmentationSequential(
                    color_jittering,
                    global_transfo2_extra,
                    self.normalize,
                )
                self.local_transfo = AugmentationSequential(
                    color_jittering,
                    local_transfo_extra,
                    self.normalize,
                )
            else:
                self.global_transfo1 = AugmentationSequential(
                    global_transfo1_extra,
                    self.normalize,
                )
                self.global_transfo2 = AugmentationSequential(
                    global_transfo2_extra,
                    self.normalize,
                )
                self.local_transfo = AugmentationSequential(local_transfo_extra, self.normalize)

        ######## TORCHVISION
        else:
            # random resized crop and flip
            self.geometric_augmentation_global = v2.Compose(
                [
                    v2.RandomResizedCrop(
                        global_crops_size,
                        scale=global_crops_scale,
                        interpolation=v2.InterpolationMode.BICUBIC,
                        antialias=True,
                    ),
                    v2.RandomHorizontalFlip(p=0.5),
                ]
            )

            self.geometric_augmentation_local = v2.Compose(
                [
                    v2.RandomResizedCrop(
                        local_crops_size,
                        scale=local_crops_scale,
                        interpolation=v2.InterpolationMode.BICUBIC,
                        antialias=True,
                    ),
                    v2.RandomHorizontalFlip(p=0.5),
                ]
            )

            # color distorsions / blurring
            color_jittering = v2.Compose(
                [
                    v2.RandomApply(
                        [
                            v2.ColorJitter(
                                brightness=0.4,
                                contrast=0.4,
                                saturation=0.2,
                                hue=0.1,
                            )
                        ],
                        p=0.8,
                    ),
                    v2.RandomGrayscale(p=0.2),
                ]
            )

            global_transfo1_extra = GaussianBlur(p=1.0)

            solarize_threshold = 0.5
            solarize_threshold = 0.5
            global_transfo2_extra = v2.Compose(
                [
                    GaussianBlur(p=0.1),
                    v2.RandomSolarize(threshold=solarize_threshold, p=0.2),
                ]
            )

            local_transfo_extra = GaussianBlur(p=0.5)

            # normalization
            self.normalize = v2.Compose(
                [
                    MaybeToTensor(),
                    make_normalize_transform(),
                ]
            )

            self.global_transfo1 = v2.Compose(
                [
                    color_jittering,
                    global_transfo1_extra,
                    self.normalize,
                ]
            )
            self.global_transfo2 = v2.Compose(
                [
                    color_jittering,
                    global_transfo2_extra,
                    self.normalize,
                ]
            )
            self.local_transfo = v2.Compose(
                [
                    color_jittering,
                    local_transfo_extra,
                    self.normalize,
                ]
            )

    def round_up_patch_size(self, crop_len: int):
        if crop_len % self.patch_size == 0:
            return crop_len
        return crop_len + (self.patch_size - crop_len % self.patch_size)

    def round_down_patch_size(self, crop_len: int):
        if crop_len % self.patch_size == 0:
            return crop_len
        return crop_len - crop_len % self.patch_size

    def make_rectangle_crop(self, image):
        img_global_list = []
        for single_image in image:
            if single_image.size(-1) > self.global_crops_size and single_image.size(-2) > self.global_crops_size:
                single_image = self.random_crop(single_image).squeeze()
            else:
                H, W = single_image.size(-2), single_image.size(-1)
                min_dim_idx = np.argmin((H, W))
                crop_0 = (H, W)[min_dim_idx]
                crop_1 = self.global_crops_area // crop_0

                if min_dim_idx == 0:
                    crop_y = crop_0
                    crop_x = crop_1
                else:
                    crop_y = crop_1
                    crop_x = crop_0
                crop_y = self.round_up_patch_size(crop_y)
                crop_x = self.round_up_patch_size(crop_x)

                if single_image.size(-2) - crop_x < 0:
                    crop_x = single_image.size(-2)
                if single_image.size(-1) - crop_y < 0:
                    crop_y = single_image.size(-1)

                offset_x, offset_y = 0, 0
                if single_image.size(-2) - crop_x:
                    offset_x = np.random.randint(single_image.size(-2) - crop_x)

                if single_image.size(-1) - crop_y:
                    offset_y = np.random.randint(single_image.size(-1) - crop_y)

                single_image = single_image[
                    offset_x : offset_x + crop_x,
                    offset_y : offset_y + crop_y,
                ]
            img_global_list.append(single_image)

        return torch.stack(img_global_list)

    def make_seg_crops(self, image, seg_algo):
        # image : C H W
        if seg_algo == SegmentationAlgo.FELZENSWALB.value:
            segments = felzenszwalb(
                image.permute((1, 2, 0)),
                scale=300,
                sigma=0.5,
                min_size=1000,
            )
        elif seg_algo == SegmentationAlgo.SLIC.value:
            # n_segments = max(int((h * w) / (STD_CROP_SIZE * STD_CROP_SIZE)), 3)
            n_segments = self.local_crops_number
            segments = slic(
                image.permute((1, 2, 0)),
                n_segments=n_segments,
                compactness=10,
                sigma=0,
                start_label=1,
            )
        else:
            raise NotImplementedError(f"Specified segmentation algorithm {seg_algo} not implemented")

        def seg_to_patched_seg(segments, image_gray):
            segments = segments / (segments.max() + 1e-5)
            conv_input = torch.tensor(segments[None, :, :])

            pooled_seg = self.patch_maxpool_op(conv_input)

            resized_masks = pooled_seg.repeat_interleave(self.patch_size, dim=1).repeat_interleave(
                self.patch_size, dim=2
            )  # back to orig shape
            # assert torch.all(resized_masks_alt == resized_masks)

            resized_masks = resized_masks[0, :, :]
            nb_seg = len(np.unique(resized_masks))

            # map tensor to continuous int values
            def map_to_cont_int_vals(input: torch.Tensor, max_val: int):
                input_int = (input * max_val).to(torch.int32)
                unique_vals = torch.unique(input_int)
                lt = torch.full((int(input_int.max()) + 1,), -1)
                lt[unique_vals] = torch.arange(len(unique_vals))
                output = lt[input_int]
                return output

            resized_masks_int = map_to_cont_int_vals(resized_masks, max_val=nb_seg)
            pooled_seg_int = map_to_cont_int_vals(pooled_seg, max_val=nb_seg)
            # update nb_seg
            nb_seg = len(np.unique(resized_masks_int))
            # First, create a mask for each segment
            masks, patches_pos_list = [], []
            for mask_idx in range(min(nb_seg, self.local_crops_number)):
                matching_mask = resized_masks_int == mask_idx
                if torch.numel(matching_mask > 0) > (
                    self.patch_size * self.patch_size * MIN_NB_PATCHES_IN_CROP
                ):  # possible that some masks are 0
                    masks.append(resized_masks_int == mask_idx)
                    patch_pos_list = torch.where(pooled_seg_int[0, :, :].ravel() == mask_idx)
                    patches_pos_list.append(patch_pos_list)

            masks = torch.stack(masks)
            # Then, find the bounding boxes for each segment
            bboxes = masks_to_boxes(masks)  # (x1, y1, x2, y2)

            bounded_images, bounded_masks = [], []
            for mask_idx in range(min(nb_seg, self.local_crops_number)):
                mask = masks[mask_idx, :, :]
                bbox = bboxes[mask_idx].to(torch.int32)

                # first bbox crop, then segment
                bounded_mask = mask[
                    bbox[1] : bbox[3] + 1,
                    bbox[0] : bbox[2] + 1,
                ]
                bounded_image = image_gray[
                    bbox[1] : bbox[3] + 1,
                    bbox[0] : bbox[2] + 1,
                ]
                bounded_images.append(bounded_image)
                bounded_masks.append(bounded_mask)

            return (
                bounded_images,
                bounded_masks,
                bboxes,
                patches_pos_list,
                resized_masks_int,
            )

        return seg_to_patched_seg(segments, image[0, :, :])

    def pad_to_patch_mutiple(self, image):
        """
        Pads the input image to a multiple of the patch size.
        Args:
            image (torch.Tensor): The input image to be padded. C H W or H W
        Returns:
            torch.Tensor: The padded image.
        Note:
            This function calculates the padding required for the image to be a multiple of the patch size.
            It then pads the image using the `torch.nn.functional.pad` function with the calculated padding values.
            The padding mode is set to "replicate".
        """
        # Can also make a version that pads down
        pad_y = self.patch_size - image.shape[-1] % self.patch_size
        pad_x = self.patch_size - image.shape[-2] % self.patch_size

        pad_x_l, pad_x_r = (
            pad_x // 2,
            pad_x // 2 + pad_x % 2,
        )
        pad_y_l, pad_y_r = (
            pad_y // 2,
            pad_y // 2 + pad_y % 2,
        )

        return torch.nn.functional.pad(
            image,
            pad=(
                pad_y_l,
                pad_y_r,
                pad_x_l,
                pad_x_r,
            ),
            mode="replicate",
        )

    def crop_to_patches(self, crop):
        # B is usually 1 here
        if len(crop.size()) == 3:
            crop = crop[None, :, :, :]
        b, c, h, w = crop.size()

        patches = crop.unfold(-1, self.patch_size, self.patch_size).unfold(-3, self.patch_size, self.patch_size)
        patches = rearrange(
            patches,
            "b c n1 n2 p1 p2 -> b c (n1 n2 p2) p1",
            b=b,
            c=c,
            p1=self.patch_size,
            p2=self.patch_size,
        )
        return patches

    def make_std_params(self):
        patch_nb_per_crop = (self.local_crops_size * self.local_crops_size) / (self.patch_size * self.patch_size)
        crop_len = torch.ones(self.local_crops_number, dtype=torch.long) * patch_nb_per_crop
        filtered_patch_pos_list = torch.tile(
            torch.arange(patch_nb_per_crop),
            (self.local_crops_number, 1),
        )
        filtered_bboxes = torch.tile(
            torch.tensor(
                [
                    0,
                    0,
                    self.local_crops_size,
                    self.local_crops_size,
                ]
            ),
            (self.local_crops_number, 1),
        )
        return (
            crop_len,
            filtered_patch_pos_list,
            filtered_bboxes,
        )

    def select_and_concat_nonzero_patches(
        self,
        local_crops,
        masks,
        image,
        nb_gc_patches=0,
        bboxes=None,
        do_drop_patches=False,
    ):
        # Select non-zero 14x14 patches from local_crops and flat concat
        # Input: list N N_p [c p p]
        # Output: C P NxP
        tot_patches = nb_gc_patches
        list_flat_patches, crop_len_list = [], []
        filtered_patch_pos_list, filtered_bboxes = [], []
        for crop, mask, bbox in zip(local_crops, masks, bboxes):
            mask_patches = self.crop_to_patches(mask).squeeze()  # c (n p) p
            img_patches = self.crop_to_patches(crop).squeeze()  # c (n p) p
            mask_patches = torch.chunk(
                mask_patches,
                chunks=mask_patches.shape[1] // self.patch_size,
                dim=1,
            )
            img_patches = torch.chunk(
                img_patches,
                chunks=img_patches.shape[1] // self.patch_size,
                dim=1,
            )
            # lists are n [c p p]
            # local_patch_pos = local_patch_pos[0]
            if (tot_patches + len(img_patches)) >= MAX_PATCHES:
                if len(crop_len_list) < 1:  # if seg fails, revert to std patching
                    print("revert to std patching")

                    def std_patching(image):
                        local_crop = [
                            self.crop_to_patches(
                                self.local_transfo(self.std_augmentation_local(image))
                            ).squeeze()  # c (n p) p
                            for _ in range(self.local_crops_number)  # 8 is std local crops nb
                        ]
                        return torch.cat(local_crop, dim=1)  # c (n p) p

                    flat_patches = std_patching(image)
                    (
                        crop_len,
                        filtered_patch_pos_list,
                        filtered_bboxes,
                    ) = self.make_std_params()
                    break
                else:
                    # otherwise, we collected enough patches and exit the loop
                    break
                # sometimes seg fails and one crop has more than max patches
            selected_patches, selected_patch_pos = [], []
            for i, (mask_patch, img_patch) in enumerate(zip(mask_patches, img_patches)):
                if torch.any(mask_patch) > 0:
                    if (not do_drop_patches) or (do_drop_patches and torch.var(img_patch) > PATCH_THRESHOLD):
                        # TODO: add randomness to decide to transpose or not
                        selected_patches.append(img_patch.transpose(-1, -2))
                        selected_patch_pos.append(i)

            curr_nb_patches = len(selected_patches)
            if curr_nb_patches == 0:  # if no patches were selectionned
                continue

            tot_patches += curr_nb_patches
            crop_len_list.append(curr_nb_patches)
            selected_patches = torch.cat(selected_patches, dim=1)  # c (n_p p) p
            list_flat_patches.append(selected_patches)
            filtered_patch_pos_list.append(torch.tensor(selected_patch_pos, dtype=torch.long))
            filtered_bboxes.append(bbox)

        # list_flat_patches n_crop (c (n_p p) p)
        if len(list_flat_patches) == 0:
            print(f"tot_patches: {tot_patches}, len: {len(crop_len_list)}, {crop_len_list}")
            flat_patches = std_patching(image)
            (
                crop_len,
                filtered_patch_pos_list,
                filtered_bboxes,
            ) = self.make_std_params()
        else:
            flat_patches = torch.cat(list_flat_patches, dim=1)  # c (n_crop n_p p) p
            filtered_bboxes = torch.stack(filtered_bboxes)
            crop_len = torch.tensor(crop_len_list)

        if tot_patches > MAX_PATCHES:
            print(
                f"WARNING: flat_patches too big dim: ({flat_patches.shape[1]}), {int(flat_patches.shape[1]/self.patch_size)} local crops tokens",
                f"nb_loc_crops: {len(list_flat_patches)}",
            )
        return (
            flat_patches,
            crop_len,
            filtered_patch_pos_list,
            filtered_bboxes,
        )

    def __call__(self, image):
        # image : C H W
        output = {}
        image = self.pad_to_patch_mutiple(image)
        # global crops:
        if self.use_native_res:
            image_global = self.make_rectangle_crop(image)
        else:
            image_global = image

        im1_base = self.geometric_augmentation_global(image_global)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image_global)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        if self.do_seg_crops:
            (
                local_crops,
                masks,
                bboxes,
                local_patch_pos_list,
                resized_masks_int,
            ) = self.make_seg_crops(image, seg_algo=self.do_seg_crops)
            # masks = [mask[None, :, :].repeat((3, 1, 1)) for mask in masks]

            output["pooled_seg"] = resized_masks_int
            # Augment crops and masks
            local_crops, masks = list(
                zip(
                    *[
                        self.geometric_augmentation_local(
                            crop[None, :, :].repeat((3, 1, 1)),
                            mask[None, :, :].repeat((3, 1, 1)).to(torch.float32),
                        )
                        for crop, mask in zip(local_crops, masks)
                    ]
                )
            )
            # masks back to bool
            masks = [mask[0, :, :] > 0 for mask in masks]  # to bool
            # crops to list of patches, masked p are dropped

            output["local_crops_vis"] = local_crops  # before patching, for visualization
            (
                local_crops,
                local_crop_len,
                filtered_patch_pos_list,
                filtered_bboxes,
            ) = self.select_and_concat_nonzero_patches(
                local_crops,
                masks,
                image=image,
                nb_gc_patches=nb_gc_patches,
                bboxes=bboxes,
            )
            output["local_crops"] = local_crops
            output["local_crop_len"] = local_crop_len
            output["local_patch_pos"] = filtered_patch_pos_list
            # (x1, y1, x2, y2)
            crop_dims = torch.cat(
                [
                    filtered_bboxes[:, 2:3] - filtered_bboxes[:, :1] + 1,
                    filtered_bboxes[:, 3:] - filtered_bboxes[:, 1:2] + 1,
                ],
                dim=1,
            )  # n_c 2
            output["local_crop_dims"] = crop_dims
        else:
            local_crops = [
                self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
            ]
            output["local_crops"] = torch.cat(local_crops, dim=0)
        output["offsets"] = ()
        return output
