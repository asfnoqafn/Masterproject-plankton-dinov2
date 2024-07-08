import glob
import os
import sys
from enum import Enum
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch

from dinov2.data.datasets import ImageNet

_TargetLMDBDataset = int


class _SplitLMDBDataset(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"  # NOTE: torchvision does not support the test split
    ALL = "all"


class NPYDataset(ImageNet):
    Target = _TargetLMDBDataset
    Split = _SplitLMDBDataset
    lmdb_handles = {}

    def get_image_data(self, index: int) -> bytes:
        return self._entries[index]["image"]

    def get_target(self, index: int) -> Optional[Target]:
        if self.split in [
            _SplitLMDBDataset.TEST,
            _SplitLMDBDataset.ALL,
        ]:
            return None
        else:
            if self.with_targets:
                return self._entries[index]["mask"]
            else:
                return None

    @property
    def _entries_path(self) -> str:
        if self.root.endswith("TRAIN") or self.root.endswith("VAL"):  # if we have a single file
            return self.root
        elif self._split.value.upper() == "ALL":
            return self.root
        else:
            return os.path.join(
                self.root,
                f"*-{self._split.value.upper()}_*",
            )

    def _get_extra_full_path(self, extra_path: str) -> str:
        if not os.path.isdir(self.root):
            return extra_path
        else:
            return self.root

    def _get_entries(self) -> list:
        if self._entries is None:
            self._load_extra(self._entries_path)
        assert self._entries is not None
        return self._entries

    def _load_extra(self, extra_path: str):
        # extra_full_path = self._get_extra_full_path(extra_path)
        print(f"extra_path {extra_path}")
        # fold1/masks/fold1/masks.npy
        # fold1/images/fold1/images.npy

        mask_path = os.path.join(
            extra_path,
            "fold*",
            "masks",
            "fold*",
            "masks.npy",
        )
        file_list_labels = sorted(glob.glob(mask_path))

        image_path = os.path.join(
            extra_path,
            "fold*",
            "images",
            "fold*",
            "images.npy",
        )
        file_list_imgs = sorted(glob.glob(image_path))

        print(f"Datasets labels file list: {file_list_labels}")
        print(f"Datasets imgs file list: {file_list_imgs}")

        accumulated = []
        if self.do_short_run:
            file_list_labels = file_list_labels[:1]
            file_list_imgs = file_list_imgs[:1]

        for image_file, mask_file in zip(file_list_imgs, file_list_labels):
            images_array = np.load(image_file)
            masks_array = np.load(mask_file)
            print(f"imgs shape: {images_array.shape}, masks shape: {masks_array.shape}")
            for image, mask in zip(images_array, masks_array):
                accumulated.append({"mask": mask, "image": image})

        self._entries = accumulated

    def __len__(self) -> int:
        entries = self._get_entries()
        return len(entries)

    def close(self):
        for handle in self.lmdb_handles.values():
            handle.close()

    def __getitem__(self, index: int) -> Union[Tuple[Any, Any], torch.Tensor]:
        img = self.get_image_data(index).transpose((2, 0, 1))

        image = torch.from_numpy(img / 255.0).to(torch.float32)

        mask = self.get_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, mask)

        return image, target
