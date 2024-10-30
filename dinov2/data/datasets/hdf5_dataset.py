import glob
import json
import os
from enum import Enum
from typing import Optional

import h5py
import numpy as np
import pandas as pd

from dinov2.data.datasets import ImageNet

_TargetHDF5Dataset = int


class _SplitHDF5Dataset(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"  # NOTE: torchvision does not support the test split
    ALL = "all"


class HDF5Dataset(ImageNet):
    Target = _TargetHDF5Dataset
    Split = _SplitHDF5Dataset
    hdf5_handles = {}

    def get_image_data(self, index: int) -> bytes:
        entries = self._get_entries()
        entry = entries[index]
        if self.is_cached:
            return entry["img"]
        image_relpath = entry["path"]
        hdf5_path = entry["hdf5_path"]
        hdf5_file = self.hdf5_handles[hdf5_path]
        image_data = hdf5_file[image_relpath][()]
        return image_data

    def get_target(self, index: int) -> Optional[Target]:
        if self.split in [
            _SplitHDF5Dataset.TEST,
            _SplitHDF5Dataset.ALL,
        ]:
            return None

        entries = self._get_entries()
        class_index = entries[index]["class_id"]
        return int(class_index)

    def get_class_names(self) -> np.ndarray:
        self._get_entries()
        return self._class_names

    def get_class_ids(self) -> np.ndarray:
        self._get_entries()
        return self._class_ids

    @property
    def _entries_path(self) -> str:
        if self.root.endswith(".hdf5"):  # if we have a single file
            return self.root
        if self._split.value.upper() == "ALL":
            return "-*.hdf5"
        else:
            return f"-{self._split.value.upper()}.hdf5"

    def _get_extra_full_path(self, extra_path: str) -> str:
        print(f"root: {self.root}, extra_root: {self._extra_root}, extra_path: {extra_path}")
        if extra_path is None:
            extra_path = ""
        if os.path.isfile(self.root):
            return self.root
        else:
            return os.path.join(self.root, self._extra_root + extra_path)

    def _get_entries(self) -> list:
        if self._entries is None:
            self._load_extra(self._entries_path)
        assert self._entries is not None
        return self._entries

    def _load_extra(self, extra_path: str):
        extra_full_path = self._get_extra_full_path(extra_path)
        file_list = glob.glob(extra_full_path)
        print("Datasets file list: ", file_list)
        accumulated = []

        if self.do_short_run:
            file_list = file_list[:1]
        for hdf5_path in file_list:
            file = h5py.File(hdf5_path, "r")
            self.hdf5_handles[hdf5_path] = file
            # Read the JSON string from the 'file_index' dataset
            file_index_json = file["file_index"][()]
            file_index = json.loads(file_index_json)

            df = pd.DataFrame(file_index["files"])
            df["hdf5_path"] = hdf5_path

            if self.is_cached:
                df["img"] = df["path"].apply(lambda val: file[val][()])  # get the images from the paths

            entries = df.to_dict(orient="records")

            # Short run, take only the first 7 classes
            if self.do_short_run:
                entries = [el for el in entries if el["class_id"] < 5]

            accumulated += entries

        unique_class_ids = np.unique([el["class_id"] for el in accumulated])
        unique_class_names = np.unique([el["class_str"] for el in accumulated])
        print(f"#unique_class_ids: {self._split}, {len(unique_class_ids)}")
        print(f"#unique_class_names: {unique_class_names[:8]}, {len(unique_class_names)}")

        self._entries = accumulated
        self._class_ids = df["class_id"].values
        self._class_names = df["class_str"].values

    def __len__(self) -> int:
        entries = self._get_entries()
        return len(entries)

    def close(self):
        for handle in self.hdf5_handles.values():
            handle.close()
