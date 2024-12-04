import glob
import os
from typing import Optional

import lmdb
import numpy as np

from dinov2.data.datasets import ImageNet

_TargetLMDBDataset = int

# TODO: Fix inheritance logic
class LMDBDataset(ImageNet):
    Target = _TargetLMDBDataset
    lmdb_handles = {}

    def get_image_data(self, index: int) -> bytes:
        entry = self._entries[index]
        lmdb_txn = self._lmdb_txns[entry["lmdb_imgs_file"]]
        image_data = lmdb_txn.get(entry["index"]) # we dont need to encode since new script already saves encoded img
        return image_data

    def get_target(self, index: int) -> Optional[Target]:
        if not self.with_targets:
            return None
        entries = self._get_entries()
        class_index = entries[index].get("class_id")
        return int(class_index) if class_index is not None else None

    @property
    def _entries_path(self) -> str:
        if self.root.endswith("TRAIN") or self.root.endswith("VAL"):  # if we have a single file
            return self.root + "_*"
        elif self._split.value.upper() == "ALL":
            return os.path.join(self.root, "**")
        else:
            return os.path.join(
                self.root,
                f"*-{self._split.value.upper()}_*",
            )

    def _get_extra_full_path(self, extra_path: str) -> str:
        if not os.path.isdir(extra_path):
            return os.path.join(self.root, extra_path)
        else:
            return os.path.join(self.root, "*")

    def _get_entries(self) -> list:
        if self._entries is None:
            self._load_extra(self._entries_path)
        assert self._entries is not None
        return self._entries
    
    def get_class_ids(self) -> np.ndarray:
        self._get_entries()
        return self._class_ids

    def _load_extra(self, extra_path: str):
        extra_full_path = self._get_extra_full_path(extra_path)
        print("extra_full_path", extra_full_path)
        file_list = glob.glob(extra_full_path, recursive=True)

        file_list_labels = sorted([el for el in file_list if el.endswith("labels")])
        print("Datasets labels file list: ", file_list_labels)

        file_list_imgs = sorted([el for el in file_list if el.endswith("imgs") or el.endswith("images")])
        print("Datasets imgs file list: ", file_list_imgs)

        accumulated = []
        self._lmdb_txns = dict()
        global_idx = 0

        if self.do_short_run:
            file_list_labels = file_list_labels[:1]
            file_list_imgs = file_list_imgs[:1]

        use_labels = len(file_list_labels) > 0 and self.with_targets
        lists_to_iterate = zip(file_list_labels, file_list_imgs) if use_labels else file_list_imgs
        for iter_obj in lists_to_iterate:
            if use_labels:
                lmdb_path_labels, lmdb_path_imgs = iter_obj
                lmdb_env_labels = lmdb.open(
                    lmdb_path_labels,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                )
                lmdb_txn_labels = lmdb_env_labels.begin()

            else:
                lmdb_path_imgs = iter_obj

            lmdb_env_imgs = lmdb.open(
                lmdb_path_imgs,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            # ex: "/home/jluesch/Documents/data/plankton/lmdb/2007-TRAIN")
            print(
                lmdb_path_imgs,
                "lmdb_env_imgs.stat()",
                lmdb_env_imgs.stat(),
            )

            lmdb_txn_imgs = lmdb_env_imgs.begin()
            # save img tcxn from which to get labels later
            self._lmdb_txns[lmdb_path_imgs] = lmdb_txn_imgs

            if use_labels:
                lmdb_cursor = lmdb_txn_labels.cursor()
            else:
                lmdb_cursor = lmdb_txn_imgs.cursor()
            for key, value in lmdb_cursor:
                entry = dict()
                if use_labels:
                    entry["class_id"] = int.from_bytes(value, byteorder="little")
                entry["index"] = key
                entry["lmdb_imgs_file"] = lmdb_path_imgs

                accumulated.append(entry)
                global_idx += 1
            lmdb_cursor.close()

        self._entries = accumulated

    def __len__(self) -> int:
        entries = self._get_entries()
        return len(entries)

    def close(self):
        for handle in self.lmdb_handles.values():
            handle.close()
