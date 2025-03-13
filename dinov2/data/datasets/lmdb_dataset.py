import glob
import os
import pickle
import time
from typing import Optional

import lmdb
import numpy as np
from itertools import zip_longest

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
    
    def get_metadata(self, index: int) -> dict:
        if not self.with_metadata:
            return None
        entry = self._entries[index]
        lmdb_txn = self._lmdb_txns[entry["lmdb_meta_file"]]
        metadata = lmdb_txn.get(entry["index"]).decode("utf-8")
        return metadata

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
        total_time = time.time()
        extra_full_path = self._get_extra_full_path(extra_path)
        print("extra_full_path", extra_full_path)
        file_list = glob.glob(extra_full_path, recursive=True)

        file_list_labels = sorted([el for el in file_list if el.endswith("labels")])
        print("Datasets labels file list: ", file_list_labels)

        file_list_imgs = sorted([el for el in file_list if el.endswith("imgs") or el.endswith("images")])
        print("Datasets imgs file list: ", file_list_imgs)

        file_list_meta = sorted([el for el in file_list if el.endswith("meta") or el.endswith("metadata")])
        print("Datasets metadata file list: ", file_list_meta)

        accumulated = []
        self._lmdb_txns = dict()
        global_idx = 0

        if self.do_short_run:
            file_list_labels = file_list_labels[:1]
            file_list_imgs = file_list_imgs[:1]

        for iter_obj in zip_longest(file_list_imgs, file_list_labels, file_list_meta):
            entries = []
            start = time.time()
                lmdb_env_labels = lmdb.open(
                    lmdb_path_labels,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                )
                lmdb_txn_labels = lmdb_env_labels.begin()

            if use_metadata:
                lmdb_env_meta = lmdb.open(
                    lmdb_path_meta,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                )
                lmdb_txn_meta = lmdb_env_meta.begin()

            end = time.time() - start
            print("lmdb open time", end)

            # ex: "/home/jluesch/Documents/data/plankton/lmdb/2007-TRAIN")
            print(
                lmdb_path_imgs,
                "lmdb_env_imgs.stat()",
                lmdb_env_imgs.stat(),
            )
            start = time.time()
            lmdb_txn_imgs = lmdb_env_imgs.begin()
            # save img tcxn from which to get labels later
            self._lmdb_txns[lmdb_path_imgs] = lmdb_txn_imgs

            if self.is_cached and os.path.exists(cache_path):
                print(f"Loading cache from {cache_path}")
                start = time.time()
                with open(cache_path, "rb") as f:
                    keys: list[bytes] = pickle.load(f)
                entries = [{"index": key, "lmdb_imgs_file": lmdb_path_imgs} for key in keys]
                accumulated.extend(entries)
                print("time to load cache", time.time() - start)
                if use_metadata or use_labels:
                    raise NotImplementedError("ERROR: Caching not implemented for metadata or labels")
            else:

                if use_labels:
                    lmdb_cursor: lmdb.Cursor = lmdb_txn_labels.cursor()
                else:
                    lmdb_cursor: lmdb.Cursor = lmdb_txn_imgs.cursor()
                    
                for key, value in lmdb_cursor:
                    entry = dict()
                    if use_labels:
                        entry["class_id"] = int.from_bytes(value, byteorder="little")
                
                    entry["index"] = key
                    entry["lmdb_imgs_file"] = lmdb_path_imgs

                    if use_metadata:
                        entry["lmdb_meta_file"] = lmdb_path_meta

                    entries.append(entry)
                    global_idx += 1
                lmdb_cursor.close()

                accumulated.extend(entries)
                end = time.time() - start
                print("looped over lmdb", end)

                if self.is_cached: # save cache
                    start = time.time()
                    print("Saving cache to", cache_path)
                    keys = [entry["index"] for entry in entries]
                    with open(cache_path, "wb") as f:
                        pickle.dump(keys, f)
                    print("time to save cache", time.time() - start)

        self._entries = accumulated
        print("Total time to load all entries", time.time() - total_time)

    def __len__(self) -> int:
        entries = self._get_entries()
        return len(entries)

    def close(self):
        for handle in self.lmdb_handles.values():
            handle.close()