import glob
import os
import pickle
import time
from typing import Optional
from typing_extensions import TypedDict, NotRequired

import lmdb
import numpy as np

from dinov2.data.datasets import ImageNet

_TargetLMDBDataset = int

class Entry(TypedDict):
    index: bytes
    lmdb_imgs_file: str
    class_id: NotRequired[int]

# TODO: Fix inheritance logic
class LMDBDataset(ImageNet):
    Target = _TargetLMDBDataset

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._entries: Optional[list[Entry]] = None
        self._class_ids: Optional[np.ndarray] = None

    def get_image_data(self, index: int) -> bytes:
        entry = self._entries[index]
        lmdb_env = self._lmdb_envs[entry["lmdb_imgs_file"]]
        with lmdb_env.begin() as lmdb_txn:
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
        total_time = time.time()
        _entries: list = []
        extra_full_path = self._get_extra_full_path(extra_path)
        print("extra_full_path", extra_full_path)
        file_list = glob.glob(extra_full_path, recursive=True)

        file_list_labels = sorted([el for el in file_list if el.endswith("labels")])
        print("Datasets labels file list: ", file_list_labels)

        file_list_imgs = sorted([el for el in file_list if el.endswith("imgs") or el.endswith("images")])
        print("Datasets imgs file list: ", file_list_imgs)

        self._lmdb_envs = dict()
        global_idx = 0

        if self.do_short_run:
            file_list_labels = file_list_labels[:1]
            file_list_imgs = file_list_imgs[:1]

        use_labels = len(file_list_labels) > 0 and self.with_targets
        lists_to_iterate = zip(file_list_labels, file_list_imgs) if use_labels else file_list_imgs
        for iter_obj in lists_to_iterate:
            entries: list[Entry] = []
            if use_labels:
                lmdb_path_labels, lmdb_path_imgs = iter_obj
                lmdb_env_labels = lmdb.open(
                    lmdb_path_labels,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                )
            else:
                lmdb_path_imgs: str = iter_obj

            start = time.time()
            lmdb_env_imgs: lmdb.Environment = lmdb.open(
                lmdb_path_imgs,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            self._lmdb_envs[lmdb_path_imgs] = lmdb_env_imgs
            end = time.time() - start
            print("lmdb open time", end)

            
            
            # ex: "/home/jluesch/Documents/data/plankton/lmdb/2007-TRAIN")
            print(
                lmdb_path_imgs,
                "lmdb_env_imgs.stat()",
                lmdb_env_imgs.stat(),
            )
            print(
                lmdb_path_imgs,
                "lmdb_env_imgs.info()",
                lmdb_env_imgs.info(),
            )

            cache_path = f"{lmdb_path_imgs}.cache"
            if self.is_cached and os.path.exists(cache_path):
                    print(f"Loading cache from {cache_path}")
                    start = time.time()
                    with open(cache_path, "rb") as f:
                        keys: list[bytes] = pickle.load(f)
                    entries = [{"index": key, "lmdb_imgs_file": lmdb_path_imgs} for key in keys]
                    _entries.extend(entries)
                    print("time to load cache", time.time() - start)
            else:
                start = time.time()
                # lmdb_txn_imgs = lmdb_env_imgs.begin()
                # save img tcxn from which to get labels later
                with lmdb_env_imgs.begin() as lmdb_txn_imgs:
                    with lmdb_txn_imgs.cursor() as lmdb_cursor:
                        # if use_labels:
                        #     lmdb_cursor = lmdb_txn_labels.cursor()
                        # else:
                        for key in lmdb_cursor.iternext(keys=True, values=False):
                            if use_labels:
                                raise NotImplementedError("Shouldnt be here")
                            entries.append({"index": key, "lmdb_imgs_file": lmdb_path_imgs}) # type: ignore
                            global_idx += 1

                end = time.time() - start
                _entries.extend(entries)
                print("looped over lmdb", end)

                if self.is_cached: # save cache
                    start = time.time()
                    print("Saving cache to", cache_path)
                    keys = [entry["index"] for entry in entries]
                    with open(cache_path, "wb") as f:
                        pickle.dump(keys, f)
                    print("time to save cache", time.time() - start)
                    
        self._entries = _entries
        print("Total time to load all entries", time.time() - total_time)

    def __len__(self) -> int:
        entries = self._get_entries()
        return len(entries)