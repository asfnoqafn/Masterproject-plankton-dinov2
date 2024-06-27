import argparse
import glob
import json
import os
import sys
from enum import Enum
from pathlib import Path

import imageio.v3 as iio
import lmdb
import numpy as np
from tqdm import tqdm

BASE_DIR = "/fast/AG_Kainmueller/data/pan_m"  # max cluster path
MAP_SIZE_IMG = int(1e12)  # 1TB
MAP_SIZE_META = int(1e8)  # 100MB


class _DataType(Enum):
    IMAGES = "images"
    METADATA = "metadata"
    LABELS = "labels"


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    ALL = "all"


def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-5)


def load_img(img_path):
    """
    Load an image from the given image path and preprocess it. All common image file types are supported
    Parameters:
        img_path (str): The path to the image file.
    Returns:
        numpy.ndarray: The preprocessed image as a numpy array of uint8 type.
    """
    img = iio.imread(img_path)  # (H W) or (H W C)
    img = normalize(np.squeeze(img))
    img = (img * 255).astype(np.uint8)
    return img


def create_lmdb_txn(
    dataset_lmdb_dir: Path,
    start_fov_idx,
    end_fov_idx,
    name: _DataType,
    split: _Split = _Split.TRAIN,
):
    lmdb_labels_path = os.path.join(
        dataset_lmdb_dir, f"{start_fov_idx}:{end_fov_idx}-{split.upper()}_{name.value}"
    )
    os.makedirs(lmdb_labels_path, exist_ok=True)
    env = lmdb.open(lmdb_labels_path, map_size=MAP_SIZE_IMG)
    txn = env.begin(write=True)
    return env, txn


def main(args):
    start_fov_idx = args.start_fov_idx
    end_fov_idx = args.end_fov_idx

    selected_dataset_paths = glob.glob(os.path.join(args.dataset_path, "*"))

    base_lmdb_dir = BASE_DIR + args.lmdb_dir_name
    os.makedirs(base_lmdb_dir, exist_ok=True)

    for dataset, path in selected_dataset_paths.items():
        txn_imgs, txn_labels, txn_meta, env_imgs, env_labels, env_metadata = (
            None,
            None,
            None,
            None,
            None,
            None,
        )
        print(f"PROCESSING DATASET {dataset} stored in {path}...")

        dataset_lmdb_dir = os.path.join(base_lmdb_dir, dataset)
        fovs = sorted(glob.glob(path))[start_fov_idx:end_fov_idx]

        print(f"TOTAL #FOVS {len(fovs)} FOR DATASET {dataset}")

        env_imgs, txn_imgs = create_lmdb_txn(
            dataset_lmdb_dir,
            start_fov_idx,
            end_fov_idx,
            name=_DataType.LABELS,
            split=_Split.TRAIN,
        )

        if args.with_labels:
            txn_labels = create_lmdb_txn(
                dataset_lmdb_dir,
                start_fov_idx,
                end_fov_idx,
                name=_DataType.LABELS,
                split=_Split.TRAIN,
            )

        if args.with_metadata:
            txn_meta = create_lmdb_txn(
                dataset_lmdb_dir,
                start_fov_idx,
                end_fov_idx,
                name=_DataType.METADATA,
                split=_Split.TRAIN,
            )

        for img_idx, fov in tqdm(enumerate(sorted(fovs)), total=len(fovs)):
            fov_name_cleaned = "".join(e for e in str(fov) if e.isalnum() or e == "_")
            do_print = img_idx % 50 == 0
            if do_print:
                print(f'idx: {img_idx}/{len(fovs)}, fov: "{fov_name_cleaned}"')

            img_idx_str = f"{dataset}_{img_idx}"
            img_idx_bytes = img_idx_str.encode("utf-8")

            fov_path = os.path.join(path, fov)

            if args.with_metadata:
                # get metadata
                metadata_dict = {}
                metadata_dict["fov"] = fov
                metadata_bytes = json.dumps(metadata_dict).encode("utf-8")
                txn_meta.put(img_idx_bytes, metadata_bytes)

            # if args.with_labels:
            # TODO: get labels_path
            # get segmentation mask
            # segmentation mask has to be uint16 because of values of to ~3000 segments
            # Thus, cannot be jpeg compressed
            # segmentation_mask = (
            #    iio.imread(segmentation_path).squeeze().astype(np.uint16)
            # )
            # txn_labels.put(img_idx_bytes, segmentation_mask.tobytes())

            uint8_img = load_img(fov_path)
            img_jpg_encoded = iio.imwrite("<bytes>", uint8_img, extension=".jpeg")
            txn_imgs.put(img_idx_bytes, img_jpg_encoded)

        env_imgs.close()
        if env_metadata is not None:
            env_metadata.close()
        if env_labels is not None:
            env_labels.close()
        print(f"FINISHED DATASET {dataset}, SAVED AT: {dataset_lmdb_dir}")


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="""Name of dataset to process.""",
    )
    parser.add_argument(
        "--start_fov_idx",
        type=int,
        help="Start index of FOVs to process",
        default=0,
    )
    parser.add_argument(
        "--end_fov_idx",
        type=int,
        help="End index of FOVs to process",
        default=-1,
    )
    parser.add_argument(
        "--lmdb_dir_name", type=str, help="Base lmdb dir name", default="_lmdb"
    )
    parser.add_argument(
        "--with_labels",
        action=argparse.BooleanOptionalAction,
        help="Toggle saving labels",
        default=False,
    )
    parser.add_argument(
        "--with_metadata",
        action=argparse.BooleanOptionalAction,
        help="Toggle saving metadata",
        default=False,
    )

    return parser


if __name__ == "__main__":
    args_parser = get_args_parser()
    args = args_parser.parse_args()
    sys.exit(main(args))
