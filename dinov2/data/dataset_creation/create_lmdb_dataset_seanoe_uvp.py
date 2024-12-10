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

# python create_lmdb_dataset_seanoe_uvp.py --base_dir "/home/hk-project-p0021769/hgf_vwg6996/data/seanoe_uvp" --csv_file "unlabeled.csv" --start_img_idx 0 --end_img_idx 500000 --split "train" --with_metadata
# python create_lmdb_dataset_seanoe_uvp.py --base_dir "/home/hk-project-p0021769/hgf_vwg6996/data/seanoe_uvp" --csv_file "unlabeled.csv" --split "train" --with_metadata
# python create_lmdb_dataset_seanoe_uvp.py --base_dir "/home/hk-project-p0021769/hgf_vwg6996/data/seanoe_uvp" --csv_file "test.csv" --start_img_idx 0 --end_img_idx 10 --split "test" --with_metadata --with_labels
# python create_lmdb_dataset_seanoe_uvp.py --base_dir "/home/hk-project-p0021769/hgf_vwg6996/data/seanoe_uvp" --csv_file "test.csv" --split "test" --with_metadata --with_labels


class _DataType(Enum):
    IMAGES = "images"
    METADATA = "metadata"
    LABELS = "labels"


class _Split(Enum):
    TRAIN = "train"
    TEST = "test"


IMG_SUFFIXES = (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG")


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
    start_img_idx,
    end_img_idx,
    name: _DataType,
    split: _Split,
    map_size=1e10,
):
    lmdb_labels_path = os.path.join(
        dataset_lmdb_dir, f"{start_img_idx}:{end_img_idx}-{split}_{name.value}"
    )
    os.makedirs(lmdb_labels_path, exist_ok=True)
    env = lmdb.open(lmdb_labels_path, map_size=map_size)
    txn = env.begin(write=True)
    return env, txn


def main(args):
    BASE_DIR = args.base_dir
    MAP_SIZE_IMG = int(args.map_size_img)
    MAP_SIZE_META = int(args.map_size_meta)

    start_img_idx = args.start_img_idx
    end_img_idx = args.end_img_idx

    print(f"PROCESSING DATASET stored in {BASE_DIR}...")
    print(f"With labels: {args.with_labels}, with metadata: {args.with_metadata}")

    txn_imgs, txn_labels, txn_meta, env_imgs, env_labels, env_metadata = (
            None,
            None,
            None,
            None,
            None,
            None,
    )

    dataset_name = BASE_DIR.split('/')[-1]
    dataset_lmdb_dir = os.path.join(BASE_DIR, dataset_name + args.lmdb_dir_name)
    os.makedirs(dataset_lmdb_dir, exist_ok=True)

    env_imgs, txn_imgs = create_lmdb_txn(
        dataset_lmdb_dir,
        start_img_idx,
        end_img_idx,
        name=_DataType.IMAGES,
        split=args.split,
        map_size=MAP_SIZE_IMG,
    )

    if args.with_labels:
        env_labels, txn_labels = create_lmdb_txn(
            dataset_lmdb_dir,
            start_img_idx,
            end_img_idx,
            name=_DataType.LABELS,
            split=args.split,
            map_size=MAP_SIZE_IMG,
        )

    if args.with_metadata:
        env_metadata, txn_meta = create_lmdb_txn(
            dataset_lmdb_dir,
            start_img_idx,
            end_img_idx,
            name=_DataType.METADATA,
            split=args.split,
            map_size=MAP_SIZE_META,
        )


    csv_path = os.path.join(BASE_DIR, args.csv_file)

    if args.with_labels:
        test_classes = []
        with open(csv_path, 'r') as f:
            if end_img_idx != -1:
                end_img_idx += 1
            lines = f.readlines()[start_img_idx+1:end_img_idx]
            print("LINES", csv_path)
            for img_idx, line in tqdm(enumerate(lines), total=len(lines)):
                label, img_path = line.strip().split(',')
                test_classes.append(label)

        print(f"Total classes: {len(np.unique(test_classes))}")
        print(f"Classes: {np.unique(test_classes)}")
        classes_to_labels = {k:v for v,k in enumerate(np.unique(test_classes))}
        with open(os.path.join(dataset_lmdb_dir, "classes_to_labels.json"), "w") as f:
            json.dump(classes_to_labels, f)

    with open(csv_path, 'r') as f:
        if end_img_idx != -1:
            end_img_idx += 1
        lines = f.readlines()[start_img_idx+1:end_img_idx]
        print(f"TOTAL #imgs {len(lines)}")

        for img_idx, line in tqdm(enumerate(lines), total=len(lines)):
            label_name, img_path = line.strip().split(',')
            img_idx_str = f"{img_path}_{img_idx}"
            img_idx_bytes = img_idx_str.encode("utf-8")

            if args.with_metadata:
                # get metadata
                metadata_dict = {}
                metadata_dict["img"] = img_path
                metadata_bytes = json.dumps(metadata_dict).encode("utf-8")
                txn_meta.put(img_idx_bytes, metadata_bytes)

            if args.with_labels:
                label = classes_to_labels[label_name]
                txn_labels.put(img_idx_bytes, label.to_bytes(1, sys.byteorder))
                    
            uint8_img = load_img(os.path.join(BASE_DIR, img_path))
            img_jpg_encoded = iio.imwrite("<bytes>", uint8_img, extension=".jpeg")
            txn_imgs.put(img_idx_bytes, img_jpg_encoded)

    env_imgs.close()
    if env_metadata is not None:
        env_metadata.close()
    if env_labels is not None:
        env_labels.close()
    print(f"FINISHED DATASET SAVED AT: {dataset_lmdb_dir}")


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        help="""Name of dataset to process.""",
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        help="""Name of csv file to process.""",
    )
    parser.add_argument(
        "--start_img_idx",
        type=int,
        help="Start index of imgs to process",
        default=0,
    )
    parser.add_argument(
        "--end_img_idx",
        type=int,
        help="End index of imgs to process",
        default=-1,
    )
    parser.add_argument(
        "--lmdb_dir_name", type=str, help="Base lmdb dir name", default="_lmdb"
    )
    parser.add_argument(
        "--split", type=str, help="Dataset split", default="train"
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
    parser.add_argument(
        "--map_size_img",
        type=int,
        help="Space to allocate for lmdb file for images",
        default=1e10,
    )
    parser.add_argument(
        "--map_size_meta",
        type=int,
        help="Space to allocate for lmdb file for images",
        default=1e8,
    )

    return parser


if __name__ == "__main__":
    args_parser = get_args_parser()
    args = args_parser.parse_args()
    sys.exit(main(args))