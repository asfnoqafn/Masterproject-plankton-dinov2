import argparse
import glob
import json
import os
import sys

import imageio.v3 as iio
import lmdb
import numpy as np
from tqdm import tqdm

BASE_DIR = "/gpfs/work/vaswani/LPCruises/rois"  # max cluster path
MAP_SIZE_IMG = int(1e12)  # 1TB
MAP_SIZE_META = int(1e8)  # 100MB


def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-5)


def load_img(img_path):
    img = iio.imread(img_path)  # (N M)
    img = normalize(np.squeeze(img))
    img = (img * 255).astype(np.uint8)
    return img


def find_files(folder,  ext=None):
    ext = '.png' if ext is None else ext
    if ext[0] != '.':
        ext = '.' + ext
    relpath, abspath, subdirs = [], [], []
    s = os.scandir(folder)
    for entry in s:
        if entry.is_dir():
            subdirs.append(entry.name)
        elif entry.name.endswith(ext):
            relpath.append(entry.name)
            abspath.append(os.path.abspath(entry.path))

    # add images from subfolders
    for subdir in subdirs:
        subdir_relpath, subdir_abspath = find_files(os.path.join(folder, subdir), ext=ext)
        relpath += [os.path.join(subdir, f) for f in subdir_relpath]
        abspath += subdir_abspath

    return relpath, abspath


def main(args):
    if not args.extension.startswith('.'): args.extension = '.' + args.extension
    img_relpath, img_abspath = find_files(args.dataset_path, ext=args.extension)

    lmdb_dir = os.path.abspath(args.lmdb_dir_name)
    os.makedirs(lmdb_dir, exist_ok=True)
    print(f"TOTAL #images {len(img_abspath)} FROM {args.dataset_path}")
    lmdb_imgs_path = lmdb_dir + "-TRAIN_imgs"
    lmdb_labels_path = lmdb_dir + "-TRAIN_labels"
    os.makedirs(lmdb_imgs_path, exist_ok=True)
    # os.makedirs(lmdb_labels_path, exist_ok=True)

    env_imgs = lmdb.open(lmdb_imgs_path, map_size=MAP_SIZE_IMG)
    # env_labels = lmdb.open(lmdb_labels_path, map_size=MAP_SIZE_IMG)

    with (
        env_imgs.begin(write=True) as txn_imgs,
        # env_labels.begin(write=True) as txn_labels,
    ):
        for img_idx, img in tqdm(enumerate(sorted(img_abspath)), total=len(img_abspath)):
            imgname = img_relpath[img_idx]
            img_name_cleaned = "".join(e for e in imgname if e.isalnum() or e == '_')
            do_print = img_idx % 50 == 0
            if do_print:
                print(f'idx: {img_idx}/{len(img_abspath)}, fov: "{img_name_cleaned}"')

            img_idx_str = f"{args.dataset_path}_{img_idx}"
            img_idx_bytes = img_idx_str.encode("utf-8")

            uint8_img = load_img(img)
            img_encoded = iio.imwrite("<bytes>", uint8_img, extension=args.extension)
            txn_imgs.put(img_idx_bytes, img_encoded)

    env_imgs.close()
    #env_labels.close()
    print(f"Finished importing from {args.dataset_path} and subdirectories, saved at: {lmdb_imgs_path}")


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="""Name of dataset to process.""",
    )
    parser.add_argument(
        "--lmdb_dir_name", type=str, help="Base lmdb dir name", default="_lmdb"
    )
    parser.add_argument(
        "--extension", type=str, help="Image extension for saving inside lmdb", default="png"
    )

    return parser


if __name__ == "__main__":
    args_parser = get_args_parser()
    args = args_parser.parse_args()
    sys.exit(main(args))
