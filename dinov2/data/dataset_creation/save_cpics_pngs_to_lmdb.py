import argparse
import glob
import json
import os
import sys

import imageio.v3 as iio
import lmdb
import numpy as np
from tqdm import tqdm

BASE_DIR = " "  # max cluster path
MAP_SIZE_IMG = int(1e12)  # 1TB
MAP_SIZE_META = int(1e8)  # 100MB

def get_image_dimensions(img_path):
    img = iio.imread(img_path, pilmode='RGB')  # This reads only metadata
    return img.shape[:2]  # Returns (height, width)

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
    if not args.extension.startswith('.'):
        args.extension = '.' + args.extension

    img_relpath, img_abspath = find_files(args.dataset_path, ext=args.extension)
    
    img_relpath, img_abspath = zip(*sorted(zip(img_relpath, img_abspath)))

    lmdb_dir = os.path.abspath(args.lmdb_dir_name)
    os.makedirs(lmdb_dir, exist_ok=True)
    print(f"TOTAL #images {len(img_abspath)} FROM {args.dataset_path}")

    lmdb_imgs_path = os.path.join(lmdb_dir , f"{args.dataset_name}_imgs")
    lmdb_labels_path = os.path.join(lmdb_dir, f"{args.dataset_name}_labels")
    os.makedirs(lmdb_imgs_path, exist_ok=True)
    os.makedirs(lmdb_labels_path, exist_ok=True)

    env_imgs = lmdb.open(lmdb_imgs_path, map_size=MAP_SIZE_IMG)
    env_labels = lmdb.open(lmdb_labels_path, map_size=MAP_SIZE_META)

    with (
        env_imgs.begin(write=True) as txn_imgs,
        env_labels.begin(write=True) as txn_labels,
    ):
        for img_idx, (rel_path, abs_path) in tqdm(enumerate(zip(img_relpath, img_abspath)), total=len(img_abspath)):
            #if img_idx >= 10:  # testing
            #    break

            height, width = get_image_dimensions(abs_path)

            # Check if the image is smaller than the minimum size
            if height < args.min_size or width < args.min_size:
                #print(f"Skipping {abs_path} due to size constraints.")
                continue

            class_name = os.path.basename(os.path.dirname(rel_path))

            img_key = rel_path.replace("/", "_")  # Replace slashes for safety
            img_key_bytes = img_key.encode("utf-8")

            # Load and encode the image
            uint8_img = load_img(abs_path)
            img_encoded = iio.imwrite("<bytes>", uint8_img, extension=args.extension)

            # Save to LMDB
            txn_imgs.put(img_key_bytes, img_encoded)
            txn_labels.put(img_key_bytes, class_name.encode("utf-8"))

    env_imgs.close()
    env_labels.close()
    print(f"Finished importing from {args.dataset_path} and subdirectories, saved at: {lmdb_imgs_path}")



    env_imgs.close()
    env_labels.close()
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
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="""Name of outputfile.""",
        default="dataset"
    )
    parser.add_argument(
    "--min_size", type=int, help="Minimum image size (width and height)", default=32
    )

    return parser


if __name__ == "__main__":
    print("wtf")
    args_parser = get_args_parser()
    args = args_parser.parse_args()
    sys.exit(main(args))
