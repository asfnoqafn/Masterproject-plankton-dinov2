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
MAP_SIZE_META = int(7e8) # 700MB

def get_image_dimensions(img_path): 
    try:
        img = iio.imread(img_path, pilmode='RGB')  # This reads only metadata
        return img.shape[:2]  # Returns (height, width)
    except Exception as e:
        print(f"Error reading image {img_path}: {e}")
        return None  # Return None if there is an error
    

def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-5)

def load_img(img_path):
    try:
        img = iio.imread(img_path)  # (N M)
        img = normalize(np.squeeze(img))
        img = (img * 255).astype(np.uint8)
        return img
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None  # Return None if there is an error


def find_files(folder, ext=None):
    ext = '.png' if ext is None else ext
    if ext[0] != '.':
        ext = '.' + ext
    relpath, abspath, subdirs = [], [], []
    s = os.scandir(folder)
    for entry in s:
        if entry.is_dir():
            # Skip the 'LRaw' directory
            if entry.name == 'LRaw':
              continue
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

    lmdb_imgs_path = os.path.join(lmdb_dir , f"images")
    lmdb_labels_path = os.path.join(lmdb_dir, f"labels")
    os.makedirs(lmdb_imgs_path, exist_ok=True)
    os.makedirs(lmdb_labels_path, exist_ok=True)

    env_imgs = lmdb.open(lmdb_imgs_path, map_size=MAP_SIZE_IMG)
    env_labels = lmdb.open(lmdb_labels_path, map_size=MAP_SIZE_META)

    with (
        env_imgs.begin(write=True) as txn_imgs,
        env_labels.begin(write=True) as txn_labels,
    ):  
        corrupted_images_count = 0
        for img_idx, (rel_path, abs_path) in tqdm(enumerate(zip(img_relpath, img_abspath)), total=len(img_abspath)):
            #print("rel_path", rel_path)
            dims = get_image_dimensions(abs_path)
            if dims is None: 
                corrupted_images_count += 1
                print(f"Skipping corrupted image: {abs_path}")
                continue
            height, width = dims

            # Check if the image is smaller than the minimum size 
            if height < args.min_size or width < args.min_size:     # for now used min_size = 0
                continue

            class_name = os.path.basename(os.path.dirname(rel_path))
            class_name_cleaned = class_name

            class_name_cleaned = class_name.replace("LClass_", "")  # Remove the 'LClass_' prefix
            print(f"Class: {class_name_cleaned}")

            img_key = rel_path.replace("/", "_")  # Replace slashes for safety
            img_key_bytes = img_key.encode("utf-8")

            # Load and encode the image
            uint8_img = load_img(abs_path)
            
            img_encoded = iio.imwrite("<bytes>", uint8_img, extension=args.extension)
            # Save to LMDB
            txn_imgs.put(img_key_bytes, img_encoded)
            txn_labels.put(img_key_bytes, class_name_cleaned.encode("utf-8"))
    print(f"Total corrupted images skipped: {corrupted_images_count}")
    env_imgs.close()
    env_labels.close()
    print(f"Finished importing from {args.dataset_path} and subdirectories, saved at: {lmdb_imgs_path}")

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "--lmdb_dir_name", type=str, help="Base directory for LMDB storage.", default="_lmdb"
    )
    parser.add_argument(
        "--extension", type=str, help="Image file extension (default: 'png').", default="png"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the output dataset (default: 'dataset').",
        default="dataset"
    )
    parser.add_argument(
        "--min_size", type=int, help="Minimum image size (width and height)", default=32
    )

    return parser

if __name__ == "__main__":
    args_parser = get_args_parser()
    args = args_parser.parse_args()
    sys.exit(main(args))

