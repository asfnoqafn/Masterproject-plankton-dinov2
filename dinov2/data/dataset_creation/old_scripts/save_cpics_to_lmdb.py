import argparse
import glob
import json
import os
import sys
from pathlib import Path

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


def find_files(folder, image_folder='imgs', extension='.png'):
    if extension[0] != '.':
        extension = '.' + extension
    
    image_folder_path = Path(folder) / image_folder

    abspath = list(image_folder_path.rglob(f'*{extension}'))

    relpath = []
    for path in abspath:
        relpath.append(str(path.relative_to(image_folder_path)))

    print(relpath[0], abspath[0])
    print(relpath[1], abspath[1])
    print(relpath[2], abspath[2])

    return relpath, abspath


def main(args):
    if not args.extension.startswith('.'):
        args.extension = '.' + args.extension

    img_relpath, img_abspath = find_files(args.dataset_path, image_folder=args.image_folder, extension=args.extension)
    
    img_relpath, img_abspath = zip(*sorted(zip(img_relpath, img_abspath)))

    lmdb_dir = os.path.abspath(args.lmdb_dir_name)
    os.makedirs(lmdb_dir, exist_ok=True)
    print(f"TOTAL #images {len(img_abspath)} FROM {args.dataset_path}")

    lmdb_imgs_path = os.path.join(lmdb_dir , 'images')
    lmdb_labels_path = os.path.join(lmdb_dir, 'labels')
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
            # probably should allow if only one dimension is smaller thin algae
            if height < args.min_size and width < args.min_size:
                #print(f"Skipping {abs_path} due to size constraints.")
                continue

            # disallow one dimensional images
            if height < 2 or width < 2:
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
        "--min_size", type=int, help="Minimum image size (width and height)", default=0
    )
    parser.add_argument(
        "--image_folder", type=str, help="Folder in the dataset that contains the label folders with the images", default='imgs'
    )


    return parser


if __name__ == "__main__":
    args_parser = get_args_parser()
    args = args_parser.parse_args()
    sys.exit(main(args))

# example usage: !python save_cpics_pngs_to_lmdb.py --dataset_path="/home/nick/Downloads/113201/FlowCamNet/imgs" --lmdb_dir_name="/home/nick/Documents/ws24/lmdb/bigger_imgs/" --min_size=128 --dataset_name="FlowCamNet"