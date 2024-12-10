import argparse
import json
import os
import sys
from zipfile import BadZipFile, ZipFile

import imageio.v3 as iio
import lmdb
import numpy as np

MAP_SIZE_IMG = int(1e12)  # 1TB
MAP_SIZE_META = int(1e8)  # 100MB


def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-5)


def load_img(img_path):
    img = iio.imread(img_path)  # (N M)

    height, width = img.shape[:2]

    # Check if the image is smaller than the minimum size
    # probably should allow if only one dimension is smaller thin algae
    if height < args.min_size and width < args.min_size:
        raise ValueError(
            f"Image is smaller than {args.min_size}x{args.min_size}"
        )

    # disallow one dimensional images
    if height < 2 or width < 2:
        raise ValueError("Image is one dimensional.")

    img = normalize(np.squeeze(img))
    img = (img * 255).astype(np.uint8)
    return img


def get_all_zips(ifcb_path, excluded_bins):
    zips = []
    for zip_filename in os.listdir(ifcb_path):
        if (
            zip_filename.endswith(".zip")
            and zip_filename not in excluded_bins
        ):
            zips.append(zip_filename)
    return sorted(zips)

def save_zip_to_lmdb(zip_filename, txn, number_of_filtered_images, total_images, total_images_in_chunk):
    zip_filepath = os.path.join(
            args.ifcb_path, zip_filename
        )
    print(f"currently processing: {zip_filepath}")
    with ZipFile(zip_filepath) as zf:
        for image_relpath in zf.namelist():
            if "__MACOSX" in image_relpath:
                continue
            if image_relpath.endswith(".png"):
                image_path = os.path.join(zip_filepath, image_relpath)
                try:
                    image = load_img(
                        zf.read(image_relpath)
                    )
                except Exception as e:
                    print(
                        f"Error loading image {image_path}: {e}. Skipping...",
                        file=sys.stderr,
                    )
                    number_of_filtered_images += 1
                    continue

                img_key = os.path.join(zip_filename, image_relpath).replace("/", "_")  # Replace slashes for safety
                img_key_bytes = img_key.encode("utf-8")

                # Load and encode the image
                img_encoded = iio.imwrite(
                    "<bytes>", image, extension=".png"
                )

                # Save to LMDB
                txn.put(img_key_bytes, img_encoded)
                total_images += 1
                total_images_in_chunk += 1
    return number_of_filtered_images, total_images, total_images_in_chunk

def build_lmdbs(args):
    lmdb_dir = os.path.abspath(args.lmdb_dir_name)
    processed_bins_path = os.path.join(
        lmdb_dir, "processed_bins.json"
    )
    number_of_filtered_images = 0
    total_images_in_chunk = 0
    try:
        with open(processed_bins_path, "r") as f:
            processed_bins_dir = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        processed_bins_dir = {
            "processed_bins": [],
            "bad_bins": [],
            "total_images": 0,
        }
    finally:
        processed_bins = processed_bins_dir[
            "processed_bins"
        ]
        bad_bins = processed_bins_dir["bad_bins"] if "bad_bins" in processed_bins_dir else []
        total_images = processed_bins_dir["total_images"]
        prior_total_images = processed_bins_dir[
            "total_images"
        ]

    zips = get_all_zips(args.ifcb_path, processed_bins + bad_bins)

    lmdb_imgs_path = os.path.join(
        lmdb_dir,
        f"startid_{total_images}_images",
    )
    os.makedirs(lmdb_imgs_path, exist_ok=True)

    env = lmdb.open(lmdb_imgs_path, map_size=MAP_SIZE_IMG)
    txn = env.begin(write=True)
    for zip_filename in zips:
        # open next zip-file and wirte images to lmdb
        try:
            number_of_filtered_images, total_images, total_images_in_chunk = save_zip_to_lmdb(zip_filename, txn, number_of_filtered_images, total_images, total_images_in_chunk)
        except BadZipFile:
            print(
                f"Error loading zip file {zip_filename}: BadZipFile. Skipping...",
                file=sys.stderr,
            )
            bad_bins.append(zip_filename)
            processed_bins_dir["bad_bins"] = bad_bins
            with open(processed_bins_path, "w") as f:
                json.dump(
                    processed_bins_dir,
                    f,
                )
            continue

        processed_bins.append(
            zip_filename
        )  # bin is processed, so add it to processed list
        if (
            total_images_in_chunk >= args.chunk_size
            or zip_filename == zips[-1]
        ):
            # close full lmdb
            txn.commit()
            env.close()
            print(
                f"Finished importing from {args.ifcb_path}, saved at: {lmdb_imgs_path}"
            )
            lmdb_imgs_path = os.path.join(
                lmdb_dir,
                f"startid_{total_images}_images",
            )
            total_images_in_chunk = 0
            # dump json, since all imgs of processed bins are saved to lmdb

            processed_bins_dir["processed_bins"] = (
                processed_bins
            )
            processed_bins_dir["total_images"] = (
                total_images
            )
            with open(processed_bins_path, "w") as f:
                json.dump(
                    processed_bins_dir,
                    f,
                )

            if zip_filename != zips[-1]:
                # open new lmdb
                os.makedirs(lmdb_imgs_path, exist_ok=True)
                env = lmdb.open(
                    lmdb_imgs_path, map_size=MAP_SIZE_IMG
                )
                txn = env.begin(write=True)

    if number_of_filtered_images > 0:
        print(
            f"Filtered {number_of_filtered_images} images"
        )

    print(
        f"TOTAL #images {total_images - prior_total_images} FROM {args.ifcb_path}"
    )


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ifcb_path",
        type=str,
        help="""Path to ifcb folder with bins to process.""",
        default="/home/hk-project-p0021769/hgf_vwg6996/data/ifcb",
    )
    parser.add_argument(
        "--lmdb_dir_name",
        type=str,
        help="Base lmdb dir name",
        default="ifcb",
    )
    parser.add_argument(
        "--min_size",
        type=int,
        help="Minimum image size (width and height)",
        default=0,
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        help="Size to chunk images into different lmdbs",
        default=10_000_000,
    )

    return parser


if __name__ == "__main__":
    args_parser = get_args_parser()
    args = args_parser.parse_args()
    sys.exit(build_lmdbs(args))
