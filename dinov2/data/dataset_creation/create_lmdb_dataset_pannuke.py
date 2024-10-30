import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "..")

import os

import lmdb

BASE_DATA_PATH = "/home/jluesch/Documents/data/cell_imaging_data/"


def main(args):
    fold_names = ["fold1", "fold2", "fold3"]

    map_size = int(1e11)
    for fold_nb in fold_names:
        train_data_path = os.path.join(BASE_DATA_PATH, fold_nb)

        lmdb_imgs_path = os.path.join(train_data_path, f"{fold_nb}_imgs")
        lmdb_labels_path = os.path.join(train_data_path, f"{fold_nb}_labels")
        print(lmdb_imgs_path, lmdb_labels_path)

        env_imgs = lmdb.open(lmdb_imgs_path, map_size=map_size)
        env_labels = lmdb.open(lmdb_labels_path, map_size=map_size)

        base_image_prefix = f"images/{fold_nb}/"
        base_mask_prefix = f"masks/{fold_nb}/"

        img_path = os.path.join(train_data_path, base_image_prefix, "images.npy")
        # type_path = os.path.join(BASE_DATA_PATH, base_image_prefix, "types.npy")

        mask_path = os.path.join(train_data_path, base_mask_prefix, "masks.npy")

        images = np.load(img_path).astype(np.uint8)  # (2656, 256, 256, 3)
        masks = np.load(mask_path).astype(np.uint8)  # (2656, 256, 256, 3)

        tot_nb_samples = images.shape[0]

        # Read the JSON string from the 'file_index' dataset
        print(f"#SAMPLES: {tot_nb_samples}")

        with env_labels.begin(write=True) as txn_labels:
            with env_imgs.begin(write=True) as txn_imgs:
                for i in range(tot_nb_samples):
                    if i % 1000 == 0:
                        print(
                            f"""progress {i}/{tot_nb_samples}
                              masks shape: {masks[i].shape}
                              images shape: {images[i].shape}"""
                        )

                    img_bytes = images[i].tobytes()
                    mask_bytes = masks[i].tobytes()
                    txn_imgs.put(str(i).encode("utf-8"), img_bytes)
                    txn_labels.put(str(i).encode("utf-8"), mask_bytes)

        env_imgs.close()
        env_labels.close()
        print(f"FINISHED {lmdb_imgs_path}")

    return 0


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        dest="data_path",
        type=str,
        help="Training dataset",
        default="/home/jluesch/Documents/data/plankton",
    )
    return parser


if __name__ == "__main__":
    args_parser = get_args_parser()
    args = args_parser.parse_args()
    sys.exit(main(args))
