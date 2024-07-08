import argparse
import os
import sys

sys.path.insert(0, "..")

import json
import os

import h5py
import lmdb


def main(args):
    data_path = args.data_path
    data_files = sorted([el for el in os.listdir(data_path) if el.endswith(".hdf5")])
    data_files = data_files[1:]
    print(data_files)
    print(data_files[0][:-5])

    map_size = int(1e11)
    for data_file in data_files:
        train_data_path = f"/home/jluesch/Documents/data/plankton/{data_file}"

        lmdb_imgs_path = f"/home/jluesch/Documents/data/plankton/lmdb/{data_file[:-5]}_imgs"
        lmdb_labels_path = f"/home/jluesch/Documents/data/plankton/lmdb/{data_file[:-5]}_labels"
        print(lmdb_imgs_path, lmdb_labels_path)

        env_imgs = lmdb.open(lmdb_imgs_path, map_size=map_size)
        env_labels = lmdb.open(lmdb_labels_path, map_size=map_size)

        file = h5py.File(train_data_path, "r")
        # Read the JSON string from the 'file_index' dataset
        file_index_json = file["file_index"][()]
        file_index = json.loads(file_index_json)
        tot_nb_samples = len(file_index["files"])
        print(f"#SAMPLES: {len(file_index['files'])}")

        with env_labels.begin(write=True) as txn_labels:
            with env_imgs.begin(write=True) as txn_imgs:
                for entry in file_index["files"]:
                    if entry["index"] % 50000 == 0:
                        print(
                            entry["index"] / tot_nb_samples,
                            entry["index"],
                            entry["class_id"],
                            entry["path"],
                        )

                    img_bytes = file[entry["path"]][()]
                    txn_imgs.put(
                        str(entry["index"]).encode("utf-8"),
                        img_bytes,
                    )
                    txn_labels.put(
                        str(entry["index"]).encode("utf-8"),
                        str(entry["class_id"]).encode("utf-8"),
                    )

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
