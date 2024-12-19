import os
import lmdb
import json
import sys
import imageio.v3 as iio
import numpy as np
from tqdm import tqdm
import psutil
from sklearn.model_selection import train_test_split
import argparse
import difflib

def get_available_memory():
    """Get available memory and return a safe allocation size for LMDB."""
    available_memory = psutil.virtual_memory().available
    print(f"Available memory: {available_memory / 1024 / 1024} MB")
    return int(available_memory * 0.75) 

MAP_SIZE_IMG = get_available_memory()
MAP_SIZE_META = int(get_available_memory() * 0.1)

def load_lmdb_data(lmdb_path):
    """
    Loads data from an LMDB file and returns it as a list of (key, value) pairs.
    """
    env = lmdb.open(lmdb_path, readonly=True)
    data = []
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            data.append((key, value))
    print(f"Loaded {len(data)} items from {lmdb_path}")
    return data

def save_lmdb_data(lmdb_path_img, lmdb_path_label, img_data, label_data, label_map, label_map_path=None):
    """
    Saves images and labels to LMDB and saves the label mapping as a JSON file.
    """
    env_imgs = lmdb.open(lmdb_path_img, map_size=MAP_SIZE_IMG)
    env_labels = lmdb.open(lmdb_path_label, map_size=MAP_SIZE_META)

    with (
        env_imgs.begin(write=True) as txn_imgs,
        env_labels.begin(write=True) as txn_labels,
    ):
        for (img_key, img_encoded), (label_key, label) in tqdm(zip(img_data, label_data), total=len(img_data)):
            if img_key != label_key:
                print(f"Warning: Mismatched keys! img_key: {img_key}, label_key: {label_key}")
                continue  # Skip if keys don't match

            # Convert label to integer ID using the provided label map
            label_str: str = label.decode("utf-8").lower()  # Assuming label is bytes
            if label_str not in label_map:
                print(f"Warning: Label {label_str} not found in label_map.")
                continue
            if label_map[label_str] == -1:
                continue

            label_id = label_map[label_str]

            # Store the image and label
            txn_imgs.put(img_key, img_encoded)
            label_entry = label_id.to_bytes(4, byteorder="little")
            txn_labels.put(label_key, label_entry)

    # Save label map to a JSON file, if required
    if label_map_path:
        with open(label_map_path, "w") as f:
            json.dump(label_map, f)

    env_imgs.close()
    env_labels.close()

def load_all_datasets(main_folder):
    """
    Load all datasets in the given folder and return combined image and label data.
    """
    img_data = []
    label_data = []
    print(f"Loading datasets from {main_folder}")
    for dataset in sorted(os.listdir(main_folder)):
        print(f"Loading dataset: {dataset}")
        dataset_path = os.path.join(main_folder, dataset)
        for subdir in sorted(os.listdir(dataset_path)):
            print(f"Loading dataset: {subdir}")
            datasetpath_lmdb = os.path.join(dataset_path, subdir)
            print("dataset path: ", datasetpath_lmdb) 
            
            if datasetpath_lmdb.endswith("_imgs") or datasetpath_lmdb.endswith('images'):
                img_data.extend(load_lmdb_data(datasetpath_lmdb))
            elif datasetpath_lmdb.endswith("_labels") or datasetpath_lmdb.endswith('labels'):
                label_data.extend(load_lmdb_data(datasetpath_lmdb))
            else:
                print(f"Skipping {datasetpath_lmdb}")
            print(f"Loaded dataset: {dataset}")
                    
    return img_data, label_data

def is_in_blacklist(label, blacklist):
    # return True if label is in blacklist
    for s in blacklist:
        if s in label:
            return True
    return False


def split_and_save_data(main_folder, output_folder, test_size=0.2):
    """
    Loads all datasets, splits the data into train and test, and saves them in the output folder.
    """
    os.makedirs(output_folder, exist_ok=True)

    img_data, label_data = load_all_datasets(main_folder)

    print(f"Total data loaded: {len(img_data)} images and {len(label_data)} labels.")

    blacklist = ['artifacts', 'artifacts_edge', 'unknown_blobs_and_smudges', 'unknown_sticks', 'unknown_unclassified', 
                 'artefact', 'badfocus', 'dark', 'light', 'streak', 'vertical line', 'artefact']
    # consistent label mapping TODO add similar string machting and blacklist classes such as blurry
    label_map = {}
    next_id = 0
    for _, label in label_data:
        label_str: str = label.decode("utf-8") # Assumes labels were stored as bytes
        label_str = label_str.lower()
        label_str = label_str.replace(" ", "")
        label_str = label_str.replace("_", "")
        label_str = label_str.replace("-", "")
        label_str = label_str.replace("'", "") # TODO ugly
        # matches = difflib.get_close_matches(label_str, label_map.keys()) 
        # Adds similarity matcher, UNTESTED and probably brkeaks classes like 'cyano a' and 'cyano b'
        if label_str not in label_map:

            if is_in_blacklist(label_str, blacklist):
                label_map[label_str] = -1
                print("blacklist label: ", label_str)
            else:
                label_map[label_str] = next_id
                next_id += 1

    print(f"Generated label map with {len(label_map)} classes.")

    train_imgs, test_imgs = train_test_split(img_data, test_size=test_size, shuffle=True, random_state=43)
    train_labels, test_labels = train_test_split(label_data, test_size=test_size, shuffle=True, random_state=43)

    # Save the split data to LMDB
    save_lmdb_data(
        os.path.join(output_folder, "-TRAIN_imgs"),
        os.path.join(output_folder, "-TRAIN_labels"),
        train_imgs,
        train_labels,
        label_map,
        os.path.join(output_folder, "TRAIN_label_map.json")
    )
    save_lmdb_data(
        os.path.join(output_folder, "-VAL_imgs"),
        os.path.join(output_folder, "-VAL_labels"),
        test_imgs,
        test_labels,
        label_map,
        os.path.join(output_folder, "VAL_label_map.json")
    )

    print(f"Finished processing and saving datasets to {output_folder}")


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
        "--min_size", type=int, help="Minimum image size (width and height)", default=0.0
    )

    return parser

def main(args):
    split_and_save_data(main_folder=args.dataset_path , output_folder=args.lmdb_dir_name, test_size=0.2)

if __name__ == "__main__":
    #split_and_save_data(main_folder="data/seaone_raw", output_folder="data/seaone_raw_lmdb", test_size=0.00001)
    args_parser = get_args_parser()
    args = args_parser.parse_args()
    sys.exit(main(args))