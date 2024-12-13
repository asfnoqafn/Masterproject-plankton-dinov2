from dinov2.data.dataset_creation.create_lmdb_dataset import *

import os
from pathlib import Path
import importlib

def collect_files(dataset_path, image_folder='imgs'):
    result = []

    paths = []
    for suffix in IMAGE_SUFFIXES:
        paths += (Path(dataset_path) / image_folder).rglob(f'*{suffix}')
    
    for path in paths:
        image_folder_name = path.parents[0].name
        label = image_folder_name

        result.append((path.as_posix(), label, None))

    return result


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path", type=str, help="""Name of dataset to process.""",
    )
    parser.add_argument(
        "--lmdb_path", type=str, help="Output lmdb directory"
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

    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.lmdb_path, exist_ok=True)
    
    print(f"PROCESSING DATASET stored in {args.dataset_path}...")

    data = collect_files(args.dataset_path, args.image_folder)
    build_databases(data)


if __name__ == "__main__":
    main()