from dinov2.data.dataset_creation.create_lmdb_dataset import *

import os
import importlib
import argparse
import shutil
from pathlib import Path
import sys

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collect_script", type=str, help='Script to collect the files, if not given a default script will be created in the repository in /scripts/',
    )
    parser.add_argument(
        "--dataset_path", type=str, help='Name of dataset to process',
    )
    parser.add_argument(
        "--lmdb_path", type=str, help="Output lmdb directory"
    )
    parser.add_argument(
        "--image_folder", type=str, help="Folder in the dataset that contains the label folders with the images", default='imgs'
    )

    args, _ = parser.parse_known_args()
    return args


def main():
    args = get_args()

    script_path = None
    if args.collect_script is None:
        dummy_path = Path(__file__).parent.resolve() / '_convert_script_dummy.py'
        os.makedirs('./scripts/', exist_ok=True)
        script_path = Path('./scripts/rename_me.py')
        shutil.copyfile(dummy_path, script_path)
    else:
        script_path = Path(args.collect_script)


    spec = importlib.util.spec_from_file_location("module.name", str(script_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = mod
    spec.loader.exec_module(mod)
    
    os.makedirs(args.lmdb_path, exist_ok=True)
    
    print(f"PROCESSING DATASET stored in {args.dataset_path}...")


    data = mod.collect_files(args.dataset_path)
    build_databases(data)


if __name__ == "__main__":
    main()