import dinov2.data.dataset_creation.create_lmdb_dataset as create

import os
import importlib
import argparse
import shutil
from pathlib import Path
import sys

def add_args(parser: argparse.ArgumentParser):
    """
    Add all the additional arguments the script needs.
    """

    parser.add_argument("--collect_script", type=str, help='Script to collect the files, if not given a default script will be created in the repository in ./dinov2/data/dataset_creation/scripts/')
    parser.add_argument("--dataset_path", type=str, help='Name of dataset to process')
    parser.add_argument("--lmdb_path", type=str, help="Output lmdb directory")


def main():
    # collect parser info for help
    parser = argparse.ArgumentParser(add_help=True)
    create.add_args(parser)
    add_args(parser)

    # temporary parser for getting the `--collect_script` argument
    collect_script_parser = argparse.ArgumentParser(add_help=False)
    collect_script_parser.add_argument('--collect_script', type=str)
    collect_script_args, _ = collect_script_parser.parse_known_args()

    # load the collect script or create a new one
    script_path = None
    if collect_script_args.collect_script is None:
        dummy_path = Path(__file__).parent.resolve() / '_convert_script_dummy.py'
        os.makedirs('./dinov2/data/dataset_creation/scripts/', exist_ok=True)
        script_path = Path('./dinov2/data/dataset_creation/scripts/rename_me.py')
        shutil.copyfile(dummy_path, script_path)
    else:
        script_path = Path(collect_script_args.collect_script)

    spec = importlib.util.spec_from_file_location("module.name", str(script_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = mod
    spec.loader.exec_module(mod)
    
    # finalize the parser
    mod.add_args(parser)
    args, _ = parser.parse_known_args()

    print(f"PROCESSING DATASET stored in {args.dataset_path}...")

    os.makedirs(args.lmdb_path, exist_ok=True)
    data = mod.collect_files(args.dataset_path, parser)
    create.build_databases(data, parser)


if __name__ == "__main__":
    main()