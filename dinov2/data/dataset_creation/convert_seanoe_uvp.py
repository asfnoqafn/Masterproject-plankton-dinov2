import argparse
import os
import sys
from pathlib import Path

from tqdm import tqdm

from dinov2.data.dataset_creation.create_lmdb_dataset import build_databases


def remove_bottom_px(px=32):
    def remove_bottom_px(image):
        if len(image) <= px:
            raise ValueError(f"Image is too small to remove {px}px from the bottom")
        return image[:-px]
    return remove_bottom_px

# collect (image, label, metadata) tuples from seanoe_uvp csv files
def collect_files(dataset_path: Path, include_morphocluster=False):
    result = []
    csv_paths = ["training.csv", "test.csv", "validation.csv", "unlabeled.csv"]
    if include_morphocluster:
        csv_paths += "morphocluster.csv"
    csv_paths = [dataset_path / csv_path for csv_path in csv_paths]

    for csv_path in csv_paths:
        if not csv_path.exists():
            print(f"File {csv_path} does not exist. Skipping...")
            continue
        with csv_path.open() as f:
             lines = f.readlines()[1:] # skip header
             for line in tqdm(lines, total=len(lines)):
                label, img_path = line.strip().split(',')
                if (label == ''):
                    label = None
                img = dataset_path / img_path
                result.append((img.as_posix(), label, None))
    return result

def main(args):
    os.makedirs(args.lmdb_dir_name, exist_ok=True)
    
    print(f"PROCESSING DATASET stored in {args.dataset_path}...")
    data = collect_files(args.dataset_path)
    build_databases(data, base_lmdb_directory=args.lmdb_dir_name, dataset_path=args.dataset_path, extra_transformations=[remove_bottom_px(args.scalebar_pixels)])

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=Path,
        help="""Name of dataset to process.""",
        default=Path("/home/hk-project-p0021769/hgf_vwg6996/data/seanoe_uvp")
    )
    parser.add_argument(
        "--lmdb_dir_name", type=Path, help="Base lmdb dir name", default=Path("/home/hk-project-p0021769/hgf_grc7525/data/seanoe_uvp_lmdb_mixed")
    )
    parser.add_argument(
        "--scalebar_pixels", type=int, help="Number of bottom pixels with scalebar that should be removed", default=32
    )
    return parser


if __name__ == "__main__":
    args_parser = get_args_parser()
    args = args_parser.parse_args()
    sys.exit(main(args))