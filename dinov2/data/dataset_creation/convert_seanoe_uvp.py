from dinov2.data.dataset_creation.create_lmdb_dataset import build_databases

import os
from pathlib import Path
from tqdm import tqdm

IMAGE_SUFFIXES = (
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
)

BASE_SEANOE_PATH = Path("/home/hk-project-p0021769/hgf_vwg6996/data/seanoe_uvp")

# zoo_scan_net
def collect_files(training_csv_path, test_csv_path, validation_csv_path, unlabeled_csv_path):
    result = []
    csv_paths = [training_csv_path, test_csv_path, validation_csv_path, unlabeled_csv_path]

    for csv_path in csv_paths:
        with open(BASE_SEANOE_PATH.joinpath(csv_path).as_posix(), 'r') as f:
             lines = f.readlines()[1:]
             for i, line in tqdm(enumerate(lines), total=len(lines)):
                label, img_path = line.strip().split(',')
                if (label == ''):
                    label = None
                img = BASE_SEANOE_PATH.joinpath(img_path)
                result_entry = (img.as_posix(), label, None)
                result.append(result_entry)
                # if Path.exists(img):
                #     result.append(result_entry)
                # else:
                #     print(f"Path doesn't exist: {img.as_posix()} (in csv {csv_path} line {i})")
    return result

def main():
    base_lmdb_directory = Path("/home/hk-project-p0021769/hgf_twg7490/data/seanoe_uvp_lmdb_mixed")
    os.makedirs(base_lmdb_directory, exist_ok=True)
    
    print(f"PROCESSING DATASET stored in {BASE_SEANOE_PATH.as_posix()}...")

    data = collect_files("training.csv", "test.csv", "validation.csv", "unlabeled.csv")
    build_databases(data, base_lmdb_directory, int(1e12))

if __name__ == "__main__":
    main()