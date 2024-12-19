import dinov2.data.dataset_creation.create_lmdb_dataset as lmdb_creator

import os
from pathlib import Path


# zoo_scan_net
def collect_files(dataset_path):
    result = []

    paths = []
    for suffix in lmdb_creator.IMAGE_SUFFIXES:
        paths += Path(dataset_path + '/imgs/').rglob(f'*{suffix}')

    for path in paths:
        image_folder_name = path.parents[0].name
        label = image_folder_name
        result.append((path.as_posix(), label, None))

    return result

def main():
    dataset_path = r"C:\Users\rk81o\Desktop\MP\data\ZooScanNet"
    base_lmdb_directory = r"C:\Users\rk81o\Desktop\data\zoo_scan_net_lmdb"
    os.makedirs(base_lmdb_directory, exist_ok=True)
    
    print(f"PROCESSING DATASET stored in {dataset_path}...")

    data = collect_files(dataset_path)
    lmdb_creator.build_databases(data, base_lmdb_directory, int(1e10))



if __name__ == "__main__":
    main()