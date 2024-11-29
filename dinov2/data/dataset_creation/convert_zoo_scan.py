from dinov2.data.dataset_creation.collect_files import collect_files_zoo_scan
from dinov2.data.dataset_creation.create_lmdb_dataset import build_databases

import os

def main():
    dataset_path = r"C:\Users\rk81o\Desktop\MP\data\ZooScanNet"
    base_lmdb_directory = r"C:\Users\rk81o\Desktop\data\zoo_scan_net_lmdb"
    os.makedirs(base_lmdb_directory, exist_ok=True)
    
    print(f"PROCESSING DATASET stored in {dataset_path}...")

    data = collect_files_zoo_scan(dataset_path)
    build_databases(data, base_lmdb_directory, int(1e10))



if __name__ == "__main__":
    main()