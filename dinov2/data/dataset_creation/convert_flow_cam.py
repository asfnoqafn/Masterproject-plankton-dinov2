from dinov2.data.dataset_creation.create_lmdb_dataset import build_databases

import os
from pathlib import Path

IMAGE_SUFFIXES = (
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
)

# zoo_scan_net
def collect_files(dataset_path):
    result = []

    paths = []
    for suffix in IMAGE_SUFFIXES:
        paths += (Path(dataset_path) / 'imgs').rglob(f'*{suffix}')

    labels = set()
    for path in paths:
        image_folder_name = path.parents[0].name
        label = image_folder_name
        labels.add(label)

        result.append((path.as_posix(), label, None))

    consistent_labels = set()
    for label in labels:
        consistent_label = label.lower()
        consistent_label = consistent_label.replace(' ', '_')
        consistent_labels.add(consistent_label)

    return result

def main():
    dataset_path = '/home/hk-project-p0021769/hgf_grc7525/data/with_labels/FlowCamNet'
    base_lmdb_directory = '~/own_data/lmdb_flowcamnet'
    os.makedirs(base_lmdb_directory, exist_ok=True)
    
    print(f"PROCESSING DATASET stored in {dataset_path}...")

    data = collect_files(dataset_path)
    build_databases(data, base_lmdb_directory, int(1e10))



if __name__ == "__main__":
    main()