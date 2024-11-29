from pathlib import Path

IMAGE_SUFFIXES = (
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
)

"""
Each function collects all image-paths with their labels and metadata and returns 
a list of tuples of image-path strings and label strings (or None) and metadata dicts (or None)
"""

# datasiencebowl
def collect_files_dsb(dataset_path):
    result = []

    paths = []
    for suffix in IMAGE_SUFFIXES:
        paths += Path(dataset_path).rglob(f'*{suffix}')

    for path in paths:
        image_folder_name = path.parents[0].name
        label = None
        if image_folder_name != 'test':
            label = image_folder_name

        result.append((path.as_posix(), label, None))

    return result


# zoo_scan_net
def collect_files_zoo_scan(dataset_path):
    result = []

    paths = []
    for suffix in IMAGE_SUFFIXES:
        paths += Path(dataset_path + '/imgs/').rglob(f'*{suffix}')

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
