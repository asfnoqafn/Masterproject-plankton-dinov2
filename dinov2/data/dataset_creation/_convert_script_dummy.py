from pathlib import Path
import argparse

IMAGE_SUFFIXES = (
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
)


def add_args(parser: argparse.ArgumentParser):
    """
    Add all the additional arguments the script needs.
    """

    parser.add_argument("--unlabeled_folders", type=str, help='Comma separated list of folders that contain unlabeled data', default='')
    parser.add_argument("--image_folder", type=str, help="Folder in the dataset that contains the label folders with the images", default='imgs')


def collect_files(dataset_path: Path, parser: argparse.ArgumentParser):
    """
    Collect all files in the dataset.
    Parameters:
        dataset_path (Path): The path to the dataset.
        parser(ArgumentParser): A parser with all arguments in it
    Returns:
        list: A list of tuples (image_path: str, label: str, metadata_json: str).
    """
    
    args, _ = parser.parse_known_args()

    result = []

    paths = []
    for suffix in IMAGE_SUFFIXES:
        paths += (Path(dataset_path) / args.image_folder).rglob(f'*{suffix}')

    unlabeled_folders = args.unlabeled_folders.split(',')
    
    for path in paths:
        image_folder_name = path.parents[0].name
        label = image_folder_name

        if label in unlabeled_folders:
            result.append((path.as_posix(), None, None))
        else:
            result.append((path.as_posix(), label, None))

    return result