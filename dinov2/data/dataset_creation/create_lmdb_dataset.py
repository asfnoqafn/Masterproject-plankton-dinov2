import json
import os
import sys
import argparse
import shutil
import lmdb
import uuid
import imageio.v3 as iio
import numpy as np
from enum import Enum
from pathlib import Path
from typing import Callable
from tqdm import tqdm

MAP_SIZE_IMG = int(1e12)  # 1TB
MAP_SIZE_META = int(1e8)  # 100MB

IMAGE_SUFFIXES = (
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
)

Image_Transformation = Callable[[np.ndarray], np.ndarray]

class _DataType(Enum):
    IMAGES = "images"
    METADATA = "metadata"
    LABELS = "labels"


def _normalize(x: np.ndarray):
    """
    Normalizes an numpy.ndarray without dividing by zero.
    """
    return (x - x.min()) / (x.max() - x.min() + 1e-5)


def _load_image(image_path, extra_transformations: list[Image_Transformation] = []):
    """
    Load an image from the given image path and preprocess it. All common image file types are supported
    Parameters:
        image_path (str): The path to the image file.
    Returns:
        numpy.ndarray: The preprocessed image as a numpy array of uint8 type.
    """
    image = iio.imread(image_path)  # (H W) or (H W C)
    for transformation in extra_transformations:
        image = transformation(image)
    image = _normalize(np.squeeze(image))
    image = (image * 255).astype(np.uint8)
    return image


def _create_lmdb(
    dataset_lmdb_dir: Path,
    name: str,
    map_size: int
):
    """
    Creates an LMDB database file and opens an environment and transaction
    Parameters:
        dataset_lmdb_dir (Path): Location of the lmdb file to be created.
        name (str): Name of the database.
        map_size (int): Memory used for the database creation.
    Returns:
        Environment: The environment for the transaction. (needs to be closed by the using function)
        Transaction: The transaction for interacting with the lmdb. (changes need to be committed)
    """

    lmdb_path = str(dataset_lmdb_dir / f"{name}")
    os.makedirs(lmdb_path, exist_ok=True)
    env = lmdb.open(lmdb_path, map_size=map_size)
    txn = env.begin(write=True)
    return env, txn


def _delete_lmdb(dataset_lmdb_dir: Path):
    lmdb_path = str(dataset_lmdb_dir)
    shutil.rmtree(lmdb_path)


def get_image_dimensions(img_path):
    img = iio.imread(img_path, pilmode='RGB')
    return img.shape[:2] # Returns (height, width)


def _write_databases(
    unlabeled_images_lmdb,
    labeled_images_lmdb,
    labels_lmdb,
    metadata_lmdb,
    data,
    min_size: int = 0,
    extra_transformations: list[Image_Transformation] = [],
    extension: str = '.png'
):
    """
    Writes all three databases (images, labels, metadata) simulataniously
    Parameters:
        images_lmdb (tuple): Enironment and transaction of the image database
        labels_lmdb (tuple): Enironment and transaction of the label database
        metadata_lmdb (tuple): Enironment and transaction of the metadata database
        data (list): List of tuples of image path strings and label strings (or None) and metadata dicts (or None)
    """

    env_unlabeled_images, txn_unlabeled_images = unlabeled_images_lmdb
    env_labeled_images, txn_labeled_images = labeled_images_lmdb
    env_labels, txn_labels = labels_lmdb
    env_metadata, txn_metadata = metadata_lmdb

    if not extension.startswith('.'):
        extension = '.' + extension

    number_of_filtered_images = 0
    for image_path, label, metadata in tqdm(data):
        img_key_bytes = uuid.uuid4().bytes

        height, width = get_image_dimensions(image_path)

        if height < min_size and width < min_size:
            continue

        # disallow one dimensional images
        if height < 2 or width < 2:
            continue

        try:
            uint8_image = _load_image(image_path, extra_transformations=extra_transformations)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}. Skipping...", file=sys.stderr)
            number_of_filtered_images += 1
            continue
        image_jpg_encoded = iio.imwrite("<bytes>", uint8_image, extension=extension)

        if label is not None:
            txn_labeled_images.put(img_key_bytes, image_jpg_encoded)
            txn_labels.put(img_key_bytes, label.encode("utf-8"))
        else:
            txn_unlabeled_images.put(img_key_bytes, image_jpg_encoded)

        if metadata is not None:
            metadata_encoded = json.dumps(metadata).encode("utf-8")
            txn_metadata.put(img_key_bytes, metadata_encoded)

    if number_of_filtered_images > 0:
        print(f"Filtered {number_of_filtered_images} images")
        
    txn_unlabeled_images.commit()
    txn_labeled_images.commit()
    txn_labels.commit()
    txn_metadata.commit()


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


def build_databases(
    data: list,
    map_size_img: int = MAP_SIZE_IMG,
    map_size_meta: int = MAP_SIZE_META,
    extra_transformations: list[Image_Transformation] = [],
):
    """
    Converts the data colleted via the collect script into lmdb databases
    Args:
        data (list): List of imagefiles with their label (or None) and their metadata (or None)
        base_lmdb_directory (Path): The directory to store the lmdb files in
        map_size (int): The memory allocated for lmdb creation
    """

    args = get_args()

    number_of_images = len(data)
    number_of_images_with_labels = sum(1 for item in data if item[1] != None)
    number_of_images_with_metadata = sum(1 for item in data if item[2] != None)
    number_of_images_without_labels = number_of_images - number_of_images_with_labels

    print('DATASET STATISTICS:')
    print(f'{number_of_images} <- number of images')
    print(f'{number_of_images_with_labels} <- number of images with labels')
    print(f'{number_of_images_with_metadata} <- number of images with metadata')
    print(f'{number_of_images_without_labels} <- number of images without labels')

    unlabeled_images_lmdb = _create_lmdb(
        Path(args.lmdb_path),
        name=f'unlabled_{_DataType.IMAGES.value}_{number_of_images_without_labels}imgs',
        map_size=map_size_img,
    )
    labeled_images_lmdb = _create_lmdb(
        Path(args.lmdb_path),
        name=f'labled_{_DataType.IMAGES.value}_{number_of_images_with_labels}imgs',
        map_size=map_size_img,
    )
    labels_lmdb = _create_lmdb(
        Path(args.lmdb_path),
        name=f'{_DataType.LABELS.value}_{number_of_images_with_labels}imgs',
        map_size=map_size_meta,
    )
    metadata_lmdb = _create_lmdb(
        Path(args.lmdb_path),
        name=f'{_DataType.METADATA.value}_{number_of_images_with_metadata}imgs',
        map_size=map_size_meta,
    )

    _write_databases(
        unlabeled_images_lmdb,
        labeled_images_lmdb,
        labels_lmdb,
        metadata_lmdb,
        data,
        min_size=args.min_size,
        extra_transformations=extra_transformations,
        extension=args.extension
    )
    
    print('removing empty datasets')

    if number_of_images_without_labels == 0:
        _delete_lmdb(Path(unlabeled_images_lmdb[0].path()))
    
    if number_of_images_with_labels == 0:
        _delete_lmdb(Path(labeled_images_lmdb[0].path()))
        _delete_lmdb(Path(labels_lmdb[0].path()))
    
    if number_of_images_with_metadata == 0:
        _delete_lmdb(Path(metadata_lmdb[0].path()))

    unlabeled_images_lmdb[0].close()
    labeled_images_lmdb[0].close()
    labels_lmdb[0].close() 
    metadata_lmdb[0].close()
    
    print(f"FINISHED DATASET SAVED AT: {args.lmdb_path}")
