import json
import os
import uuid
import lmdb
import imageio.v3 as iio
import numpy as np
from enum import Enum
from pathlib import Path
from tqdm import tqdm


class _DataType(Enum):
    IMAGES = "images"
    METADATA = "metadata"
    LABELS = "labels"


def _normalize(x: np.ndarray):
    """
    Normalizes an numpy.ndarray without dividing by zero.
    """
    return (x - x.min()) / (x.max() - x.min() + 1e-5)


def _load_image(image_path):
    """
    Load an image from the given image path and preprocess it. All common image file types are supported
    Parameters:
        image_path (str): The path to the image file.
    Returns:
        numpy.ndarray: The preprocessed image as a numpy array of uint8 type.
    """

    image = iio.imread(image_path)  # (H W) or (H W C)
    image = _normalize(np.squeeze(image))
    image = (image * 255).astype(np.uint8)
    return image


def _create_lmdb(
    dataset_lmdb_dir: Path,
    name: _DataType,
    map_size: int = 1e10
):
    """
    Creates an LMDB database file and opens an environment and transaction
    Parameters:
        dataset_lmdb_dir (Path): Location of the lmdb file to be created.
        name (_DataType): Enum to indicate the database content.
        map_size (int): Memory used for the database creation.
    Returns:
        Environment: The environment for the transaction. (needs to be closed by the using function)
        Transaction: The transaction for interacting with the lmdb. (changes need to be committed)
    """

    lmdb_path = os.path.join(
        dataset_lmdb_dir,
        f"{name.value}",
    )
    os.makedirs(lmdb_path, exist_ok=True)
    env = lmdb.open(lmdb_path, map_size=map_size)
    txn = env.begin(write=True)
    return env, txn


def _write_databases(unlabeled_images_lmdb, labeled_images_lmdb, labels_lmdb, metadata_lmdb, data):
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

    for image_path, label, metadata in tqdm(data):
        index = str(uuid.uuid4()).encode("utf-8")

        uint8_image = _load_image(image_path)
        image_jpg_encoded = iio.imwrite("<bytes>", uint8_image, extension=".jpeg")

        if label != None:
            txn_labeled_images.put(index, image_jpg_encoded)
            txn_labels.put(index, label.encode("utf-8"))
        else:
            txn_unlabeled_images.put(index, image_jpg_encoded)

        if metadata != None:
            metadata_encoded = json.dumps(metadata).encode("utf-8")
            txn_metadata.put(index, metadata_encoded)
        
    txn_unlabeled_images.commit()
    txn_labeled_images.commit()
    txn_labels.commit()
    txn_metadata.commit()

    env_unlabeled_images.close()
    env_labeled_images.close()
    env_labels.close() 
    env_metadata.close()


def build_databases(
    data: list,
    base_lmdb_directory: Path,
    map_size: int,
):
    """
    Converts the data colleted via the collect script into lmdb databases
    Args:
        data (list): List of imagefiles with their label (or None) and their metadata (or None)
        base_lmdb_directory (Path): The directory to store the lmdb files in
        map_size (int): The memory allocated for lmdb creation
    """

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
        base_lmdb_directory,
        name=f'unlabled_{_DataType.IMAGES}_{number_of_images_without_labels}',
        map_size=map_size,
    )

    labeled_images_lmdb = _create_lmdb(
        base_lmdb_directory,
        name=f'labled_{_DataType.IMAGES}_{number_of_images_with_labels}',
        map_size=map_size,
    )

    labels_lmdb = _create_lmdb(
        base_lmdb_directory,
        name=f'{_DataType.LABELS}_{number_of_images_with_labels}',
        map_size=map_size,
    )

    metadata_lmdb = _create_lmdb(
        base_lmdb_directory,
        name=f'{_DataType.METADATA}_{number_of_images_with_metadata}',
        map_size=map_size,
    )

    _write_databases(unlabeled_images_lmdb, labeled_images_lmdb, labels_lmdb, metadata_lmdb, data)
    
    print(f"FINISHED DATASET SAVED AT: {base_lmdb_directory}")
