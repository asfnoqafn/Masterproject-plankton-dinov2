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


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    ALL = "all"


IMAGE_SUFFIXES = (
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
)


def normalize(x: np.ndarray):
    """
    Normalizes an numpy.ndarray without dividing by zero.
    """
    return (x - x.min()) / (x.max() - x.min() + 1e-5)


def load_image(image_path):
    """
    Load an image from the given image path and preprocess it. All common image file types are supported
    Parameters:
        image_path (str): The path to the image file.
    Returns:
        numpy.ndarray: The preprocessed image as a numpy array of uint8 type.
    """

    image = iio.imread(image_path)  # (H W) or (H W C)
    image = normalize(np.squeeze(image))
    image = (image * 255).astype(np.uint8)
    return image


def create_lmdb(
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


def _find_files(dataset_path):
    """
    Finds all images and labels in the dataset
    Parameters:
        dataset_path (str): The folder of the dataset
    Returns:
        list: A list of tuples of image path strings and label strings (or None) and metadata dicts (or None)
    """

    result = []

    for path in Path(dataset_path).rglob('*.jpg'):
        image_folder_name = path.parents[0].name
        label = None
        if image_folder_name != 'test':
            label = image_folder_name

        result.append((path.as_posix(), label, None))

    return result


def write_databases(images_lmdb, labels_lmdb, metadata_lmdb, data):
    """
    Writes all three databases (images, labels, metadata) simulataniously
    Parameters:
        images_lmdb (tuple): Enironment and transaction of the image database
        labels_lmdb (tuple): Enironment and transaction of the label database
        metadata_lmdb (tuple): Enironment and transaction of the metadata database
        data (list): List of tuples of image path strings and label strings (or None) and metadata dicts (or None)
    """

    env_images, txn_images = images_lmdb
    env_labels, txn_labels = labels_lmdb
    env_metadata, txn_metadata = metadata_lmdb

    for image_path, label, metadata in tqdm(data):
        index = str(uuid.uuid4()).encode("utf-8")

        uint8_image = load_image(image_path)
        image_jpg_encoded = iio.imwrite("<bytes>", uint8_image, extension=".jpeg")
        txn_images.put(index, image_jpg_encoded)
        
        if label != None:
            txn_labels.put(index, label.encode("utf-8"))

        if metadata != None:
            metadata_encoded = json.dumps(metadata).encode("utf-8")
            txn_metadata.put(index, metadata_encoded)
        
    txn_images.commit()
    txn_labels.commit()
    txn_metadata.commit()

    env_images.close()
    env_labels.close() 
    env_metadata.close()


def main():
    MAP_SIZE_IMAGE = int(1e10)
    MAP_SIZE_LABEL = int(1e10)
    MAP_SIZE_METADATA = int(1e10)
    dataset_path = "/home/hk-project-p0021769/hgf_auh3910/data/datasciencebowl"
    base_lmdb_dir = "/home/hk-project-p0021769/hgf_auh3910/data/lmdb_datasciencebowl"

    print(f"PROCESSING DATASET stored in {dataset_path}...")

    os.makedirs(base_lmdb_dir, exist_ok=True)

    data = _find_files(dataset_path)

    print(f"TOTAL #images {len(data)}")

    images_lmdb = create_lmdb(
        base_lmdb_dir,
        name=_DataType.IMAGES,
        map_size=MAP_SIZE_IMAGE,
    )

    labels_lmdb = create_lmdb(
        base_lmdb_dir,
        name=_DataType.LABELS,
        map_size=MAP_SIZE_LABEL,
    )

    metadata_lmdb = create_lmdb(
        base_lmdb_dir,
        name=_DataType.METADATA,
        map_size=MAP_SIZE_METADATA,
    )

    write_databases(images_lmdb, labels_lmdb, metadata_lmdb, data)
    
    print(f"FINISHED DATASET SAVED AT: {base_lmdb_dir}")


if __name__ == "__main__":
    main()