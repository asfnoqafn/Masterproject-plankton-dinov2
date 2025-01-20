import lmdb
import argparse
from pathlib import Path
from tqdm import tqdm
import os

MAP_SIZE_META = int(1e10)  # 10GB

def add_args(parser: argparse.ArgumentParser):
    """
    Add all the additional arguments the script needs.
    """

    parser.add_argument("--lmdb_path", type=str, help="The folder of the lmdb", default="")
    parser.add_argument("--metadata", type=str, help="The metadata as json string", default="")


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


def get_all_keys(env):
    keys = []
    with env.begin(write=False) as txn:
        with txn.cursor() as cursor:
            for key, _ in cursor:
                keys.append(key)
    return keys


def main():
    parser = argparse.ArgumentParser(add_help=True)
    add_args(parser)

    args, _ = parser.parse_known_args()

    env = lmdb.open(str(Path(args.lmdb_path) / 'images'))

    keys = get_all_keys(env)
    print(len(keys), str(Path(args.lmdb_path) / 'images'))

    metadata_lmdb = _create_lmdb(
        Path(args.lmdb_path),
        name=f'metadata',
        map_size=MAP_SIZE_META,
    )

    env_metadata, txn_metadata = metadata_lmdb

    metadata_encoded = args.metadata.encode("utf-8")

    for key in tqdm(keys):
        txn_metadata.put(key, metadata_encoded)
        
    txn_metadata.commit()
    env_metadata.close()




if __name__ == "__main__":
    main()