import argparse
import concurrent.futures
import contextlib
import json
import logging
import logging.handlers
import multiprocessing
import os
import shutil
import sys
import threading
import uuid
from multiprocessing.managers import ValueProxy
from pathlib import Path
from queue import Empty, Queue
from typing import Optional
from zipfile import ZipFile

import imageio.v3 as iio
import lmdb
import numpy as np
from imageio.typing import ImageResource

from csv_parser import Bin, download_metadata_csv, get_downloaded_bins, get_args_parser as get_csv_args_parser

# setup logging for each lmdb worker
def setup_logging(queue: multiprocessing.Queue):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a QueueHandler to send log messages to the queue
    queue_handler = logging.handlers.QueueHandler(queue)
    logger.addHandler(queue_handler)

    return logger


def listener(queue: Queue):
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s -  %(processName)-10s %(name)s -  %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Continuously listen for log messages from the queue
    while True:
        try:
            record = queue.get()
            if record is None:  # We send this as a sentinel to tell the listener to quit.
                break
            logger.handle(record)  # No level or filter logic applied - just do it!
        except Exception:
            continue

def init_log_queue(queue: multiprocessing.Queue):
    global log_queue
    log_queue = queue

# later just import both of these fronm create lmdb script
def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-5)


def load_img(img_path: ImageResource) -> np.ndarray:
    img = iio.imread(img_path)  # (N M)

    height, width = img.shape[:2]

    # disallow one dimensional images
    if height < 2 or width < 2:
        raise ValueError("Image is one dimensional.")

    img = normalize(np.squeeze(img))
    img = (img * 255).astype(np.uint8)
    return img

def save_zip_to_lmdb(bin: Bin, txn, bin_output_dir, txn_meta: Optional[lmdb.Transaction] = None) -> int:
    zip_filename: str = bin.id + ".zip"
    zip_filepath = os.path.join(bin_output_dir, zip_filename)
    total_images = 0

    with ZipFile(zip_filepath) as zf:
        for image_relpath in zf.namelist():
            if "__MACOSX" in image_relpath:
                continue
            if image_relpath.endswith(".png"):
                image_path = os.path.join(zip_filepath, image_relpath)
                try:
                    image = load_img(
                        zf.read(image_relpath)
                    )
                except Exception as e:
                    print(
                        f"Error loading image {image_path}: {e}. Skipping...",
                        file=sys.stderr,
                    )
                    continue

                img_key_bytes = uuid.uuid4().bytes

                # Load and encode the image
                img_encoded = iio.imwrite(
                    "<bytes>", image, extension=".png"
                )

                # Save to LMDB
                txn.put(img_key_bytes, img_encoded)

                if txn_meta is not None:
                    metadata = json.dumps({
                        "bin": bin.id,
                        "filename": image_relpath,
                        "dataset": bin.dataset,
                        "utc": bin.sample_time,
                        "lat": bin.latitude,
                        "long": bin.longitude,
                        "ml_analyzed": bin.ml_analyzed,
                        "instrument": bin.instrument,
                    })
                    metadata_encoded = metadata.encode("utf-8")
                    txn_meta.put(img_key_bytes, metadata_encoded)
                total_images += 1
    return total_images


def build_lmdb(bin_queue: Queue, lock: threading.Lock, lmdb_count: int, lmdb_dir: str, processed_bins_path: str, chunk_size: int, bin_output_dir: str, logger: logging.Logger, tmp_dir: Optional[str] = None, timeout=1200, save_metadata: bool = False) -> None:
    lmdb_name = f"{lmdb_count:03}_images"
    meta_lmdb_name = f"{lmdb_count:03}_metadata"
    lmdb_path = os.path.join(lmdb_dir, lmdb_name)
    meta_lmdb_path = os.path.join(lmdb_dir, meta_lmdb_name)
    tmp_lmdb_path = os.path.join(tmp_dir, lmdb_name) if tmp_dir is not None else None
    tmp_meta_lmdb_path = os.path.join(tmp_dir, meta_lmdb_name) if tmp_dir is not None else None
    total_images = 0
    bad_bins = []
    processed_bins: dict[str, int] = {}
    # open lmdb
    env: lmdb.Environment = lmdb.open(tmp_lmdb_path if tmp_lmdb_path is not None else lmdb_path, map_size=int(1e12))
    meta_env: Optional[lmdb.Environment] = None

    if save_metadata:
        meta_env = lmdb.open(tmp_meta_lmdb_path if tmp_dir is not None else meta_lmdb_path, map_size=int(1e10))

    with (env.begin(write=True) as txn, (meta_env.begin(write=True) if meta_env else contextlib.nullcontext()) as txn_meta):
        while total_images < chunk_size:
            bin: Bin = bin_queue.get(block=True, timeout=timeout)
            zip_filename: str = bin.id + ".zip"
            # open next zip-file and write images to lmdb
            try:
                logger.info(f"Reading zip file {os.path.join(bin_output_dir, zip_filename)} from queue...")
                num_images = save_zip_to_lmdb(bin=bin, txn=txn, bin_output_dir=bin_output_dir, txn_meta=txn_meta)
                total_images += num_images
                logger.info(f"Added {num_images} images from {zip_filename} to {lmdb_name}")
            except Exception as e:
                logger.error(
                    f"Error loading zip file {zip_filename}: {e}. Skipping...",
                )
                bad_bins.append(Path(zip_filename).stem)
                continue

            processed_bins[Path(zip_filename).stem] = lmdb_count  # bin is processed, so add it to processed dict
    
    env.close()
    meta_env.close() if meta_env else None

    if tmp_lmdb_path is not None:
        try:
            shutil.move(tmp_lmdb_path, lmdb_path)
        except Exception as e:
            logger.error(f"Error moving lmdb from {tmp_lmdb_path} to {lmdb_path}: {e}")
            pass
        if tmp_meta_lmdb_path is not None:
            try:
                shutil.move(tmp_meta_lmdb_path, meta_lmdb_path)
            except Exception as e:
                logger.error(f"Error moving lmdb from {tmp_meta_lmdb_path} to {meta_lmdb_path}: {e}")
            pass
    logger.info(f"Saved new lmdb at: {lmdb_path}")

    # dump json, since all imgs of processed bins are saved to lmdb
    with lock:
        if not os.path.exists(processed_bins_path):
            with open(processed_bins_path, "w") as f:
                state = {"total_images": 0, "processed_bins": {}, "bad_bins": []}
                json.dump(state, f)
        else:
            with open(processed_bins_path, "r") as f:
                state: dict = json.load(f)
        state["total_images"] = total_images + state.get("total_images", 0)
        state["processed_bins"] = state.get("processed_bins", {}) | processed_bins
        state["bad_bins"] = bad_bins + state.get("bad_bins", [])
        with open(processed_bins_path, "w") as f:
            json.dump(state, f)

def lmdb_worker(
    bin_queue: Queue,
    lock: threading.Lock,
    lmdb_counter: ValueProxy[int],
    lmdb_dir: str,
    processed_bins_path: str,
    chunk_size: int,
    bin_output_dir: str,
    lmdb_counter_lock: threading.Lock,
    tmp_dir: Optional[str] = None,
    queue_timeout = 1200,
    save_metadata: bool = False,
    # worker_configurer
) -> None:
    logger = setup_logging(log_queue)
    while True:
        with lmdb_counter_lock:
            lmdb_counter.value += 1
            lmdb_count = lmdb_counter.value
        logger.info(f"Started building LMDB #{lmdb_count:03}")
        try:
            build_lmdb(bin_queue, lock, lmdb_count, lmdb_dir, processed_bins_path, chunk_size, bin_output_dir, logger, tmp_dir, queue_timeout, save_metadata=save_metadata)
        except Empty:
            logging.info(f"Process finished. Queue was empty for longer than {queue_timeout} seconds")
            break
        except Exception as e:
            logger.error(f"Process crashed. Error building lmdb: {e}")
            break

log_queue = multiprocessing.Queue()

def process_bins(bins: list[Bin], bin_output_dir: str,  lmdb_output_dir: str, num_lmdb_workers: int, chunk_size: int, tmp_dir: Optional[str] = None, queue_timeout: int = 1200, save_metadata: bool = False,  **args) -> None:
    m = multiprocessing.Manager()
    
    bin_queue: Queue[Bin] = m.Queue()
    # add unprocessed bins to queue
    for bin in bins:
        bin_queue.put(bin)
    logging.info(f"Added {len(bins)} unprocessed but already downloaded bins to queue.")

    lock = m.Lock()
    num_existing_lmdbs = len(list(Path(lmdb_output_dir).glob("*_images")))
    logging.info(f"Found {num_existing_lmdbs} existing lmdbs...")
    lmdb_counter = m.Value('i', num_existing_lmdbs)
    lmdb_counter_lock = m.Lock()

    listener_process = multiprocessing.Process(target=listener, args=(log_queue,))
    listener_process.start()

    processed_bins_path = os.path.join(
        lmdb_output_dir, "processed_bins.json"
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_lmdb_workers, initializer=init_log_queue, initargs=(log_queue,)) as process_pool:
        for _ in range(num_lmdb_workers):
            process_pool.submit(lmdb_worker, bin_queue, lock, lmdb_counter, lmdb_output_dir, processed_bins_path, chunk_size, bin_output_dir, lmdb_counter_lock, tmp_dir, queue_timeout, save_metadata)
    log_queue.put(None)
    listener_process.join()


def main(args):
    os.makedirs(args.lmdb_output_dir, exist_ok=True)

    csv_path: Path = args.csv_path if args.csv_path is not None else download_metadata_csv(args.dataset, args.api_path, args.bin_output_dir)

    processed_bins_path = os.path.join(
        args.lmdb_output_dir, "processed_bins.json"
    )
    if os.path.exists(processed_bins_path):
        with open(processed_bins_path, "r") as f:
            processed_bins = json.load(f).get("processed_bins", {})
            processed_bins = list(processed_bins.keys())
    else:
        processed_bins = []

    bins = [bin for bin in get_downloaded_bins(csv_path, args.bin_output_dir, blacklisted_sample_types=args.blacklisted_sample_types.split(","), blacklisted_tags=args.blacklisted_tags.split(",")) if bin.id not in processed_bins]

    # download bins and add path to zipfile in queue
    process_bins(bins=bins, **vars(args))

def get_args_parser():
    parser = argparse.ArgumentParser(parents=[get_csv_args_parser(add_help=False)])

    parser.add_argument(
        "--lmdb_output_dir",
        type=str,
        help="Path to the directory to save the lmdb files.",
        default="/hkfs/work/workspace/scratch/hgf_grc7525-nick/data/lmdb_without_labels/IFCB_downloader",
    )
    parser.add_argument(
        "--tmp_dir",
        type=str,
        help="Path to $TMPDIR to save temporary lmdb files if used",
         default=None,
    )
    parser.add_argument(
        "--num_lmdb_workers", type=int, help="Number of workers (processes) to use for lmdb creation.", default=4
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        help="Size to chunk images into different lmdbs",
        default=2_000_000,
    )
    parser.add_argument(
        "--queue_timeout",
        type=int,
        help="Timeout for queue to be empty before process exits",
        default=1200,
    )
    parser.add_argument(
        "--save_metadata",
        action=argparse.BooleanOptionalAction,
        help="Toggle saving metadata",
        default=False,
    )
    return parser


if __name__ == "__main__":
    args_parser = get_args_parser()
    args = args_parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    sys.exit(main(args))