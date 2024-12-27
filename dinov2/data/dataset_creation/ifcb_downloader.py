import argparse
import concurrent.futures
import csv
import json
import logging
import logging.handlers
import multiprocessing
import os
import sys
import threading
from dataclasses import dataclass
from multiprocessing.managers import ValueProxy
from pathlib import Path
from queue import Queue
from typing import Optional
from zipfile import ZipFile

import imageio.v3 as iio
import lmdb
import numpy as np
import requests
from imageio.typing import ImageResource

# Lock to prevent multiple threads from writing to the same file
state_lock = threading.Lock()

@dataclass
class Bin:
    """Represents an IFCB bin with its metadata"""
    sample_time: str
    bin_id: str
    dataset: str
    n_images: int
    
# "complete crap" according to mvco csv, even though they don't have the correct tags, so we skip these
blacklisted_mvco_bins = ["D20211123T063204_IFCB010", "D20211123T065325_IFCB010", "D20211123T071429_IFCB010", "D20211123T073533_IFCB010", "D20211123T075637_IFCB010", "D20211123T081741_IFCB010", "D20211123T083845_IFCB010", "D20211123T085949_IFCB010", "D20211123T092053_IFCB010", "D20211123T094157_IFCB010", "D20211123T100301_IFCB010", "D20211123T102405_IFCB010", "D20211123T104509_IFCB010", "D20211123T110613_IFCB010", "D20211123T112717_IFCB010", "D20211123T114821_IFCB010", "D20211123T120925_IFCB010", "D20211123T123029_IFCB010", "D20211123T125133_IFCB010", "D20211123T131237_IFCB010", "D20211123T133341_IFCB010", "D20211123T135445_IFCB010", "D20211123T141550_IFCB010", "D20211123T143654_IFCB010", "D20211123T145758_IFCB010", "D20211123T151902_IFCB010", "D20211123T154007_IFCB010", "D20211123T160111_IFCB010", "D20211123T162215_IFCB010", "D20211123T164320_IFCB010", "D20211123T170424_IFCB010", "D20211123T172528_IFCB010", "D20211123T174632_IFCB010", "D20211123T180736_IFCB010", "D20211123T182840_IFCB010", "D20211123T184944_IFCB010", "D20211123T191048_IFCB010", "D20211123T193153_IFCB010", "D20211123T195257_IFCB010", "D20211123T201401_IFCB010", "D20211123T203505_IFCB010", "D20211123T205610_IFCB010", "D20211123T211714_IFCB010", "D20211123T213818_IFCB010", "D20211123T215922_IFCB010", "D20211123T222026_IFCB010", "D20211123T224131_IFCB010", "D20211123T230235_IFCB010", "D20211123T232338_IFCB010", "D20211123T234442_IFCB010", "D20211124T000546_IFCB010", "D20211124T002650_IFCB010", "D20211124T004754_IFCB010", "D20211124T010858_IFCB010", "D20211124T013002_IFCB010", "D20211124T015106_IFCB010", "D20211124T021210_IFCB010", "D20211124T023314_IFCB010", "D20211124T025418_IFCB010", "D20211124T031522_IFCB010", "D20211124T033626_IFCB010", "D20211124T035730_IFCB010", "D20211124T041834_IFCB010", "D20211124T043938_IFCB010", "D20211124T050042_IFCB010", "D20211124T054310_IFCB010"]


def download_bin(bin: Bin, api_path: str, output_dir: str, q: Queue[str], include_features=True, include_bin_metadata=True):
    """
    Downloads a ZIP file from a URL and saves it to the specified output directory.

    Args:
        url (str): The API endpoint to download the ZIP file.
        output_dir (str): Directory to save the downloaded ZIP file.
    """
    files_to_download = [".zip"]
    if include_features:
        files_to_download.append("_features.csv")
    if include_bin_metadata:
        files_to_download.append(".hdr")

    failed_files = []

    url = f"{api_path}/{bin.dataset}/{bin.bin_id}"

    for file_to_download in files_to_download:
        try:
            url_and_path = url + file_to_download

            # Extract filename from the URL or headers
            filename = f"{bin.sample_time}_{url_and_path.split('/')[-1]}"
            output_path = os.path.join(output_dir, filename)
            download_path = f"{output_path}.download"

            response = requests.get(url_and_path, stream=True, timeout=60)
            response.raise_for_status()

            # Save the file to disk
            with open(download_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            os.rename(download_path, output_path)
        except requests.HTTPError as e:
            if file_to_download == "_features.csv" and response.status_code == 404:
                # features file is not available for all bins
                continue
            failed_files.append((filename, e))
        except requests.RequestException as e:
            failed_files.append((filename, e))

    zip_failed = len(failed_files) > 0 and '.zip' in failed_files[0][0]

    with state_lock:
        state_path = os.path.join(output_dir, "state.json")
        if not os.path.exists(state_path):
            with open(state_path, "w") as f:
                json.dump({"total_bins": 0, "total_images": 0, "last_bin": "", "failed_bins": []}, f)

        with open(state_path, "r+") as f:
            data = json.load(f)
            if zip_failed:
                data["failed_bins"] = data.get("failed_bins", []) + [bin.bin_id]
            else:
                data["total_bins"] = data.get("total_bins", 0) + 1
                data["total_images"] = data.get("total_images", 0) + bin.n_images
                data["last_bin"] = bin.bin_id
            f.seek(0)
            json.dump(data, f)

    if zip_failed:
        raise failed_files[0][1]
    else:
        q.put(filename)
        return failed_files


# def force_download_bin_list(bin_list: list[str], api_path: str, output_dir, include_features=True, include_bin_metadata=True):
#     for bin in bin_list:
#         download_bin(Bin("", bin, "", 0), api_path, output_dir, include_features, include_bin_metadata)


def get_bin_data(csv_path, output_dir: str, max_bins: Optional[int] = None, start_bin: Optional[str] = None, blacklisted_tags: list[str] = [], blacklisted_sample_types: list[str] = []):
    """
    Get the binary data from a CSV file.

    Args:
        csv (str): Path to the CSV file containing the URLs.

    Returns:
        list: List of URLs from the CSV file.
    """
    bins: list[Bin] = []
    blacklisted_bins: list[Bin] = []
    logging.info(f"Reading CSV file from {csv_path}")
    logging.info(f"Blacklisted tags: {blacklisted_tags}")
    logging.info(f"Blacklisted sample types: {blacklisted_sample_types}")
    with open(csv_path, 'r') as file:
        csv_reader = csv.DictReader(file, delimiter=',', quotechar='"')
        # data cleaning
        for i, line in enumerate(csv_reader):
            # dataset, pid, sample_time, ifcb, ml_analyzed, latitude, longitude, depth, cruise, cast, niskin, sample_type, n_images, tag1, tag2, tag3, tag4, comment_summary, trigger_selection, skip
            if line["dataset"] == "mvco":
                all_tags: list[str] = []
                for tags in [line["tag1"], line["tag2"], line["tag3"], line["tag4"]]:
                    all_tags.extend(tags.split(","))
                all_tags = [tag.strip().lower() for tag in all_tags if tag != ""]
                if line["skip"] == "1" or any(tag in blacklisted_tags for tag in all_tags) or line["sample_type"] in blacklisted_sample_types or line["pid"] in blacklisted_mvco_bins:
                    blacklisted_bins.append(Bin(line["sample_time"], line["pid"], line["dataset"], int(line["n_images"])))
                    continue
                bins.append(Bin(line["sample_time"], line["pid"], line["dataset"], int(line["n_images"])))
    total_bins = i
    logging.info(f"Total bins: {total_bins}")
    logging.info(f"Total bins after cleaning: {len(bins)}. Total images: {sum([int(bin.n_images) for bin in bins])}")
    logging.info(f"Total blacklisted bins: {len(blacklisted_bins)}. Total blacklisted images: {sum([bin.n_images for bin in blacklisted_bins])}")
    bins = sorted(bins, key=lambda bin: bin.sample_time)

    # filter out existing files
    bins = [bin for bin in bins if not os.path.exists(os.path.join(output_dir, f'{bin.bin_id}.zip'))]
    logging.info(f"Total bins left to download: {len(bins)}. Total images left to download: {sum([bin.n_images for bin in bins])}")

    if start_bin is not None:
        logging.info(f"Starting from bin {start_bin}")
        try:
            start_index = next(i for i, bin in enumerate(bins) if bin.bin_id == start_bin)
            bins = bins[start_index:]
        except StopIteration:
            logging.error(f"Bin {start_bin} not found in {csv_path}")
    if max_bins is not None:
        bins = bins[:max_bins]
    return bins


def download_metadata_csv(dataset: str, api_path: str, output_dir: str):
    url = f"{api_path}/api/export_metadata/{dataset}"
    logging.info(f"Downloading metadata CSV from {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    output_path = os.path.join(output_dir, f"{dataset}.csv")
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return output_path


def download_multiple_zips(bins: list[Bin], api_path, output_dir, q: Queue[str], max_workers):
    """
    Downloads multiple ZIP files concurrently from a list of URLs.

    Args:
        urls (list): List of API endpoints to download ZIP files.
        output_dir (str): Directory to save the downloaded ZIP files.
        max_workers (int): Maximum number of threads to use for concurrent downloads.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Use ThreadPoolExecutor for concurrent downloading
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map each URL to the download_zip function
        future_to_bin = {executor.submit(download_bin, bin=bin, api_path=api_path, output_dir=output_dir, q=q): bin for bin in bins}

        # Process completed downloads
        for future in concurrent.futures.as_completed(future_to_bin):
            bin = future_to_bin[future]
            try:
                failed_files = future.result()
                if len(failed_files) == 0:
                    logging.info(f"Finished downloading {bin.bin_id}.")
                else:
                    for filename, e in failed_files:
                        logging.warning(f"Download failed for {filename}: {e}")
            except Exception as e:
                logging.error(f"Error downloading {bin.bin_id}: {e}")
        logging.info(f"Finished downloading {len(bins)} bins.")

# Queue for log messages from processes
log_queue = multiprocessing.Queue()

# Function to setup logging inside each worker


def setup_logging(queue):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create a QueueHandler to send log messages to the queue
    queue_handler = logging.handlers.QueueHandler(queue)
    logger.addHandler(queue_handler)

    return logger


def listener(queue):
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Continuously listen for log messages from the queue
    while True:
        try:
            record = queue.get()
            if record is None:  # Poison pill means shutdown
                break
            logger.handle(record)
        except Exception:
            continue


def download_zips_and_build_lmdbs(bins: list[Bin], api_path: str, output_dir: str, lmdb_dir: str, num_api_workers: int, num_lmdb_workers: int, chunk_size: int):
    m = multiprocessing.Manager()
    bin_queue: Queue[str] = m.Queue()
    lock = m.Lock()
    lmdb_counter = m.Value('i', 0)

    listener_process = multiprocessing.Process(target=listener, args=(log_queue,))
    listener_process.start()

    processed_bins_path = os.path.join(
        lmdb_dir, "processed_bins.json"
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_api_workers) as thread_pool, concurrent.futures.ProcessPoolExecutor(max_workers=num_lmdb_workers) as process_pool:
        future_to_bin = {thread_pool.submit(download_bin, bin, api_path, output_dir, bin_queue): bin for bin in bins}
        process_futures = {process_pool.submit(lmdb_builder, bin_queue, lock, lmdb_counter, log_queue, lmdb_dir, processed_bins_path, chunk_size, output_dir): i for i in range(num_lmdb_workers)}

        for future in concurrent.futures.as_completed(process_futures):
            i = process_futures[future]
            try:
                logging.info(f"Futures {i} completed: {future.result()}")
            except Exception as e:
                logging.error(f"Error building lmdb: {e}")

        # Process completed downloads
        for future in concurrent.futures.as_completed(future_to_bin):
            bin = future_to_bin[future]
            try:
                failed_files = future.result()
                if len(failed_files) == 0:
                    logging.info(f"Finished downloading {bin.bin_id}.")
                else:
                    for filename, e in failed_files:
                        logging.warning(f"Download failed for {filename}: {e}")
            except Exception as e:
                logging.error(f"Error downloading {bin.bin_id}: {e}")
        logging.info(f"Finished downloading {len(bins)} bins.")

    log_queue.put(None)
    listener_process.join()


def test_lmdb_builder():
    m = multiprocessing.Manager()
    bin_queue = m.Queue()
    lock = m.Lock()
    lmdb_counter = m.Value('i', 0)
    lmdb_dir = "/Users/Johann/masterproject/ifcb_lmdb"
    processed_bins_path = os.path.join(
        lmdb_dir, "processed_bins.json"
    )
    bin_output_dir = "/Users/Johann/masterproject/ifcb_zips"
# /Users/Johann/masterproject/ifcb_zips/D20241209T145839_IFCB127.zip
    chunk_size = 10
    for zipfile in ["D20241210T085035_IFCB127.zip", "D20241210T084351_IFCB127.zip", "D20241210T085717_IFCB127.zip"]:
        bin_queue.put(zipfile)
    lmdb_builder(bin_queue, lock, lmdb_counter, log_queue, lmdb_dir, processed_bins_path, chunk_size, bin_output_dir)



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


def lmdb_builder(
    bin_queue: Queue,
    lock: threading.Lock,
    lmdb_counter: ValueProxy[int],
    log_queue: multiprocessing.Queue,
    lmdb_dir: str,
    processed_bins_path: str,
    chunk_size: int,
    bin_output_dir: str
) -> None:
    while True:
        lmdb_counter.value += 1
        logger = setup_logging(log_queue)
        logger.info(f"Started building LMDB #{lmdb_counter.value}")
        build_lmdb(bin_queue, lock, lmdb_counter, lmdb_dir, processed_bins_path, chunk_size, bin_output_dir, logger)


def save_zip_to_lmdb(zip_filename, txn, bin_output_dir):
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

                img_key = os.path.join(zip_filename, image_relpath).replace("/", "_")  # Replace slashes for safety
                img_key_bytes = img_key.encode("utf-8")

                # Load and encode the image
                img_encoded = iio.imwrite(
                    "<bytes>", image, extension=".png"
                )

                # Save to LMDB
                txn.put(img_key_bytes, img_encoded)
                total_images += 1
    return total_images


def build_lmdb(bin_queue: Queue, lock: threading.Lock, lmdb_counter: ValueProxy[int], lmdb_dir: str, processed_bins_path: str, chunk_size: int, bin_output_dir: str, logger: logging.Logger) -> None:
    lmdb_imgs_path = os.path.join(lmdb_dir, f"{lmdb_counter.value}_images")
    total_images = 0
    bad_bins = []
    processed_bins = []
    # open lmdb
    env = lmdb.open(lmdb_imgs_path, map_size=int(1e12))

    with env.begin(write=True) as txn:
        while total_images < chunk_size:
            zip_filename = bin_queue.get(block=True, timeout=10)
            # open next zip-file and write images to lmdb
            try:
                total_images += save_zip_to_lmdb(zip_filename=zip_filename, txn=txn, bin_output_dir=bin_output_dir)
            except Exception as e:
                logger.error(
                    f"Error loading zip file {zip_filename}: {e}. Skipping...",
                )
                bad_bins.append(zip_filename)
                continue

            processed_bins.append(zip_filename)  # bin is processed, so add it to processed list
    env.close()
    logger.info(f"Saved new lmdb at: {lmdb_imgs_path}")

    # dump json, since all imgs of processed bins are saved to lmdb
    with lock:
        with open(processed_bins_path, "r") as f:
            state = json.load(f)
        state["total_images"] += total_images if state.get("total_images") is not None else 0
        state["processed_bins"] += processed_bins if state.get("processed_bins") is not None else []
        state["bad_bins"] += bad_bins if state.get("bad_bins") is not None else []
        with open(processed_bins_path, "w") as f:
            json.dump(state, f)


def main(args):
    os.makedirs(args.bin_output_dir, exist_ok=True)
    os.makedirs(args.lmdb_output_dir, exist_ok=True)

    csv_path = args.csv_path if args.csv_path is not None else download_metadata_csv(args.dataset, args.api_path, args.bin_output_dir)
    bins = get_bin_data(csv_path=csv_path, output_dir=args.bin_output_dir, start_bin=args.start_bin, blacklisted_sample_types=args.blacklisted_sample_types.split(","), blacklisted_tags=args.blacklisted_tags.split(","), max_bins=args.max_bins)  # gets all bins not present in the folder

    # download bins and add path to zipfile in queue
    download_zips_and_build_lmdbs(bins=bins, api_path=args.api_path, output_dir=args.bin_output_dir, lmdb_dir=args.lmdb_output_dir, num_api_workers=args.num_api_workers, num_lmdb_workers=args.num_lmdb_workers, chunk_size=args.chunk_size)


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bin_output_dir", type=str, help="Path to the dataset we download the images to.", default="/Users/Johann/masterproject/ifcb_api"
    )
    parser.add_argument(
        "--lmdb_output_dir",
        type=str,
        help="Path to the directory to save the lmdb files.",
        default="ifcb",
    )
    parser.add_argument(
        "--api_path", type=str, help="Path to api to download from", default="https://ifcb-data.whoi.edu"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--csv_path", type=str, help="Path to the CSV file containing the bin metadata. A new one will be downloaded if not provided.", default=None
    )
    group.add_argument(
        "--dataset", type=str, help="Dataset to download the CSV file/bins from.", default="mvco"
    )
    parser.add_argument(
        "--num_api_workers", type=int, help="Number of workers (threads) to use for concurrent downloads.", default=4
    )
    parser.add_argument("--num_lmdb_workers", type=int, help="Number of workers (processes) to use for lmdb creation.", default=2)
    parser.add_argument(
        "--include_bin_metadata", type=bool, help="Whether to include the bin metadata in the download.", default=False
    )
    parser.add_argument(
        "--include_features", type=bool, help="Whether to include the features file in the download.", default=True
    )
    parser.add_argument(
        "--start_bin", type=str, help="The bin to start downloading from.", default=None)
    parser.add_argument(
        "--max_bins", type=int, help="Maximum number of bins to download.", default=None
    )
    # Tags in mvco: ['DAQ_error', 'adjust_focus', 'amphidinium grazing', 'auxospore', 'bacillaria', 'bad flow', 'bad_flow', 'bad_focus', 'bad_ml', 'bad_ml_analyzed', 'beads', 'blob', 'cast', 'cerataulina flagellate', 'changing settings', 'changing_settings', 'chl_only', 'chlorox', 'ciliate', 'ciliates', 'cleaning', 'clorox_contaminated', 'coccolithophore', 'computer_error', 'computer_stall_cause_smaller_file', 'daq_error', 'deployed', 'detergent', 'different_view', 'dirt_spot', 'discrete', 'dividing', 'dividing ciliate', 'dividing diatoms', 'dock_water', 'extreme_ypos', 'failing pump', 'grazing', 'guinardia detritus', 'guinardia flaccida', 'incomplete_syringe', 'intake_overgrown', 'less_than_10_rois', 'low_camera_gain', 'monster', 'need refocus', 'none', 'pallium', 'parasite', 'parasites', 'partial_file', 'pheaocystis', 'planktoniella', 'possible_cool_ciliate', 'possible_reagent_damaged', 'post_manual_intake_clorox_flush', 'postbeads', 'pseudonitzschia auxospores', 'pump fail', 'reagent_damaged', 'refocused', 'run_fast', 'sample_volume_less_than_5ml', 'spore', 'ssc_on', 'test', 'thalassionema', 'trigger_SSC', 'trigger_ssc', 'troubleshooting', 'weird', 'weird_cell_orientation', 'what', 'wow', 'zooplankton']
    parser.add_argument(
        "--blacklisted_tags", type=str, help="Tags to blacklist.", default="bad flow,bad_flow,bad_focus,computer_error,daq_error,extreme_ypos,failing pump,incomplete_syringe,monster,partial_file,possible_reagent_damaged,pump fail,reagent_damaged,troubleshooting"
    )
    # Sample types in mvco: ['', 'bad', 'beads', 'cast', 'junk', 'test', 'testwell']
    parser.add_argument(
        "--blacklisted_sample_types", type=str, help="Sample types to blacklist.", default="bad,junk"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        help="Size to chunk images into different lmdbs",
        default=10_000_000,
    )
    return parser


if __name__ == "__main__":
    # args_parser = get_args_parser()
    # args = args_parser.parse_args()
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # sys.exit(main(args))
    test_lmdb_builder()
