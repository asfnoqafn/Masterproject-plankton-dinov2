import argparse
import concurrent.futures
import json
import os
import sys
from pathlib import Path
import csv
import threading

import requests


# Lock to prevent multiple threads from writing to the same file
file_lock = threading.Lock()


def download_bin(url, output_dir, include_features=True, include_bin_metadata=True):
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

    for path in files_to_download:
        try:
            url_and_path = f"{url}{path}"
            print(f"Downloading {url_and_path}")

            response = requests.get(url_and_path, stream=True)
            response.raise_for_status()

            # Extract filename from the URL or headers
            filename = url_and_path.split("/")[-1]
            output_path = Path(output_dir) / filename

            # Save the file to disk
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            with file_lock:
                with open(os.path.join(output_dir, "info.json"), "w") as f:
                    json.dump({"last_file": f"{url}{files_to_download[0]}"}, f)

        except requests.RequestException as e:
            failed_files.append((filename, e))
    return failed_files


def get_bin_data(csv_path, output_dir: str, amount: int = 16, start_bin: str = None):
    """
    Get the binary data from a CSV file.

    Args:
        csv (str): Path to the CSV file containing the URLs.

    Returns:
        list: List of URLs from the CSV file.
    """
    bins: list[tuple[str, str]] = []
    with open(csv_path, 'r') as file:
        csv_reader = csv.DictReader(file, delimiter=',', quotechar='"')

        for line in csv_reader:
            # dataset, pid, sample_time, ifcb, ml_analyzed, latitude, longitude, depth, cruise, cast, niskin, sample_type, n_images, tag1, tag2, tag3, tag4, comment_summary, trigger_selection, skip = line_list
            bins.append((line["sample_time"], line["pid"]))
    bins = sorted(bins)

    # filter out existing files
    bins = [bin for bin in bins if not os.path.exists(os.path.join(output_dir, f'{bin[1]}.zip'))]

    if start_bin is not None:
        start_index = next(i for i, (_, bin) in enumerate(bins) if bin == start_bin)
        bins = bins[start_index:]
    return line["dataset"], bins


def download_multiple_zips(urls, output_dir, downloads_per_bin: list[str], max_workers=5):
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
        future_to_url = {executor.submit(download_bin, url, downloads_per_bin, output_dir): url for url in urls}

        # Process completed downloads
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result = future.result()
                if len(result) > 0:
                    for filename, e in result:
                        print(f"Download failed for {filename}: {e}")
            except Exception as e:
                print(f"Error downloading {url}: {e}")


def main(args):
    dataset, bins = get_bin_data(args.csv_path, output_dir=args.output_path, start_bin="D20241213T100522_IFCB127")  # gets all bins not present in the folder
    for bin in bins:
        print(bin)
    # bin_urls = [os.path.join(args.api_path, dataset, bin_id) for bin_id in bins]
    # download_multiple_zips(bin_urls, output_directory, downloads_per_bin=args.downloads_per_bin.split(",") , max_workers=args.num_workers)


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path", type=str, help="Path to the CSV file containing the URLs.", default="/Users/Johann/masterproject/ifcb/mvco.csv"
    )
    parser.add_argument(
        "--output_path", type=str, help="Path to the dataset we download the images to.", default="/Users/Johann/masterproject/ifcb_api"
    )
    parser.add_argument(
        "--api_path", type=str, help="Path to api to download from", default="https://ifcb-data.whoi.edu"
    )
    parser.add_argument(
        "--num_workers", type=int, help="Number of workers to use for concurrent downloads.", default=4
    )
    # "--downloads_per_bin", type=str, help="", default=".zip,.hdr,_features.csv"
    parser.add_argument(
        "--include_bin_metadata", type=bool, help="Whether to include the bin metadata in the download.", default=True
    )
    parser.add_argument(
        "--include_features", type=bool, help="Whether to include the features file in the download.", default=True
    )
    parser.add_argument(
        "--start_bin", type=str, help="The bin to start downloading from.", default=None)
    return parser


if __name__ == "__main__":
    args_parser = get_args_parser()
    args = args_parser.parse_args()

    sys.exit(main(args))

# example usage: !python save_cpics_pngs_to_lmdb.py --dataset_path="/home/nick/Downloads/113201/FlowCamNet/imgs" --lmdb_dir_name="/home/nick/Documents/ws24/lmdb/bigger_imgs/" --min_size=128 --dataset_name="FlowCamNet"
