import argparse
import csv
import logging
import logging.handlers
from pathlib import Path
from dataclasses import dataclass

import requests

@dataclass
class Bin:
    """Represents an IFCB bin with its metadata"""
    id: str
    dataset: str
    sample_time: str
    n_images: int
    latitude: float
    longitude: float
    ml_analyzed: float
    instrument: str
    
# "complete crap" according to mvco csv, even though they don't have the correct tags, so we skip these
blacklisted_mvco_bins = ["D20211123T063204_IFCB010", "D20211123T065325_IFCB010", "D20211123T071429_IFCB010", "D20211123T073533_IFCB010", "D20211123T075637_IFCB010", "D20211123T081741_IFCB010", "D20211123T083845_IFCB010", "D20211123T085949_IFCB010", "D20211123T092053_IFCB010", "D20211123T094157_IFCB010", "D20211123T100301_IFCB010", "D20211123T102405_IFCB010", "D20211123T104509_IFCB010", "D20211123T110613_IFCB010", "D20211123T112717_IFCB010", "D20211123T114821_IFCB010", "D20211123T120925_IFCB010", "D20211123T123029_IFCB010", "D20211123T125133_IFCB010", "D20211123T131237_IFCB010", "D20211123T133341_IFCB010", "D20211123T135445_IFCB010", "D20211123T141550_IFCB010", "D20211123T143654_IFCB010", "D20211123T145758_IFCB010", "D20211123T151902_IFCB010", "D20211123T154007_IFCB010", "D20211123T160111_IFCB010", "D20211123T162215_IFCB010", "D20211123T164320_IFCB010", "D20211123T170424_IFCB010", "D20211123T172528_IFCB010", "D20211123T174632_IFCB010", "D20211123T180736_IFCB010", "D20211123T182840_IFCB010", "D20211123T184944_IFCB010", "D20211123T191048_IFCB010", "D20211123T193153_IFCB010", "D20211123T195257_IFCB010", "D20211123T201401_IFCB010", "D20211123T203505_IFCB010", "D20211123T205610_IFCB010", "D20211123T211714_IFCB010", "D20211123T213818_IFCB010", "D20211123T215922_IFCB010", "D20211123T222026_IFCB010", "D20211123T224131_IFCB010", "D20211123T230235_IFCB010", "D20211123T232338_IFCB010", "D20211123T234442_IFCB010", "D20211124T000546_IFCB010", "D20211124T002650_IFCB010", "D20211124T004754_IFCB010", "D20211124T010858_IFCB010", "D20211124T013002_IFCB010", "D20211124T015106_IFCB010", "D20211124T021210_IFCB010", "D20211124T023314_IFCB010", "D20211124T025418_IFCB010", "D20211124T031522_IFCB010", "D20211124T033626_IFCB010", "D20211124T035730_IFCB010", "D20211124T041834_IFCB010", "D20211124T043938_IFCB010", "D20211124T050042_IFCB010", "D20211124T054310_IFCB010"]

def get_downloaded_bins(csv_path: Path, bins_path: Path, blacklisted_tags: list[str] = [], blacklisted_sample_types: list[str] = []):
    bins = get_filtered_bins(csv_path, blacklisted_tags, blacklisted_sample_types)
    downloaded_bins: list[Bin] = []
    for bin in bins:
        if (bins_path / f"{bin.id}.zip").exists():
            downloaded_bins.append(bin)
    return downloaded_bins

def parse_line(line: dict):
    return Bin(
        id=line["pid"],
        dataset=line["dataset"],
        sample_time=line["sample_time"],
        n_images=int(line["n_images"]),
        latitude=float(line["latitude"]),
        longitude=float(line["longitude"]),
        ml_analyzed=float(line["ml_analyzed"] or 0),
        instrument=f"IFCB{line['ifcb']}"
    )

def get_filtered_bins(csv_path: Path, blacklisted_tags: list[str] = [], blacklisted_sample_types: list[str] = []):
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
                if line["skip"] == "1" or any(tag in blacklisted_tags for tag in all_tags) or line["sample_type"] in blacklisted_sample_types or line["pid"] in blacklisted_mvco_bins or line["ml_analyzed"] == "":
                    blacklisted_bins.append(parse_line(line))
                    continue
                bins.append(parse_line(line))
    total_bins = i # type: ignore
    logging.info(f"Total bins: {total_bins}")
    logging.info(f"Total bins after cleaning: {len(bins)}. Total images: {sum([int(bin.n_images) for bin in bins])}")
    logging.info(f"Total blacklisted bins: {len(blacklisted_bins)}. Total blacklisted images: {sum([bin.n_images for bin in blacklisted_bins])}")
    bins = sorted(bins, key=lambda bin: bin.sample_time)

    return bins

def download_metadata_csv(dataset: str, api_path: str, output_dir: Path):
    url = f"{api_path}/api/export_metadata/{dataset}"
    logging.info(f"Downloading metadata CSV from {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    csv_path = output_dir / f"{dataset}.csv"
    with open(csv_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return csv_path

def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser(add_help=add_help)
    parser.add_argument(
        "--bin_output_dir", type=Path, help="Path to the dataset we download the images to.", default="/hkfs/work/workspace/scratch/hgf_grc7525-nick/data/ifcb"
    )
    parser.add_argument(
        "--api_path", type=str, help="Path to api to download from", default=""
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--csv_path", type=Path, help="Path to the CSV file containing the bin metadata. A new one will be downloaded if not provided.", default=None
    )
    group.add_argument(
        "--dataset", type=str, help="Dataset to download the CSV file/bins from.", default="mvco"
    )
    # Tags in mvco: ['DAQ_error', 'adjust_focus', 'amphidinium grazing', 'auxospore', 'bacillaria', 'bad flow', 'bad_flow', 'bad_focus', 'bad_ml', 'bad_ml_analyzed', 'beads', 'blob', 'cast', 'cerataulina flagellate', 'changing settings', 'changing_settings', 'chl_only', 'chlorox', 'ciliate', 'ciliates', 'cleaning', 'clorox_contaminated', 'coccolithophore', 'computer_error', 'computer_stall_cause_smaller_file', 'daq_error', 'deployed', 'detergent', 'different_view', 'dirt_spot', 'discrete', 'dividing', 'dividing ciliate', 'dividing diatoms', 'dock_water', 'extreme_ypos', 'failing pump', 'grazing', 'guinardia detritus', 'guinardia flaccida', 'incomplete_syringe', 'intake_overgrown', 'less_than_10_rois', 'low_camera_gain', 'monster', 'need refocus', 'none', 'pallium', 'parasite', 'parasites', 'partial_file', 'pheaocystis', 'planktoniella', 'possible_cool_ciliate', 'possible_reagent_damaged', 'post_manual_intake_clorox_flush', 'postbeads', 'pseudonitzschia auxospores', 'pump fail', 'reagent_damaged', 'refocused', 'run_fast', 'sample_volume_less_than_5ml', 'spore', 'ssc_on', 'test', 'thalassionema', 'trigger_SSC', 'trigger_ssc', 'troubleshooting', 'weird', 'weird_cell_orientation', 'what', 'wow', 'zooplankton']
    parser.add_argument(
        "--blacklisted_tags", type=str, help="Tags to blacklist.", default="bad flow,bad_flow,bad_focus,computer_error,daq_error,extreme_ypos,failing pump,incomplete_syringe,monster,partial_file,possible_reagent_damaged,pump fail,reagent_damaged,troubleshooting"
    )
    # Sample types in mvco: ['', 'bad', 'beads', 'cast', 'junk', 'test', 'testwell']
    parser.add_argument(
        "--blacklisted_sample_types", type=str, help="Sample types to blacklist.", default="bad,junk"
    )
    return parser