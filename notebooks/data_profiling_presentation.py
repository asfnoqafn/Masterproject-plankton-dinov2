import math
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import lmdb
import io
import zipfile
import tarfile
from collections import Counter
import sys
import json
from matplotlib.colors import LogNorm
import struct

def get_png_dimensions(data):
    # PNG files start with an 8-byte signature
    png_signature = b'\x89PNG\r\n\x1a\n'
    if data[:8] != png_signature:
        raise ValueError("Not a valid PNG file")

    # IHDR chunk follows immediately after the signature
    # The IHDR chunk contains the width and height as the first two fields
    ihdr_chunk = data[8:33]
    if ihdr_chunk[4:8] != b'IHDR':
        raise ValueError("Missing IHDR chunk in PNG file")

    # Extract width and height from the IHDR chunk
    width, height = struct.unpack(">II", ihdr_chunk[8:16])
    return width, height

def open_and_measure_lmdbs(data_paths: list, max_size, optimized=False):
    n_imgs_per_bin: np.ndarray = np.full((max_size, max_size), fill_value= 0, dtype=int)
    bin_channels = np.ndarray = np.zeros((3, 255), dtype=int)
    n_images_too_large: int = 0
    minmax_dict_global: dict = {} # dict to store minmax values for each path
    # run the following code for each path in data_path
    for data_path in data_paths:
        minmax_dict_tmp: dict[str, int] = {"min_width": sys.maxsize, "max_width": -1, "min_height": sys.maxsize, "max_height": -1, "n_images": 0}
       
        n_images_too_large, minmax_dict_tmp = process_lmdb(data_path, n_imgs_per_bin, max_size, minmax_dict_tmp, bin_channels=bin_channels, optimized=optimized)
        
        print(f"Profiling done for: {data_path}")
        print(minmax_dict_tmp)
        # minmax_dict_global[data_path] = minmax_dict_tmp
        for key, value in minmax_dict_tmp.items():
            if key not in minmax_dict_global:
                minmax_dict_global[key] = 0
            minmax_dict_global[key] = minmax_dict_global[key] + value

    calculate_channel_overview(bin_channels)

    heights = n_imgs_per_bin.sum(axis=0)
    widths = n_imgs_per_bin.sum(axis=1)
    n_images = minmax_dict_global["n_images"]
    assert n_images == np.sum(heights) == np.sum(widths), f"Number of images does not match: {n_images} != {np.sum(heights)} != {np.sum(widths)}"

    h_indices = np.arange(len(heights))
    h_mean = np.sum(h_indices * heights) / n_images
    h_variance = np.sum(((h_indices - h_mean) ** 2) * heights) / n_images
    h_std = np.sqrt(h_variance)

    w_indices = np.arange(len(widths))
    w_mean = np.sum(w_indices * widths) / n_images
    w_variance = np.sum(((w_indices - w_mean) ** 2) * widths) / n_images
    w_std = np.sqrt(w_variance)

    height_stats = {
            'mean': h_mean,
            # 'mean': np.mean(heights),
            'std': h_std,
        }

    width_stats = {
            # 'median': , # np.repeat(np.arange(n_imgs_per_bin.shape[0]), n_imgs_per_bin.astype(int))),
            'mean': w_mean,
            'std': w_std,
        }
    
    print(f"Height stats: \n{height_stats}\n")
    print(f"Width stats: \n{width_stats}\n")
    print(f"Number of images too large: {n_images_too_large}")
    
    return n_imgs_per_bin, minmax_dict_global

def calculate_channel_overview(bin_channels):
    print(bin_channels)
    channel_means = np.zeros(3)
    channel_stds = np.zeros(3)

    for i in range(3):
        channel_means[i] = np.sum(bin_channels[i] * np.arange(255)) / np.sum(bin_channels[i])
        channel_var = np.sum(((np.arange(255) - channel_means[i]) ** 2) * bin_channels[i]) / np.sum(bin_channels[i])
        channel_std = np.sqrt(channel_var)

    print(f"Channel means: {channel_means}")
    print(f"Channel stds: {channel_stds}")

    return channel_means

def process_lmdb(data_path, n_imgs_per_bin, max_size:int, minmax_dict:dict, bin_channels, optimized=False):
    path = os.path.join(data_path)
    env_imgs: lmdb.Environment = lmdb.open(path, readonly=True)

    n_images_too_large = 0
    
    with env_imgs.begin() as txn_imgs:
        cursor_imgs = txn_imgs.cursor()
        
        for (img_key, img_value) in cursor_imgs:
            if optimized:
                img = io.BytesIO(img_value)
                # print(img.getvalue()[0:32])
                # print(Image.open(img).size)
                width, height = get_png_dimensions(img.getvalue())
                # print(width, height)

            else:
                img = Image.open(io.BytesIO(img_value))
                width, height = img.size
                image_array = np.array(img)
                if len(image_array.shape) == 3:  # Color image (RGB or RGBA)
                    mean_per_channel = image_array.mean(axis=(0, 1))
                    for i in range(3):
                        bin_channels[i, int(mean_per_channel[i])] += 1

                else:  # Grayscale image
                    mean_per_channel = image_array.mean()
             

            # print('Image format = ', img.size)

            
            minmax_dict["min_width"] = min(minmax_dict["min_width"], width)
            minmax_dict["max_width"] = max(minmax_dict["max_width"], width)
            minmax_dict["min_height"] = min(minmax_dict["min_height"], height)
            minmax_dict["max_height"] = max(minmax_dict["max_height"], height)
            minmax_dict["n_images"] += 1

            if width < max_size and height < max_size:
               n_imgs_per_bin[ width, height] += 1
            else:
                n_images_too_large += 1
                
    return n_images_too_large, minmax_dict

def open_and_measure_tars(data_paths: list, max_size, optimized=False):
    n_imgs_per_bin: np.ndarray = np.full((max_size, max_size), fill_value= 0, dtype=int)
    n_images_too_large: int = 0
    minmax_dict_global: dict = {} # dict to store minmax values for each path
    # run the following code for each path in data_path
    for data_path in data_paths:
        minmax_dict_tmp: dict[str, int] = {"min_width": sys.maxsize, "max_width": -1, "min_height": sys.maxsize, "max_height": -1, "n_images": 0}
       
        n_images_too_large, minmax_dict_tmp = process_tar_folder(data_path, n_imgs_per_bin, max_size, minmax_dict_tmp, optimized=optimized)
        
        print(f"Profiling done for: {data_path}")
        print(minmax_dict_tmp)
        # minmax_dict_global[data_path] = minmax_dict_tmp
        for key, value in minmax_dict_tmp.items():
            if key not in minmax_dict_global:
                minmax_dict_global[key] = 0
            minmax_dict_global[key] = minmax_dict_global[key] + value

    heights = n_imgs_per_bin.sum(axis=0)
    widths = n_imgs_per_bin.sum(axis=1)

    height_stats = {
            'median': np.median(heights),
            'mean': np.mean(heights),
            'std': np.std(heights),
        }

    width_stats = {
            'median': np.median(widths), # np.repeat(np.arange(n_imgs_per_bin.shape[0]), n_imgs_per_bin.astype(int))),
            'mean': np.mean(widths),
            'std': np.std(widths),
        }
    
    print(f"Height stats: \n{height_stats}\n")
    print(f"Width stats: \n{width_stats}\n")
    print(f"Number of images too large: {n_images_too_large}")
    
    return n_imgs_per_bin, minmax_dict_global

def process_tar_folder(data_path, n_imgs_per_bin, max_size: int, minmax_dict: dict, optimized=False):
    """
    Processes tar files in a given folder structure. For each tar file, it:
      - Prints the tar file path if a CSV file is found inside.
      - Processes PNG images to update image statistics.
    
    Parameters:
      data_path (str): Root folder containing tar files.
      n_imgs_per_bin (dict): A dictionary to bin images by (width, height).
      max_size (int): Maximum size allowed for width and height.
      minmax_dict (dict): Dictionary tracking image dimension minima and maxima, 
                          and the total count of images (keys: "min_width", "max_width", 
                          "min_height", "max_height", "n_images").
      optimized (bool): If True, uses an optimized function to extract PNG dimensions.
    
    Returns:
      n_images_too_large (int): The number of images that exceed the max_size.
      minmax_dict (dict): Updated dictionary with image statistics.
    """
    n_images_too_large = 0

    # Walk through all subdirectories starting from data_path
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.tar'):
                tar_path = os.path.join(root, file)
                csv_found = False  # Flag to ensure we print the tar path only once per tar file
                try:
                    with tarfile.open(tar_path, 'r') as tar:
                        for member in tar.getmembers(): # Iterate over all members in the tar file
                            # Process only regular files.
                            if member.isfile():
                                # Check for CSV files.
                                if member.name.lower().endswith('.csv'):
                                    if not csv_found:
                                        print(f"CSV found in tar: {tar_path}")
                                        csv_found = True
                                    # If needed, you could also process the CSV file here.
                                    continue
                                # Process PNG images.
                                if member.name.lower().endswith('.png'):
                                    fileobj = tar.extractfile(member)
                                    if fileobj is None:
                                        continue
                                    img_bytes = fileobj.read()

                                    # Get image dimensions
                                    if optimized:
                                        # Assumes get_png_dimensions is defined elsewhere
                                        width, height = get_png_dimensions(img_bytes)
                                    else:
                                        img = Image.open(io.BytesIO(img_bytes))
                                        width, height = img.size

                                    # Update min/max and image count statistics
                                    minmax_dict["min_width"] = min(minmax_dict.get("min_width", width), width)
                                    minmax_dict["max_width"] = max(minmax_dict.get("max_width", width), width)
                                    minmax_dict["min_height"] = min(minmax_dict.get("min_height", height), height)
                                    minmax_dict["max_height"] = max(minmax_dict.get("max_height", height), height)
                                    minmax_dict["n_images"] = minmax_dict.get("n_images", 0) + 1

                                    # Bin the image or count as too large
                                    if width < max_size and height < max_size:
                                        n_imgs_per_bin[(width, height)] = n_imgs_per_bin.get((width, height), 0) + 1
                                    else:
                                        n_images_too_large += 1
                except tarfile.TarError as e:
                    print(f"Error processing tar file {tar_path}: {e}")

    return n_images_too_large, minmax_dict


def create_heatmap(widths, heights, num_images=-1, bins=100, folder_name=""):
    plt.hist2d(widths, heights, bins)
    plt.colorbar()
    plt.xlabel('Width (in px)')
    plt.ylabel('Height (in px)')
    plt.title(f'{folder_name} Size Distribution (Log Scale) ({num_images} images)')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

def create_heatmap_array(n_imgs_per_bin, path=os.path.join(os.getcwd(), "output", "heatmap.png"), label="size_distribution"):
    plt.imshow(n_imgs_per_bin,  cmap='viridis', origin='lower', norm=LogNorm())
    plt.colorbar()
    plt.xlabel('Width (in px)')
    plt.ylabel('Height (in px)')
    plt.title(label)
    # plt.yscale('log')
    # plt.xscale('log')

    plt.savefig(path)
    plt.clf()

if __name__ == "__main__":
    # 
    lmdb_path_channel_means = ["/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/lmdb_with_labels/ZooScanNet/images"]
    open_and_measure_lmdbs(lmdb_path_channel_means, max_size=2000, optimized=False)

    # lmdb_paths_unlabeled:list[str] = [
    #     "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/lmdb_without_labels/datasciencebowl/images",
    #     "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/lmdb_without_labels/pisco/images",
    #     # "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/lmdb_without_labels/IFCB_downloader/images",
    #     "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/lmdb_without_labels/seanoe_uvp_unlabeled/images",
    #     "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/ecotaxa_lmdb/UVP5SD/images-unlabeled",
    #     "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/ecotaxa_lmdb/UVP5HD/images-unlabeled",
    #     "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/ecotaxa_lmdb/UVP6/images-unlabeled",
    #     # "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/ecotaxa_lmdb/Zooscan/images-unlabeled",
    #     #"/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/ecotaxa_lmdb/Other scanner/images-unlabeled",
    # ]

    # test_path = ["/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/lmdb_with_labels/datasciencebowl/images"]
    # print(os.getcwd())

    # array_unlabeled, dict_unlabeled = open_and_measure_lmdbs(lmdb_paths_unlabeled, max_size=2000)
    # path_heatmap_unlabeled = "/home/hk-project-p0021769/hgf_col5747/output/heatmap_unlabeled.png"
    # create_heatmap_array(array_unlabeled, path_heatmap_unlabeled, label=f"Size Distribution of unlabeled Images (Log Scale, {dict_unlabeled['n_images']} images)")

    # path_unlabeled_zoomed = "/home/hk-project-p0021769/hgf_col5747/output/heatmap_unlabeled_zoomed.png"
    # create_heatmap_array(array_unlabeled[:300, :300], path_unlabeled_zoomed, label=f"Zoomed Size Distribution of unlabeled Images (Log Scale, {dict_unlabeled['n_images']} images)")

    # labeled data



    # lmdb_paths_labeled:list[str] = [
        # "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/lmdb_with_labels/CPICS/images"
            # "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/lmdb_with_labels/datasciencebowl/images",
            # "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/lmdb_with_labels/FlowCamNet/images",
            # "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/lmdb_with_labels/ISIISNet/images",
            # "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/lmdb_with_labels/seanoe_uvp_labeled/images",
            # "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/lmdb_with_labels/UVPEC/images",
            # "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/lmdb_with_labels/ZooCamNet/images",
            # "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/lmdb_with_labels/ZooScanNet/images",
            # "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/ecotaxa_lmdb/UVP5SD/images-labeled",
            # "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/ecotaxa_lmdb/UVP5HD/images-labeled",
            # "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/ecotaxa_lmdb/Zooscan/images-labeled",
            #"/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/ecotaxa_lmdb/Other scanner/images-labeled",
        # ]
    
    # array_labeled, dict_labeled = open_and_measure_lmdbs(lmdb_paths_labeled, max_size=2000)
    # path_heatmap_labeled: str = "/home/hk-project-p0021769/hgf_col5747/output/heatmap_labeled.png"
    # create_heatmap_array(array_labeled, path_heatmap_labeled, label=f"Size Distribution of Labeled Images (Log Scale, {dict_labeled['n_images']} images)")


    # path_labeled_zoomed: str = "/home/hk-project-p0021769/hgf_col5747/output/heatmap_labeled_zoomed.png"
    # create_heatmap_array(array_labeled[:300, :300], path_labeled_zoomed, label=f"Zoomed Size Distribution of Labeled Images (Log Scale, {dict_labeled['n_images']} images)")


    # ecotaxa = ["/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/ecotaxa_lmdb/UVP5SD/images-labeled",
    #                     "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/ecotaxa_lmdb/UVP5SD/images-unlabeled",
    #                     "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/ecotaxa_lmdb/UVP5HD/images-labeled",
    #                     "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/ecotaxa_lmdb/UVP5HD/images-unlabeled",
    #                     "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/ecotaxa_lmdb/UVP6/images-labeled",
    #                     "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/ecotaxa_lmdb/UVP6/images-unlabeled",
    #                     "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/ecotaxa_lmdb/Zooscan/images-labeled",
    #                     "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/ecotaxa_lmdb/Zooscan/images-unlabeled",
    #                     "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/ecotaxa_lmdb/Other scanner/images-labeled",
    #                     "/home/hk-project-p0021769/hgf_grc7525/workspace/hkfswork/hgf_grc7525-nick/data/ecotaxa_lmdb/Other scanner/images-unlabeled"
    #                     ]
    
    # for path in ecotaxa:
    #     print(path)
    #     array_ecotaxa = open_and_measure_lmdbs([path], max_size=1000, optimized=True)

    path_ankita = [""]

    # array_ankita, dict_ankita = open_and_measure_tars(path_ankita, max_size=2000, optimized=True)
    # create_heatmap_array(array_ankita, path=os.path.join(os.getcwd(), "output", "ankita_heatmap.png"), label="Size Distribution of Ankita's Images (Log Scale)")
